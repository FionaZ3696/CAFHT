import os
import sys
import random
import pandas as pd
import numpy as np
import numpy.linalg as la
import shutil
import torch
import pickle

sys.path.append("../")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from third_party.theory import *
from ConformalizedTS.methods import CAFHT, Split_Conformal, Max_calibrate
from ConformalizedTS.black_box import Blackbox
from ConformalizedTS.networks import MyLSTM
from ConformalizedTS.evals import evaluation, evaluation_multivariate
from ConformalizedTS.utils import split_train_sequence, TSDataset


#########################
# Experiment parameters #
#########################
# if True: # Input parameters
# Parse input arguments
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
if len(sys.argv) != 7:
    print("Error: incorrect number of parameters.")
    quit()

n_train_calib = int(sys.argv[1])
learning_rate = float(sys.argv[2])
n_epoch = int(sys.argv[3])
seed = int(sys.argv[4])
noise_profile = str(sys.argv[5])
noise_level = int(sys.argv[6])


# Fixed experiment parameters
num_workers = 0
batch_size = 20
alpha = 0.1 # target miscoverage level
gamma_grid = np.concatenate([np.arange(0.001, 0.1, 0.01), np.arange(0.2, 1.1, 0.1)])

## data splitting and generation
n_train = int(n_train_calib*0.75)
n_calib = int(n_train_calib*0.25)
n_test = 291


hetero = True
delta = 0.1

output_len = 1
scaling = True
p_len = 20
horizon = p_len - 1
ndim = 2
y_trim = None



random.seed(seed)

# model parameters
hidden_size = 128 #number of features in hidden state
num_layers =  4 #number of stacked lstm layers


###############
# Output file #
###############
save_path = 'realdata/pedestrian_data/'
outdir = "results/real_data/"
os.makedirs(outdir, exist_ok=True)
outfile_name = "n"+str(n_train_calib) + "_modped" + "_prof" + str(noise_profile) + "_seed" + str(seed) + "_ndim" + str(ndim) + "_level" + str(noise_level)

outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

modeldir = "models/real_data/"+outfile_name
print(modeldir)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    # Make CuDNN Determinist
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Is CUDA available? {}".format(torch.cuda.is_available()))

    
#################
# Download/Simulate Data #
#################

fid = open(save_path + "x_value.pkl", 'rb')
x_value = pickle.load(fid)
fid.close()
fid = open(save_path + "y_value.pkl", 'rb')
y_value = pickle.load(fid)
fid.close()

dict_keys_x = list(x_value.keys())
dict_key_x = dict_keys_x[0]
dict_keys_y = list(y_value.keys())
dict_key_y = dict_keys_y[0]

keys_x = list(x_value[dict_key_x].keys())
all_x = []
all_y = []
for k in keys_x:
    final_key = list(x_value[dict_key_x][k].keys())[0]
    x = x_value[dict_key_x][k][final_key]
    y = y_value[dict_key_y][k][final_key]
    all_x.append(x[p_len:2*p_len])
    all_y.append(y[p_len:2*p_len])

## create train/test split
temp = list(zip(all_x, all_y))
random.shuffle(temp)
res1,res2 = zip(*temp)
all_x,all_y = list(res1),list(res2)
all_x = np.array(all_x)
all_y = np.array(all_y)

n_total = all_x.shape[0]
horizon = all_x.shape[1] - 1
hetero_scaling_param = 200

if hetero:
  u = np.random.uniform(0,1, n_total)
  for i in range(horizon):
    all_x[u <= delta, i] += np.random.normal(0, i*noise_level/hetero_scaling_param, sum(u <= delta))
    all_x[u > delta, i] += np.random.normal(0, i/hetero_scaling_param, sum(u > delta))
    all_y[u <= delta, i] += np.random.normal(0, i*noise_level/hetero_scaling_param, sum(u <= delta))
    all_y[u > delta, i] += np.random.normal(0, i/hetero_scaling_param, sum(u > delta))
    
scaling_param = np.maximum(np.max(np.abs(all_x)), np.max(np.abs(all_y))) # or any constant upper bound
if scaling:
  all_x = all_x/scaling_param
  all_y = all_y/scaling_param

  y_trim = [-1, 1]

train_calib_data = np.dstack((all_x[:-n_test], all_y[:-n_test]))
test_data = np.dstack((all_x[-n_test:], all_y[-n_test:]))
train_data = train_calib_data[:n_train, :, :]
calib_data = train_calib_data[n_train: n_train_calib, :, :]

hard_idx = u[-n_test:] <= delta
easy_idx = u[-n_test:] > delta

max_scaling = np.max(train_data, axis = 0)[output_len:]
print('the range of values of the train data is [{},{}]'.format(np.min(train_data), np.max(train_data)))
print('the range of values of the calib data is [{},{}]'.format(np.min(calib_data), np.max(calib_data)))
print('the range of values of the test data is [{},{}]'.format(np.min(test_data), np.max(test_data)))

print('the shape of values of the train data is {}'.format(train_data.shape))
print('the shape of values of the calib data is {}'.format(calib_data.shape))
print('the shape of values of the test data is {}'.format(test_data.shape))

################################ First do the iterative training & predictions ################################
################
#  Data Split  #
################
X_train, Y_train = split_train_sequence(train_data, output_len)

train_dataset = TSDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, drop_last=False, shuffle=True)

X_calib, Y_calib = split_train_sequence(calib_data, output_len)

calib_dataset = TSDataset(X_calib, Y_calib)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size = 1, drop_last=False, shuffle=False)

X_test, Y_test = split_train_sequence(test_data, output_len)
test_dataset = TSDataset(X_test, Y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, drop_last=False, shuffle=False)

print("Training Shape", X_train.shape, Y_train.shape)
print("Calibration Shape", X_calib.shape, Y_calib.shape)
print("Testing Shape", X_test.shape, Y_test.shape)

################
#   Training   #
################

lstm = MyLSTM(input_size=ndim, hidden_size=hidden_size, num_layers=num_layers, output_size=ndim)
criterion = torch.nn.MSELoss()    # mean-squared error for regression
# optimizer = torch.optim.SGD(lstm.parameters(), momentum=0.95, lr=learning_rate, nesterov=True)
optimizer = torch.optim.AdamW(lstm.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training the model
bbox = Blackbox(lstm, device, train_loader, batch_size=batch_size, max_epoch=n_epoch,
                learning_rate=learning_rate, val_loader=calib_loader, criterion=criterion, optimizer=optimizer, verbose=True)
if os.path.isfile(modeldir+"_"):
  print('loading existing model..')
  saved_stats = torch.load(modeldir+"_", map_location=device)
  bbox.net.load_state_dict(saved_stats['model_state'])
else:
  bbox_stats = bbox.full_train(save_dir = modeldir,  model_name = '_')
  
################
#   Inference  #
################
# making predictions
test_pred, test_true = bbox.predict_iterate(test_loader, horizon = horizon , return_y_true = True, y_trim = y_trim, ndim = ndim)
calib_pred, calib_true = bbox.predict_iterate(calib_loader, horizon = horizon, return_y_true = True, y_trim = y_trim, ndim = ndim)

calib_results = [calib_pred, calib_true]
test_results = [test_pred, test_true]

train_loader_2 = torch.utils.data.DataLoader(train_dataset, batch_size = 1, drop_last=False, shuffle=False)
train_pred, train_true = bbox.predict_iterate(train_loader_2, horizon = horizon, return_y_true = True, y_trim = y_trim, ndim = ndim)
train_qts = []
for k in range(train_pred.shape[0]):
  score_ = la.norm(train_pred[k] - train_true[k], np.inf, axis = 1)
  qt_ = np.quantile(score_, 1-alpha, interpolation='higher')
  train_qts.append(qt_)
  
q0 = np.mean(train_qts)
  
def complete_df(df):
    df["n_data"] = n_train_calib
    df["seed"] = seed
    df["lr"] = learning_rate
    df["batch_size"] = batch_size
    df['alpha'] = alpha 
    df['horizon'] = horizon
    df['noise_profile'] = noise_profile
    df['noise_level'] = noise_level
    df['delta'] = delta
    df['ndim'] = ndim
    return df

results_full = pd.DataFrame()

# Standard method
method = Split_Conformal(alpha = alpha, horizon = horizon, ndim = ndim)
method.calibrate(calib_pred, calib_true)
standard_PI = method.predict(test_pred, y_trim = y_trim)
results_icp = evaluation_multivariate(test_pred, test_true, standard_PI, "CFRNN", 'pedestrian_data', hard_idx, easy_idx)
results_icp = complete_df(results_icp)
results_full = pd.concat([results_full, results_icp])
print('CFRNN method finished..')

## Maximum method
method = Max_calibrate(alpha = alpha, horizon = horizon, normalize = max_scaling, ndim = ndim)
method.calibrate(calib_pred, calib_true)
max_PI = method.predict(test_pred, y_trim = y_trim)
results_max = evaluation_multivariate(test_pred, test_true, max_PI, "NCTP", 'pedestrian_data', hard_idx, easy_idx)
results_max = complete_df(results_max)
results_full = pd.concat([results_full, results_max])
print('NCTP method finished..')

# # ACI
# method = Adaptive_Conformal_Inference(bbox, alpha = alpha, horizon = horizon)
# PI_ACI = method.predict_intervals(test_pred, test_true, gamma = 0.1)
# results_ACI = evaluation(test_pred, test_true, PI_ACI, "ACI", data_model, K = 2, band_width = 10)
# results_ACI = complete_df(results_ACI)
# results_full = pd.concat([results_full, results_ACI])
# print('ACI inference finished..')


# New method - CFHT 
fixed_gamma = None

#### data splitting - PID
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'PID', adaptive = False) #np.arange(0.001, 0.1, 0.01)
DS_PI_PID_fixed = method.predict_bands('data_splitting', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma, y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_ds = evaluation_multivariate(test_pred, test_true, DS_PI_PID_fixed, "CAFHT_fixed_PID_DS", 'pedestrian_data', hard_idx, easy_idx)
results_ds = complete_df(results_ds)
results_full = pd.concat([results_full, results_ds])
print('data splitting fixed PID inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'PID', adaptive = True) #np.arange(0.001, 0.1, 0.01)
DS_PI_PID_adaptive = method.predict_bands('data_splitting', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma, y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_ds = evaluation_multivariate(test_pred, test_true, DS_PI_PID_adaptive, "CAFHT_adaptive_PID_DS", 'pedestrian_data', hard_idx, easy_idx)
results_ds = complete_df(results_ds)
results_full = pd.concat([results_full, results_ds])
print('data splitting adaptive PID inference finished..')

#### data splitting - ACI
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'ACI', adaptive = False) #np.arange(0.001, 0.1, 0.01)
DS_PI_ACI_fixed = method.predict_bands('data_splitting', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma, y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_ds = evaluation_multivariate(test_pred, test_true, DS_PI_ACI_fixed, "CAFHT_fixed_ACI_DS", 'pedestrian_data', hard_idx, easy_idx)
results_ds = complete_df(results_ds)
results_full = pd.concat([results_full, results_ds])
print('data splitting fixed ACI inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'ACI', adaptive = True) #np.arange(0.001, 0.1, 0.01)
DS_PI_ACI_adaptive = method.predict_bands('data_splitting', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma, y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_ds = evaluation_multivariate(test_pred, test_true, DS_PI_ACI_adaptive, "CAFHT_adaptive_ACI_DS", 'pedestrian_data', hard_idx, easy_idx)
results_ds = complete_df(results_ds)
results_full = pd.concat([results_full, results_ds])
print('data splitting adaptive ACI inference finished..')


#### theoretical correction - ACI
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'ACI', adaptive = False)
theory_PI_ACI_fixed = method.predict_bands('theoretical_correction', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_theory = evaluation_multivariate(test_pred, test_true, theory_PI_ACI_fixed, "CAFHT_fixed_ACI_theory", 'pedestrian_data', hard_idx, easy_idx)
results_theory = complete_df(results_theory)
results_full = pd.concat([results_full, results_theory])
print('theoretical correction ACI fixed inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'ACI', adaptive = True)
theory_PI_ACI_adaptive = method.predict_bands('theoretical_correction', calib_results, test_results, q0 = q0,  fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_theory = evaluation_multivariate(test_pred, test_true, theory_PI_ACI_adaptive, "CAFHT_adaptive_ACI_theory", 'pedestrian_data', hard_idx, easy_idx)
results_theory = complete_df(results_theory)
results_full = pd.concat([results_full, results_theory])
print('theoretical correction ACI adaptive inference finished..')

### theoretical correction - PID
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'PID', adaptive = False)
theory_PI_PID_fixed = method.predict_bands('theoretical_correction', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_theory = evaluation_multivariate(test_pred, test_true, theory_PI_PID_fixed, "CAFHT_fixed_PID_theory", 'pedestrian_data', hard_idx, easy_idx)
print(results_theory)
results_theory = complete_df(results_theory)
results_full = pd.concat([results_full, results_theory])
print('theoretical correction PID fixed inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'PID', adaptive = True)
theory_PI_PID_adaptive = method.predict_bands('theoretical_correction', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_theory = evaluation_multivariate(test_pred, test_true, theory_PI_PID_adaptive, "CAFHT_adaptive_PID_theory", 'pedestrian_data', hard_idx, easy_idx)
print(results_theory)
results_theory = complete_df(results_theory)
results_full = pd.concat([results_full, results_theory])
print('theoretical correction PID adaptive inference finished..')



#### naive - ACI
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = "ACI", adaptive = False)
naive_PI_ACI_fixed = method.predict_bands('naive', calib_results, test_results, q0 = q0,fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed)
results_naive = evaluation_multivariate(test_pred, test_true, naive_PI_ACI_fixed, "CAFHT_fixed_ACI_naive", 'pedestrian_data', hard_idx, easy_idx)
results_naive = complete_df(results_naive)
results_full = pd.concat([results_full, results_naive])
print('naive ACI fixed inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = "ACI", adaptive = True)
naive_PI_ACI_adaptive = method.predict_bands('naive', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed)
results_naive = evaluation_multivariate(test_pred, test_true, naive_PI_ACI_adaptive, "CAFHT_adaptive_ACI_naive", 'pedestrian_data', hard_idx, easy_idx)
results_naive = complete_df(results_naive)
results_full = pd.concat([results_full, results_naive])
print('naive ACI adaptive inference finished..')

#### naive - PID
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = "PID", adaptive = False)
naive_PI_PID_fixed = method.predict_bands('naive', calib_results, test_results, q0 = q0,fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed)
results_naive = evaluation_multivariate(test_pred, test_true, naive_PI_PID_fixed, "CAFHT_fixed_PID_naive", 'pedestrian_data', hard_idx, easy_idx)
results_naive = complete_df(results_naive)
results_full = pd.concat([results_full, results_naive])
print('naive PID fixed inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = "PID", adaptive = True)
naive_PI_PID_adaptive = method.predict_bands('naive', calib_results, test_results, q0 = q0,fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed)
results_naive = evaluation_multivariate(test_pred, test_true, naive_PI_PID_adaptive, "CAFHT_adaptive_PID_naive", 'pedestrian_data', hard_idx, easy_idx)
results_naive = complete_df(results_naive)
results_full = pd.concat([results_full, results_naive])
print('naive PID adaptive inference finished..')


# plot_loss(bbox_stats['train_loss'], bbox_stats['val_loss'], out_file="plots/"+outfile_prefix+".png")









################
# Save Results #
################
results_full.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()

# Clean up temp model directory to free up disk space
shutil.rmtree(modeldir + "_", ignore_errors=True)