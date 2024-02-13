import os
import sys
import random
import pandas as pd
import numpy as np
import numpy.linalg as la
import shutil
import torch

sys.path.append("../")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from third_party.theory import *
from ConformalizedTS.methods import CAFHT, Split_Conformal, Max_calibrate
from ConformalizedTS.black_box import Blackbox
from ConformalizedTS.networks import MyLSTM
from ConformalizedTS.evals import evaluation, evaluation_multivariate
from ConformalizedTS.utils import split_train_sequence, TSDataset
from experiments.data_gen import data_gen


#########################
# Experiment parameters #
#########################
# if True: # Input parameters
# Parse input arguments
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
if len(sys.argv) != 12:
    print("Error: incorrect number of parameters.")
    quit()

n_train_calib = int(sys.argv[1])
learning_rate = float(sys.argv[2])
n_epoch = int(sys.argv[3])
seed = int(sys.argv[4])
horizon = int(sys.argv[5])
data_model = str(sys.argv[6])
noise_profile = str(sys.argv[7])
ndim = int(sys.argv[8])
noise_level = int(sys.argv[9])
delta = float(sys.argv[10])
delta_test = float(sys.argv[11])

if delta_test == 1: # when varying delta, delta_test = 1 by default. 
  delta_test = delta

# Fixed experiment parameters
num_workers = 0
batch_size = 20
alpha = 0.1 # target miscoverage level
gamma_grid = np.concatenate([np.arange(0.001, 0.1, 0.01), np.arange(0.2, 1.1, 0.1)])

## data splitting and generation
n_train = int(n_train_calib*0.75)
n_calib = int(n_train_calib*0.25)
n_test = 500

phi = [0.9, 0.1, -0.2] # parameters of the autoregressive model
order = len(phi) # order of the autoregressive model
hetero = True
noise_profile = noise_profile

output_len = 1
total_sequence_length = horizon + output_len + order
scaling = True

# model parameters
hidden_size = 128 #number of features in hidden state
num_layers =  4 #number of stacked lstm layers


###############
# Output file #
###############
outdir = "results/simulated_data/"
os.makedirs(outdir, exist_ok=True)
outfile_name = "n"+str(n_train_calib) + "_h" + str(horizon) + "_mod" + str(data_model) + "_prof" + str(noise_profile) + "_seed" + str(seed) + "_ndim" + str(ndim) + "_level" + str(noise_level) + "_del" + str(delta) + "_tdel" + str(delta_test)



outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

modeldir = "models/simulated_data/"+outfile_name
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

data_generator = data_gen(n_train_calib, total_sequence_length, hetero = hetero, seed = seed, delta = delta, ndim = ndim)
if 'AR' in data_model:
  data_train_calib = data_generator.generate_AR(noise_profile = noise_profile)
  ## To explore in the future
  # if data_model == 'AR+Seasonality':
  #   data_train_calib = data_generator.generate_AR_seasonality(5, 25, data = data_train_calib, season_hetero = True)
  # elif data_model == 'AR+Randompeaks':
  #   data_train_calib = data_generator.generate_AR_random_peaks(num_peaks = 5, max_amplitude = 5, data = data_train_calib, peak_hetero = True)
  # elif data_model == 'AR+VolClust':
  #   data_train_calib = data_generator.generate_AR_volClus(max_amplitude = 5, data = data_train_calib, vol_hetero = True)
train_data = data_train_calib[:n_train, order:, :]
calib_data = data_train_calib[n_train:, order:, :]

data_generator = data_gen(n_test, total_sequence_length, hetero = hetero, seed = seed, delta = delta_test, ndim = ndim)
if 'AR' in data_model:
  data_test, test_idx = data_generator.generate_AR(noise_profile = noise_profile, return_index = True)
  ## To explore in the future
  # if data_model == 'AR+Seasonality':
  #   data_test = data_generator.generate_AR_seasonality(5, 25, data = data_test, season_hetero = True)
  # elif data_model == 'AR+Randompeaks':
  #   data_test = data_generator.generate_AR_random_peaks(num_peaks = 5, max_amplitude = 5, data = data_test, peak_hetero = True)
  # elif data_model == 'AR+VolClust':
  #   data_test = data_generator.generate_AR_volClus(max_amplitude = 5, data = data_test, vol_hetero = True)
test_data = data_test[:, order:, :]

hard_idx = test_idx
easy_idx = test_idx == False

y_trim = None
if scaling:
  max_ = np.max(np.abs(train_data))
  train_data = train_data/max_
  calib_data = calib_data/max_
  test_data = test_data/max_

  y_trim = [-1,1]

max_scaling = np.max(train_data, axis = 0)[output_len:]

print('the range of values of the train data is [{},{}]'.format(np.min(train_data), np.max(train_data)))
print('the range of values of the calib data is [{},{}]'.format(np.min(calib_data), np.max(calib_data)))
print('the range of values of the test data is [{},{}]'.format(np.min(test_data), np.max(test_data)))

print('the shape of the train data is {}'.format(train_data.shape))
print('the shape of the calib data is {}'.format(calib_data.shape))
print('the shape of the test data is {}'.format(test_data.shape))


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
  # bbox.net.load_state_dict(torch.load(modeldir+"_"))
else:
  bbox_stats = bbox.full_train(save_dir = modeldir,  model_name = '_')

# plot_loss(bbox_stats['train_loss'], bbox_stats['val_loss'], out_file="plots/"+outfile_prefix+".png")

################
#   Inference  #
################
# making predictions
test_pred, test_true = bbox.predict_iterate(test_loader, horizon = horizon, return_y_true = True, y_trim = y_trim, ndim = ndim)
calib_pred, calib_true = bbox.predict_iterate(calib_loader, horizon = horizon, return_y_true = True, y_trim = y_trim, ndim = ndim)

calib_results = [calib_pred, calib_true]
test_results = [test_pred, test_true]


# for estimating the initial quantile
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
    df['delta_test'] = delta_test
    df['ndim'] = ndim
    return df

results_full = pd.DataFrame()

# Bonferroni benchmark - CFRNN
method = Split_Conformal(alpha = alpha, horizon = horizon, ndim = ndim)
method.calibrate(calib_pred, calib_true)
standard_PI = method.predict(test_pred, y_trim = y_trim)
results_icp = evaluation_multivariate(test_pred, test_true, standard_PI, "CFRNN", data_model, hard_idx, easy_idx)
results_icp = complete_df(results_icp)
results_full = pd.concat([results_full, results_icp])
print('CFRNN method finished..')


# Normalized benchmark - NCTP
method = Max_calibrate(alpha = alpha, horizon = horizon, normalize = max_scaling, ndim = ndim)
method.calibrate(calib_pred, calib_true)
max_PI = method.predict(test_pred, y_trim = y_trim)
results_max = evaluation_multivariate(test_pred, test_true, max_PI, "NCTP", data_model, hard_idx, easy_idx)
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


# New method - CAFHT 
fixed_gamma = None

#### data splitting - PID
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'PID', adaptive = False) #np.arange(0.001, 0.1, 0.01)
DS_PI_PID_fixed = method.predict_bands('data_splitting', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma, y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_ds = evaluation_multivariate(test_pred, test_true, DS_PI_PID_fixed, "CAFHT_fixed_PID_DS", data_model, hard_idx, easy_idx)
results_ds = complete_df(results_ds)
results_full = pd.concat([results_full, results_ds])
print('data splitting fixed PID inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'PID', adaptive = True) #np.arange(0.001, 0.1, 0.01)
DS_PI_PID_adaptive = method.predict_bands('data_splitting', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma, y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_ds = evaluation_multivariate(test_pred, test_true, DS_PI_PID_adaptive, "CAFHT_adaptive_PID_DS", data_model, hard_idx, easy_idx)
results_ds = complete_df(results_ds)
results_full = pd.concat([results_full, results_ds])
print('data splitting adaptive PID inference finished..')

#### data splitting - ACI
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'ACI', adaptive = False) #np.arange(0.001, 0.1, 0.01)
DS_PI_ACI_fixed = method.predict_bands('data_splitting', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma, y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_ds = evaluation_multivariate(test_pred, test_true, DS_PI_ACI_fixed, "CAFHT_fixed_ACI_DS", data_model, hard_idx, easy_idx)
results_ds = complete_df(results_ds)
results_full = pd.concat([results_full, results_ds])
print('data splitting fixed ACI inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'ACI', adaptive = True) #np.arange(0.001, 0.1, 0.01)
DS_PI_ACI_adaptive = method.predict_bands('data_splitting', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma, y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_ds = evaluation_multivariate(test_pred, test_true, DS_PI_ACI_adaptive, "CAFHT_adaptive_ACI_DS", data_model, hard_idx, easy_idx)
results_ds = complete_df(results_ds)
results_full = pd.concat([results_full, results_ds])
print('data splitting adaptive ACI inference finished..')


#### theoretical correction - ACI
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'ACI', adaptive = False)
theory_PI_ACI_fixed = method.predict_bands('theoretical_correction', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_theory = evaluation_multivariate(test_pred, test_true, theory_PI_ACI_fixed, "CAFHT_fixed_ACI_theory", data_model, hard_idx, easy_idx)
results_theory = complete_df(results_theory)
results_full = pd.concat([results_full, results_theory])
print('theoretical correction ACI fixed inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'ACI', adaptive = True)
theory_PI_ACI_adaptive = method.predict_bands('theoretical_correction', calib_results, test_results, q0 = q0,  fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_theory = evaluation_multivariate(test_pred, test_true, theory_PI_ACI_adaptive, "CAFHT_adaptive_ACI_theory", data_model, hard_idx, easy_idx)
results_theory = complete_df(results_theory)
results_full = pd.concat([results_full, results_theory])
print('theoretical correction ACI adaptive inference finished..')

### theoretical correction - PID
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'PID', adaptive = False)
theory_PI_PID_fixed = method.predict_bands('theoretical_correction', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_theory = evaluation_multivariate(test_pred, test_true, theory_PI_PID_fixed, "CAFHT_fixed_PID_theory", data_model, hard_idx, easy_idx)
print(results_theory)
results_theory = complete_df(results_theory)
results_full = pd.concat([results_full, results_theory])
print('theoretical correction PID fixed inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = 'PID', adaptive = True)
theory_PI_PID_adaptive = method.predict_bands('theoretical_correction', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed) #, fixed_gamma = fixed_gamma
results_theory = evaluation_multivariate(test_pred, test_true, theory_PI_PID_adaptive, "CAFHT_adaptive_PID_theory", data_model, hard_idx, easy_idx)
print(results_theory)
results_theory = complete_df(results_theory)
results_full = pd.concat([results_full, results_theory])
print('theoretical correction PID adaptive inference finished..')



#### naive - ACI
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = "ACI", adaptive = False)
naive_PI_ACI_fixed = method.predict_bands('naive', calib_results, test_results, q0 = q0,fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed)
results_naive = evaluation_multivariate(test_pred, test_true, naive_PI_ACI_fixed, "CAFHT_fixed_ACI_naive", data_model, hard_idx, easy_idx)
results_naive = complete_df(results_naive)
results_full = pd.concat([results_full, results_naive])
print('naive ACI fixed inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = "ACI", adaptive = True)
naive_PI_ACI_adaptive = method.predict_bands('naive', calib_results, test_results, q0 = q0, fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed)
results_naive = evaluation_multivariate(test_pred, test_true, naive_PI_ACI_adaptive, "CAFHT_adaptive_ACI_naive", data_model, hard_idx, easy_idx)
results_naive = complete_df(results_naive)
results_full = pd.concat([results_full, results_naive])
print('naive ACI adaptive inference finished..')

#### naive - PID
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = "PID", adaptive = False)
naive_PI_PID_fixed = method.predict_bands('naive', calib_results, test_results, q0 = q0,fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed)
results_naive = evaluation_multivariate(test_pred, test_true, naive_PI_PID_fixed, "CAFHT_fixed_PID_naive", data_model, hard_idx, easy_idx)
results_naive = complete_df(results_naive)
results_full = pd.concat([results_full, results_naive])
print('naive PID fixed inference finished..')
method = CAFHT(alpha = alpha, gamma_grid = gamma_grid, base_model = "PID", adaptive = True)
naive_PI_PID_adaptive = method.predict_bands('naive', calib_results, test_results, q0 = q0,fixed_gamma = fixed_gamma,  y_trim = y_trim, seed = seed)
results_naive = evaluation_multivariate(test_pred, test_true, naive_PI_PID_adaptive, "CAFHT_adaptive_PID_naive", data_model, hard_idx, easy_idx)
results_naive = complete_df(results_naive)
results_full = pd.concat([results_full, results_naive])
print('naive PID adaptive inference finished..')



################
# Save Results #
################
results_full.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()

# Clean up temp model directory to free up disk space
shutil.rmtree(modeldir + "_", ignore_errors=True)