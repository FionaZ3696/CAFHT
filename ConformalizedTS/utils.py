
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset

###################################################################################################################
#                                             Auxilary function                                                   #
###################################################################################################################

colors = [[31, 120, 180], [51, 160, 44], [250,159,181]]
colors = [(r / 255, g / 255, b / 255) for (r, g, b) in colors]

def plot_loss(train_loss, val_loss):
    x = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(x, train_loss, color=colors[0], label="Training loss", linewidth=2)
    plt.plot(x, val_loss, color=colors[1], label="Validation loss", linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the training, validation and test loss")

    plt.show()

def plot_PI_for_single_seq(idx, PI, Y_true, horizon, title_ = 'Standard benchmark Conformal PI for a single time sequence'):
    x_grid = np.arange(horizon)
    
    pred_low = []
    pred_high = []
    for i in range(horizon):
        pred_low.append(PI[idx,:][i][0])
        pred_high.append(PI[idx,:][i][1])

    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, Y_true[idx], marker='o', label='True Values', color='blue')
    plt.fill_between(x_grid, pred_low, pred_high, color='gray', alpha=0.5, label='Conformal prediction Interval')
    plt.xlabel('horizon')
    plt.ylabel('Value')
    plt.title(title_)
    plt.legend()
    plt.grid(True)
    plt.show()


def split_train_sequence(sequences, horizon = 1):
  #X, y = [], []

  X = sequences[:, 0:-horizon]
  y = sequences[:, horizon:]

  #return np.array(X).reshape(-1, shift_band), np.array(y).reshape(-1, 1)
  return np.array(X), np.array(y)

def trimming(y_trim, pred_low, pred_high):
    pred_low = np.maximum(pred_low, y_trim[0])
    pred_high = np.minimum(pred_high,y_trim[1])

    return pred_low, pred_high


class TSDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.X_tensors = Variable(torch.Tensor(X_train))
        self.Y_tensors = Variable(torch.Tensor(Y_train))

    def __getitem__(self, index):
        #X = self.X_tensors[index][..., None]  # (N, L, H) batch, seq_length, feature
        #Y = self.Y_tensors[index][..., None]
        X = self.X_tensors[index]
        Y = self.Y_tensors[index]
        return X, Y

    def __len__(self):
        return self.X_tensors.shape[0]
    

def mytan(x):
    if x >= np.pi/2:
        return np.infty
    elif x <= -np.pi/2:
        return -np.infty
    else:
        return np.tan(x)
def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    out = KI * tan_out
    return  out