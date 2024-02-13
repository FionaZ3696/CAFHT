
import pathlib
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


###################################################################################################################
#                                                                    Blackbox, Networks                                                              #
###################################################################################################################

class Blackbox:
    def __init__(self, net, device, train_loader, batch_size, max_epoch, learning_rate, criterion, optimizer,
                 val_loader = None, verbose = True):
        self.net = net.to(device)
        self.device = device
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_loader = val_loader

        self.ID = np.random.randint(0, high=2**31)

        if self.verbose:
            print("===== HYPERPARAMETERS =====")
            print("batch_size=", self.batch_size)
            print("n_epochs=", self.max_epoch)
            print("learning_rate=", self.learning_rate)
            print("=" * 30)

    def train_single_epoch(self):
        """
        Train the model for a single epoch
        :return
        """
        single_train_loss = 0

        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            # if len(outputs.shape) == 2:
            #   outputs = outputs.unsqueeze(-1)
            # print(outputs.shape, targets.shape)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            single_train_loss += loss.item()

        single_train_loss /= len(self.train_loader)

        return single_train_loss

    def full_train(self, save_dir = './models', model_name = None):
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        if self.val_loader is not None:
            stats = {'epoch': [], 'train_loss':[], 'val_loss':[]}
        else:
            stats = {'epoch': [], 'train_loss':[]}

        print("Begin training.")

        for e in tqdm(range(1, self.max_epoch+1)):
            epoch_train_loss = 0
            epoch_val_loss = 0

            self.net.train()
            epoch_train_loss = self.train_single_epoch()
            # scheduler.step()

            if self.val_loader is not None:
                self.net.eval()
                for inputs, targets in self.val_loader:
                  inputs, targets = inputs.to(self.device), targets.to(self.device)
                  outputs = self.net(inputs)
                  val_loss = self.criterion(outputs, targets)
                  epoch_val_loss += val_loss.item()

                epoch_val_loss /= len(self.val_loader)
                stats['epoch'].append(e)
                stats['train_loss'].append(epoch_train_loss)
                stats['val_loss'].append(epoch_val_loss)
                if self.verbose:
                  print(f'Epoch {e+0:03}: | train_loss: {epoch_train_loss:.3f} | ', end = '')
                  print(f'val_loss: {epoch_val_loss:.3f} | ', end = '')
                  print('', flush = True)
            else:
                if self.verbose:
                  print(f'Epoch {e+0:03}: | train_loss: {epoch_train_loss:.3f} | ', end = '')
                  print('', flush = True)


        saved_final_state = dict(stats=stats, model_state=self.net.state_dict())
        torch.save(saved_final_state, save_dir + model_name)
        return stats


    def predict_single(self, test_loader, horizon, return_y_true = True):

        y_pred = []
        y_true = []

        with torch.no_grad():
            self.net.eval()
            for inputs, targets in test_loader:
                assert horizon < targets.shape[1], 'Prediction horizon longer than total length'

                inputs = inputs.to(self.device)
                y_pred_ = self.net(inputs)[:, -horizon:, :]
                y_pred.append(y_pred_.detach().cpu().numpy().flatten())
                y_true_ = targets[:, -horizon:, :]
                y_true.append(y_true_.data.numpy().flatten())


        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        if return_y_true:
            return (y_pred, y_true)
        else:
            return y_pred

    def predict_iterate(self, test_loader, horizon, return_y_true = True, y_trim = None, ndim = 1):

        n_test = test_loader.__len__()
        y_pred = np.empty((n_test, horizon, ndim))
        y_true = np.empty((n_test, horizon, ndim))

        if y_trim:
            assert isinstance(y_trim, list) and len(y_trim) == 2, "Need to input y_trim as a list with two elements"

        with torch.no_grad():
            self.net.eval()

            for idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = self.net(inputs)
                pred = output[0].cpu().numpy()
                true = targets[0].cpu().numpy()
                if y_trim:
                    pred = np.clip(pred, y_trim[0], y_trim[1])
                y_pred[idx, :, :] = pred
                y_true[idx, :, :] = true


        if return_y_true:
            return y_pred, y_true
        else:
            return y_pred

