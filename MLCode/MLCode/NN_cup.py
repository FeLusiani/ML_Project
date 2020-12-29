import torch
from torch import nn
import pandas as pd
from collections import OrderedDict
from .NN import NN_HyperParameters, epoch_minibatches, NN_module
import datetime


def writeOutput(result, name):
    # Name1  Surname1, Name2 Surname2
    # Group Nickname
    # ML-CUP18
    # 02/11/2018
    df = pd.DataFrame(result)
    now = datetime.datetime.now()
    f = open(name, "w")
    f.write("# Christian Esposito, Federico Lusiani\n")
    f.write("# Cheesleaders\n")
    f.write("# ML-CUP20\n")
    f.write("# " + str(now.day) + "/" + str(now.month) + "/" + str(now.year) + "\n")
    df.index += 1
    df.to_csv(f, sep=",", encoding="utf-8", header=False)
    f.close()


class NN_Regressor(NN_module):
    def forward(self, x):
        out = self.net(x)
        return out


def train_NN_cup(model, X_train, Y_train, X_val, Y_val, stop_after=20, max_epochs=1000, logging=True):

    X_train = torch.Tensor(X_train)
    Y_train = torch.Tensor(Y_train)
    X_val = torch.Tensor(X_val)
    Y_val = torch.Tensor(Y_val)

    tr_errors = []
    val_errors = []
    losses = []

    lowest_val_err = float('inf')
    since_lowest = 0

    mb_size = model.NN_HP.mb_size
    n_minibatches = X_train.shape[0] // mb_size

    for _ in range(max_epochs):
        tr_minibatches = epoch_minibatches(mb_size, X_train, Y_train)
        epoch_loss = 0.0
        tr_epoch_err = 0.0

        for inputs, labels in tr_minibatches:
            model.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = model.loss_f(outputs, labels)
            loss.backward()
            model.optimizer.step()

            # logging is made optional for performance
            if logging:
                epoch_loss += loss.item()
                tr_epoch_err += model.err_f(outputs, labels).item()

        # epoch statistics
        if logging:
            tr_err = tr_epoch_err / n_minibatches
            tr_errors.append(tr_err)
            losses.append(epoch_loss / n_minibatches)

        val_err = model.err_f(model(X_val), Y_val).item()
        val_errors.append(val_err)

        # early stopping implementation
        if val_err < lowest_val_err:
            lowest_val_err = val_err
            since_lowest = 0
        else:
            since_lowest += 1
        
        if since_lowest >= stop_after: break


    return tr_errors, val_errors, losses