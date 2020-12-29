import torch
from torch import nn
import pandas as pd
from collections import OrderedDict
from .NN import NN_HyperParameters, NN_module, epoch_minibatches


class NN_BinClassifier(NN_module):
    def __init__(self, NN_HP: NN_HyperParameters):
        super().__init__(NN_HP)
        self.outfunc = nn.Sigmoid()

    def forward(self, x):
        out = self.net(x)
        return self.outfunc(out)


def binary_acc(y_pred, y_test):
    y_pred = torch.round(y_pred)
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc.item()


def train_NN_monk(model, X_train, Y_train, X_val, Y_val, n_epochs):

    X_train = torch.Tensor(X_train)
    Y_train = torch.Tensor(Y_train)
    X_val = torch.Tensor(X_val)
    Y_val = torch.Tensor(Y_val)

    tr_errors = []
    val_errors = []
    losses = []
    tr_accuracies = []
    val_accuracies = []

    mb_size = model.NN_HP.mb_size
    for _ in range(n_epochs):
        tr_minibatches = epoch_minibatches(mb_size, X_train, Y_train)
        epoch_loss = 0.0
        tr_epoch_acc = 0.0
        tr_epoch_err = 0.0
        n_minibatches = 0

        for inputs, labels in tr_minibatches:
            model.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = model.loss_f(outputs, labels)
            loss.backward()
            model.optimizer.step()

            # logging
            epoch_loss += loss.item()
            tr_epoch_acc += binary_acc(outputs, labels)
            tr_epoch_err += model.err_f(outputs, labels).item()

            n_minibatches += 1

        # epoch statistics
        tr_acc = tr_epoch_acc / n_minibatches
        tr_accuracies.append(tr_acc)
        tr_err = tr_epoch_err / n_minibatches
        tr_errors.append(tr_err)

        val_acc = binary_acc(model(X_val), Y_val)
        val_accuracies.append(val_acc)
        val_err = model.err_f(model(X_val), Y_val).item()
        val_errors.append(val_err)

        losses.append(epoch_loss / n_minibatches)
        # print(f"epoch {epoch} \t val_err: {val_err :.3f} \t val_acc: {val_acc :.3f}")

    return tr_errors, val_errors, tr_accuracies, val_accuracies, losses
