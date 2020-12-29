import torch
from torch import nn
from pathlib import Path
import pandas as pd
import toml
import pytorch_lightning as pl
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader


class HyperParameters:
    def save_to(self, dest_f: Path):
        data = self.__dict__
        dest_f.write_text(toml.dumps(data))

    def load_from(self, src_f: Path):
        data = toml.loads(src_f.read_text())
        for k in data:
            self.__dict__[k] = data[k]


class NN_HyperParameters(HyperParameters):
    def __init__(
        self,
        layers,
        stop_after: int = None,
        lr: float = None,
        beta1: float = None,
        beta2: float = None,
        weight_decay=None,
        mb_size: int = None,
    ):
        self.layers = layers
        self.stop_after = stop_after
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.mb_size = mb_size


class NN_BinClassifier(pl.LightningModule):
    def __init__(self, NN_HP: NN_HyperParameters):
        super().__init__()

        self.NN_HP = NN_HP
        net_topology = OrderedDict()
        layers = NN_HP.layers
        n_layers = len(NN_HP.layers)

        for i in range(n_layers):
            if i != n_layers - 1:
                net_topology[f"fc{i}"] = torch.nn.Linear(layers[i], layers[i + 1])
                net_topology[f"LeakyReLu{i}"] = torch.nn.LeakyReLU()
            else:
                # last layer (output)
                net_topology[f"fc{i}"] = torch.nn.Linear(layers[i], 1)
                net_topology[f"Sigmoid"] = torch.nn.Sigmoid()

        self.net = torch.nn.Sequential(net_topology)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.NN_HP.lr,
            betas=(self.NN_HP.beta1, self.NN_HP.beta1),
            weight_decay=self.NN_HP.weight_decay,
        )

        self.loss_f = torch.nn.MSELoss()
        self.err_f = torch.nn.L1Loss()

    def forward(self, x):
        out = self.net(x)
        return out


def epoch_minibatches(mb_size, X, Y):
    """Returns minibatches iterator form numpy ndarrays.
    Data is sampled randomly to create minibatches."""

    data_size = X.shape[0]
    permutation = torch.randperm(data_size)
    for i in range(0, data_size, mb_size):
        indices = permutation[i : i + mb_size]
        yield (X[indices], Y[indices])


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
    for epoch in range(n_epochs):
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
