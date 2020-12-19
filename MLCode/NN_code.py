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
        n_epochs: int = None,
        lr: float = None,
        momentum: float = None,
        mb_size: int = None,
    ):
        self.layers = layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.momentum = momentum
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

        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.NN_HP.lr, momentum=self.NN_HP.momentum
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
    for i in range(0,data_size, mb_size):
        indices = permutation[i:i+mb_size]
        yield(X[indices], Y[indices])
        


def train_NN(model, X_train, Y_train, X_val, Y_val):

    X_train = torch.Tensor(X_train)
    Y_train = torch.Tensor(Y_train)
    X_val = torch.Tensor(X_val)
    Y_val = torch.Tensor(Y_val)

    tr_errors = []
    val_errors = []
    losses = []

    mb_size = model.NN_HP.mb_size
    for epoch in range(model.NN_HP.n_epochs):
        tr_minibatches = epoch_minibatches(mb_size, X_train, Y_train)
        epoch_loss = 0.0
        tr_epoch_err = 0.0
        n_minibatches = 0

        for inputs, labels in tr_minibatches:
        # for i in range(X_train.shape[0]):
            # inputs = X_train[i,:]
            # labels = Y_train[i]

            # zero the parameter gradients
            model.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = model.loss_f(outputs, labels)
            loss.backward()
            model.optimizer.step()
            epoch_loss += loss.item()
            tr_epoch_err += model.err_f(outputs, labels).item()

            n_minibatches += 1

        # print epoch statistics
        tr_err = tr_epoch_err/n_minibatches
        tr_errors.append(tr_err)
        val_err = model.err_f(model(X_val), Y_val).item()
        val_errors.append(val_err)
        losses.append(epoch_loss/n_minibatches)
        print(f"epoch {epoch} \t val_err: {val_err :.3f} \t tr_err: {tr_err :.3f}")
        # print(f"epoch {epoch} \t loss: {epoch_loss :.3f}")
    return tr_errors, val_errors, losses
