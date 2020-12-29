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
        n_epochs: int = None,
        lr: float = None,
        momentum: float = None,
        mb_size: int = None,
    ):
        self.n_epochs = n_epochs
        self.lr = lr
        self.momentum = momentum
        self.mb_size = mb_size


class NN_BinClassifier(nn.Module):
    def __init__(self, layers, NN_HP: NN_HyperParameters, train_data, val_data):
        super().__init__()

        self.NN_HP = NN_HP
        net_topology = OrderedDict()
        for i in range(len(layers)):
            if i != len(layers) - 1:
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

        X_tr, Y_tr = train_data
        X_val, Y_val = val_data

        X_tr = torch.Tensor(X_tr)
        Y_tr = torch.Tensor(Y_tr)
        X_val = torch.Tensor(X_val)
        Y_val = torch.Tensor(Y_val)

        tr_dataset = TensorDataset(X_tr, Y_tr)
        self.tr_dl = DataLoader(tr_dataset, batch_size=NN_HP.mb_size)

        val_dataset = TensorDataset(X_val, Y_val)
        self.val_dl = DataLoader(val_dataset, batch_size=NN_HP.mb_size)

        self.loss_f = torch.nn.MSELoss()
        self.loss = 0
        self.err_f = torch.nn.L1Loss()
        self.tr_err = 0
        self.val_err = 0

    def forward(self, x):
        out = self.net(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        # for scalar output, MEE == L1 Loss
        err = self.err_f(out, y)
        loss = self.loss_f(out, y)
        self.tr_err += err
        return loss

    def train_epoch_end(self):
        # Logging to TensorBoard by default
        self.log("train_err", self.tr_err)
        self.tr_err = 0

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        # for scalar output, MEE == L1 Loss
        err = self.err_f(out, y)
        self.val_err += err
        return err

    def validation_epoch_end(self, val_step_outputs):
        # Logging to TensorBoard by default
        self.log("val_err", sum(val_step_outputs) / len(val_step_outputs))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)

        return optimizer

    def train_dataloader(self):
        return self.tr_dl

    def val_dataloader(self):
        return self.val_dl