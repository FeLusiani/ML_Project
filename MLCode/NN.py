import torch
from torch import nn
from pathlib import Path
import pandas as pd
import toml
import pytorch_lightning as pl
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader


class NN_BinClassifier(pl.LightningModule):

    def __init__(self, layers, NN_HP: NN_HyperParameters, train_data, val_data):
        super().__init__()

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

        X_tr, Y_tr = train_data
        X_val, Y_val = val_data

        X_tr = torch.Tensor(X_tr)
        Y_tr = torch.Tensor(Y_tr)
        X_val = torch.Tensor(X_val)
        Y_val = torch.Tensor(Y_val)

        train_dataset = TensorDataset(X_tr, Y_tr) # create your datset
        tr_dataloader = DataLoader(train_dataset) # create your dataloader

    def forward(self, x):
        out = self.net(x)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        out = self.net(x)
        loss = nn.functional.mse_loss(out, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss
    
    def training_epoch_end(self, training_step_outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        return optimizer
    
    def val_dataloader(self):
        return DataLoader(mnist_val, batch_size=64)

    def test_dataloader(self):
        my_dataset = TensorDataset(X,Y) # create your datset
        my_dataloader = DataLoader(my_dataset) # create your dataloader
    

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



net = NN_BinClassifier([17, 7, 2])
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train_NN(X_train, Y_train, X_val, Y_val, NN_HP: NN_HyperParameters):
    for epoch in range(NN_HP.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i in range(X_train.shape[0]):
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.Tensor(X_train[i : min(i + 10, X_train.shape[0]), :])
            labels = torch.Tensor(
                Y_train[i : min(i + 10, X_train.shape[0])].reshape(-1, 1)
            )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # print epoch statistics

        if i % 1 == 0:  # print every 1 epochs
            print(f"epoch {epoch} \t loss: {running_loss :.3f}")
