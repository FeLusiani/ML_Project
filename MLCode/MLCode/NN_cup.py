import torch
from torch import nn
import pandas as pd
from collections import OrderedDict
from .NN import NN_HyperParameters, epoch_minibatches
import datetime

def writeOutput(result, name):
    # Name1  Surname1, Name2 Surname2
    # Group Nickname
    # ML-CUP18
    # 02/11/2018
    df = pd.DataFrame(result)
    now = datetime.datetime.now()
    f = open(name, 'w')
    f.write('# Christian Esposito, Federico Lusiani\n')
    f.write('# Cheesleaders\n')
    f.write('# ML-CUP20\n')
    f.write('# '+str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'\n')
    df.index += 1 
    df.to_csv(f, sep=',', encoding='utf-8', header = False)
    f.close()

class NN_BinClassifier(nn.Module):
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