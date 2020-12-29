import torch
from torch import nn
from collections import OrderedDict


class NN_HyperParameters:
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


class NN_module(nn.Module):
    def __init__(self, NN_HP: NN_HyperParameters):
        super().__init__()

        self.NN_HP = NN_HP
        net_topology = OrderedDict()
        layers = NN_HP.layers
        n_layers = len(NN_HP.layers)

        for i in range(n_layers):
            if i != n_layers - 1:
                net_topology[f"fc{i}"] = nn.Linear(layers[i], layers[i + 1])
                net_topology[f"LeakyReLu{i}"] = nn.LeakyReLU()
            else:
                # last layer (output)
                net_topology[f"fc{i}"] = nn.Linear(layers[i], 1)

        self.net = nn.Sequential(net_topology)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.NN_HP.lr,
            betas=(self.NN_HP.beta1, self.NN_HP.beta1),
            weight_decay=self.NN_HP.weight_decay,
        )

        self.loss_f = nn.MSELoss()
        self.err_f = nn.L1Loss()


def epoch_minibatches(mb_size, X, Y):
    """Returns minibatches iterator form numpy ndarrays.
    Data is sampled randomly to create minibatches."""

    data_size = X.shape[0]
    permutation = torch.randperm(data_size)
    for i in range(0, data_size, mb_size):
        indices = permutation[i : i + mb_size]
        yield (X[indices], Y[indices])