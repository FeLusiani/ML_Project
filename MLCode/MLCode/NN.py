import torch
from torch import nn
from torch.linalg import norm
from collections import OrderedDict


class NN_HyperParameters:
    def __init__(
        self,
        layers,
        lr: float = None,
        beta1: float = None,
        beta2: float = None,
        weight_decay=None,
        mb_size: int = None,
    ):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.mb_size = mb_size


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class NN_module(nn.Module):
    """Creates a `torch.nn.Module` that takes a `NN_HyperParameters` object
    for initialization. Note that the output layer has no activation function.
    It can be added when defining the `forward` method.
    """

    def __init__(self, out_size: int, NN_HP: NN_HyperParameters):
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
                net_topology[f"fc{i}"] = nn.Linear(layers[i], out_size)

        self.net = nn.Sequential(net_topology)
        self.apply(weights_init)


        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.NN_HP.lr,
            betas=(self.NN_HP.beta1, self.NN_HP.beta1),
            weight_decay=self.NN_HP.weight_decay,
        )

        self.loss_f = nn.MSELoss()


def epoch_minibatches(mb_size, X, Y):
    """Returns minibatches iterator form numpy ndarrays.
    Data is sampled randomly to create minibatches."""

    data_size = X.shape[0]
    permutation = torch.randperm(data_size)
    for i in range(0, data_size, mb_size):
        indices = permutation[i : i + mb_size]
        yield (X[indices], Y[indices])


def MEE(x,y,mean=True):
    """Iteratates along the first axis and
    applies the euclidean distance to each element.
    The resulting values are averaged if `mean=True` (default),
    otherwise summed."""
    assert x.shape == y.shape, 'x and y must have same shape'
    length = x.shape[0]
    r = sum([norm(x[i]-y[i]) for i in range(length)])
    r = r.item()
    return r/length if mean else r