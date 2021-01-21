import torch
from torch import nn
from collections import OrderedDict
from MLCode.NN import weights_init


class fixed_NN(nn.Module):
    """A simple nn.Module to use for Extreme Learning.
    It can be initialized specifying the number and size of layers,
    the type of weights initialization and the activation function. 
    """

    def __init__(self, layers: list, a_func=nn.LeakyReLU):
        super().__init__()

        n_layers = len(layers)
        net_topology = OrderedDict()

        for i in range(n_layers-1):
            net_topology[f"fc{i}"] = nn.Linear(layers[i], layers[i + 1])
            net_topology['NonLinear'+str(i)] = a_func()

        self.net = nn.Sequential(net_topology)
        self.apply(weights_init)
    
    
    def forward(self, x):
        return self.net(x)