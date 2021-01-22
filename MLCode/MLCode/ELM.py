import torch
from torch import nn
from collections import OrderedDict
from MLCode.NN import weights_init
from sklearn import linear_model
from MLCode.conf import path_data
from pathlib import Path
import pandas as pd


class ExtremeLearningMachine(nn.Module):
    """Extreme Learning Machine.
    Initialization takes as arguments the number and size of layers,
    the alpha parameters for the Ridge regression,
    and the activation function.
    """

    def __init__(self, layers: list, alpha=0.1, a_func='LeakyRelu'):
        super().__init__()

        self.out_layer = linear_model.Ridge(alpha)

        n_layers = len(layers)
        net_topology = OrderedDict()
        if a_func == 'LeakyRelu':
            a_func = nn.LeakyReLU
        elif a_func == 'Sigmoid':
            a_func = nn.Sigmoid

        for i in range(n_layers-1):
            net_topology[f"fc{i}"] = nn.Linear(layers[i], layers[i + 1])
            net_topology['NonLinear'+str(i)] = a_func()

        self.net = nn.Sequential(net_topology)
    
    
    def fit(self, X, y):
        self.apply(weights_init)

        X = torch.Tensor(X)
        X = self.net(X)
        #X = X.to_numpy()

        self.out_layer.fit(X, y)
    

    def predict(self, X):
        X = torch.Tensor(X)
        X = self.net(X)
        #X = X.to_numpy()
        return self.out_layer.predict(X)


# save and load results
saved_csv = path_data / Path('ELM_results.csv')

def load_results():
    if saved_csv.exists():
        return pd.read_csv(saved_csv)
    else:
        columns=['n_layers', 'size', 'alpha', 'activation_f', 'MEE_mean', 'MEE_std', 'seconds']
        return pd.DataFrame(columns=columns)


def save_results(df):
    saved_df = load_results()
    saved_df = saved_df.append(df, ignore_index=True)
    saved_df = saved_df.drop_duplicates(['n_layers', 'size', 'alpha', 'activation_f'], keep='last')
    saved_df.to_csv(saved_csv, index=False)