
#%%

import numpy as np
from torch import Tensor
from MLCode.utils import load_cup_data, np_cup_TR, plot_NN_TR_VAL
from MLCode.NN import NN_HyperParameters
from MLCode.NN_cup import NN_Regressor, train_NN_cup, save_training


import matplotlib.pyplot as plt

df = load_cup_data()
X, Y = np_cup_TR(df)
X, Y = Tensor(X), Tensor(Y)

n_samples = X.shape[0]
# 20% validation data
val_samples = n_samples // 5

X_train = X[:-val_samples]
Y_train = Y[:-val_samples]

X_val = X[-val_samples:]
Y_val = Y[-val_samples:]


NN_HP = NN_HyperParameters(
    [10, 10, 5],
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0,
    mb_size=1,
)
net = NN_Regressor(2,NN_HP)

stats = train_NN_cup(net, X_train, Y_train, X_val, Y_val,20,300)
save_training(stats, NN_HP)

# %%
