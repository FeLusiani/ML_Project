#%%
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from collections import OrderedDict
from pathlib import Path
from sklearn.model_selection import train_test_split
from MLCode.utils import load_monk_data, np_monk, rescale_bin, plot_NN_TR_TS
from MLCode.NN_code import NN_HyperParameters, NN_BinClassifier, train_NN_monk
import matplotlib.pyplot as plt


df = load_monk_data(1)
X_train, Y_train = np_monk(df, np.float64, np.int32)

df = load_monk_data(1, train=False)
X_test, Y_test = np_monk(df, np.float64, np.int32)

print(X_test.shape, X_train.shape, Y_test.shape, Y_train.shape)

NN_HP = NN_HyperParameters(
    [17, 4],
    stop_after=30,
    lr=0.01,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0,
    mb_size=25,
)
net = NN_BinClassifier(NN_HP)
tr_errors, val_errors, tr_accuracies, val_accuracies, loss = train_NN_monk(
    net, X_train, Y_train, X_test, Y_test, 250
)

plot_NN_TR_TS(tr_errors, val_errors, "MEE")
plt.show()
plot_NN_TR_TS(tr_accuracies, val_accuracies, "accuracy")
plt.show()