
#%%
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from collections import OrderedDict
from pathlib import Path
from sklearn.model_selection import train_test_split
from MLCode.utils import load_monk_data, np_monk, rescale_bin
from MLCode.NN_code import NN_HyperParameters, NN_BinClassifier, train_NN
import matplotlib.pyplot as plt

# from pytorch_lightning import Trainer


df = load_monk_data(1)
X_train, Y_train = np_monk(df, np.float64, np.float64)

df = load_monk_data(1, train=False)
X_test, Y_test = np_monk(df, np.float64, np.float64)


NN_HP = NN_HyperParameters([17,12,7], n_epochs=100,lr=0.3,momentum=0.9,mb_size=25)
net = NN_BinClassifier(NN_HP)
tr_errors, val_errors, loss = train_NN(net,X_train,Y_train,X_test,Y_test)

_, ax = plt.subplots(figsize=(16,8))

x = range(len(tr_errors))

ax.plot(tr_errors, label='TR error')
ax.plot(val_errors, label='VAL error')
ax.plot(loss, label='TR loss')

ax.legend()

plt.ylabel('Traning error per epoch')
plt.show()
# %%
