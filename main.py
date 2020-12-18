import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from collections import OrderedDict
from pathlib import Path
from sklearn.model_selection import train_test_split
from MLCode.utils import load_monk_data, preprocess_monk


df = load_monk_data(1)
X, Y = preprocess_monk(df)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)




print('Finished Training')