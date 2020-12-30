from pathlib import Path
import pandas as pd
import numpy as np
from .conf import path_data
import matplotlib.pyplot as plt


def load_cup_data(train=True):
    """Returns dataframe with CUP dataset.
    If `train=True` (default), training data is loaded,
    else test data is loaded.
    """
    type = "TR" if train else "TS"
    csv_file = path_data / Path(f"ML_CUP/ML-CUP20-{type}.csv")
    return pd.read_csv(csv_file, skiprows=7, header=None, index_col=0)


def load_monk_data(num: int, train=True):
    """Returns dataframe with monk dataset.
    One-hot encoding is applied to the attribute values.

    Args:
        num (int): determines which monk dataset to load (1, 2 or 3)
        train (Bool): If True (default), load training data.
            If False, load test data.

    Returns:
        pandas.DataFrame: loaded dataset
    """
    data_type = "train" if train else "test"
    file = path_data / Path(f"./MONK/monks-{num}.{data_type}.txt")
    lines = file.read_text().split("\n")
    # delete last empty line
    lines = lines[:-1]
    rows = [l.strip().split(" ") for l in lines]
    col_names = ["out", "a1", "a2", "a3", "a4", "a5", "a6", "ID"]
    df = pd.DataFrame(rows, columns=col_names).set_index("ID")
    return pd.get_dummies(df, columns=df.columns[1:])


def np_monk(df, X_type=np.float, Y_type=np.int):
    """Returns monk dataset as `(X, Y)` numpy arrays,
    respectively of type `X_type` and `Y_type`.
    """
    matrix = df.to_numpy()

    X = matrix[:, 1:]
    X = X.astype(X_type)

    Y = matrix[:, 0]
    Y = Y.astype(Y_type)
    Y = Y.reshape(-1, 1)

    return X, Y


def np_cup_TR(df):
    """Returns the CUP TR dataset as `(X, Y)` numpy arrays.
    The data is also shuffled. 
    """
    matrix = df.to_numpy()
    np.random.shuffle(matrix)

    X = matrix[:, :10]
    Y = matrix[:, 10:]

    return X, Y


def rescale_bin(data):
    """Rescale numpy binary array: `(0,1)` -> `(0.1,0.9)` """
    rescale = lambda x: 0.1 if x == 0 else 0.9
    v_rescale = np.vectorize(rescale)
    return v_rescale(data)


def plot_NN_TR_TS(tr_stat, test_stat, name='error'):
    _, ax = plt.subplots()
    ax.plot(tr_stat, label="training")
    ax.plot(test_stat, label="test")
    ax.legend()
    ax.set(xlabel='epoch', ylabel=name)
    ax.set_title(name+' per epoch')
    return ax


def plot_NN_TR_VAL(tr_stat, val_stat, name='error'):
    _, ax = plt.subplots()
    ax.plot(tr_stat, label="training")
    ax.plot(val_stat, label="validation")
    ax.legend()
    ax.set(xlabel='epoch', ylabel=name)
    ax.set_title(name+' per epoch')
    return ax