from pathlib import Path
import pandas as pd
import numpy as np
from .conf import path_data
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler


def load_cup_data(train=True):
    """Returns dataframe with CUP dataset.
    If `train=True` (default), training data is loaded,
    else test data is loaded.
    """
    type = "TR" if train else "TS"
    csv_file = path_data / Path(f"ML_CUP/ML-CUP20-{type}.csv")
    return pd.read_csv(csv_file, skiprows=7, header=None, index_col=0)


def load_shuffled_cup():
    csv_file = path_data / Path('ML_CUP/shuffled_cup.csv')
    return pd.read_csv(csv_file, index_col=0)


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


def np_cup_TR(df, test=False):
    """Returns `(X, Y)` numpy arrays from a CUP dataset (`pandas.DataFrame`).
    If `test=False` (default), only the first 90% of the data will be returned,
    otherwise only the remaining 10% will be returned.
    `X` data is scaled with sklearn `StandardScaler` (fitted on development set).
    """
    matrix = df.to_numpy()
    # test samples are 10% of all the samples
    test_samples = matrix.shape[0] // 10

    # the first 90%
    dev_set = matrix[:-test_samples]
    # the last 10%
    test_set = matrix[-test_samples:]
    
    X_dev = dev_set[:, :10]
    Y_dev = dev_set[:, 10:]

    X_test = test_set[:, :10]
    Y_test = test_set[:, 10:]

    X_scaler = StandardScaler()
    X_scaler.fit(X_dev)

    if test:
        return X_scaler.transform(X_test), Y_test
    else:
        return X_scaler.transform(X_dev), Y_dev


def rescale_bin(data):
    """Rescale numpy binary array: `(0,1)` -> `(0.1,0.9)` """
    rescale = lambda x: 0.1 if x == 0 else 0.9
    v_rescale = np.vectorize(rescale)
    return v_rescale(data)


def plot_NN_TR_TS(tr_stat, test_stat, name='error'):
    _, ax = plt.subplots()
    ax.plot(tr_stat, label="training")
    ax.plot(test_stat, '--', label="test")
    ax.legend()
    ax.set(xlabel='epoch', ylabel=name)
    ax.set_title(name+' per epoch')
    return ax


def plot_NN_TR_VAL(tr_stat, val_stat, name='error'):
    _, ax = plt.subplots()
    ax.plot(tr_stat, label="training")
    ax.plot(val_stat, '--', label="validation")
    ax.legend()
    ax.set(xlabel='epoch', ylabel=name)
    ax.set_title(name+' per epoch')
    return ax


def unique_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


def writeOutput(result, name):
    # Name1  Surname1, Name2 Surname2
    # Group Nickname
    # ML-CUP18
    # 02/11/2018
    df = pd.DataFrame(result)
    now = datetime.datetime.now()
    f = open(name, "w")
    f.write("# Christian Esposito, Federico Lusiani\n")
    f.write("# Cheesleaders\n")
    f.write("# ML-CUP20\n")
    f.write("# " + str(now.day) + "/" + str(now.month) + "/" + str(now.year) + "\n")
    df.index += 1
    df.to_csv(f, sep=",", encoding="utf-8", header=False)
    f.close()