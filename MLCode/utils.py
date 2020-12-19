from pathlib import Path
import pandas as pd
import numpy as np
from .conf import path_data

def load_CUP_data(csv_file: Path):
    return pd.read_csv(csv_file,skiprows=7, header=None, index_col=0)

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
    data_type = 'train' if train else 'test'
    file = path_data / Path(f'./MONK/monks-{num}.{data_type}.txt')
    lines = file.read_text().split('\n')
    # delete last empty line
    lines = lines[:-1]
    rows = [l.strip().split(' ') for l in lines]
    col_names = ['out','a1','a2','a3','a4','a5','a6', 'ID']
    df = pd.DataFrame(rows, columns=col_names).set_index('ID')
    return pd.get_dummies(df, columns=df.columns[1:])


def np_monk(df, X_type=np.float, Y_type=np.int):
    """Returns monk dataset as `(X, Y)` numpy arrays.
    The dataset is also shuffled.

    Args:
        df (pandas.DataFrame): monk dataset
        X_type (numpy.dtype): data type for instances `X`.
        Y_type (numpy.dtype): data type for labels `Y`.
    """
    matrix = df.to_numpy()
    # shuffle rows
    np.random.shuffle(matrix)
    X = matrix[:,1:]
    X = X.astype(X_type)

    Y = matrix[:,0]
    Y = Y.astype(Y_type)
    Y = Y.reshape(-1,1)

    return X, Y


def rescale_bin(data):
    """Rescale numpy binary array: `(0,1)` -> `(0.1,0.9)` """
    rescale = lambda x: 0.1 if x == 0 else 0.9
    v_rescale = np.vectorize(rescale)
    return v_rescale(data)


    
