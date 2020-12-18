from pathlib import Path
import pandas as pd
import numpy as np
from conf import path_data

def load_CUP_data(csv_file: Path):
    return pd.read_csv(csv_file,skiprows=7, header=None, index_col=0)

def load_monk_data(num: int, train=True):
    """Returns dataframe with monk dataset.

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
    return pd.DataFrame(rows, columns=col_names).set_index('ID')


def preprocess_monk(df):
    """Applies one-hot encoding of input values,
    and rescales binary labels `0,1` to `0.1,0.9`.

    Args:
        df (pandas.DataFrame): monk dataset

    Returns:
        X, Y: `(values, labels)` tuple
    """
    # one-hot encoding
    df = pd.get_dummies(df, columns=df.columns[1:])
    matrix = df.to_numpy(dtype=np.float64)
    # shuffle rows
    np.random.shuffle(matrix)

    X = matrix[:,1:]
    # X = matrix[:,1:].astype(np.float)
    # rescale labels
    rescale = lambda x: 0.1 if x == 0 else 0.9
    v_rescale = np.vectorize(rescale)
    Y = v_rescale(matrix[:,0])
    return X, Y


    
