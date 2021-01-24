from MLCode.conf import path_data
from pathlib import Path
import pandas as pd

def get_HP_names(kernel):
    if kernel == 'rbf':
        hp_names=['C', 'epsilon', 'gamma']
    elif kernel == 'poly':
        hp_names=['C', 'epsilon', 'degree', 'coeff']
    elif kernel == 'sigmoid':
        hp_names=['C', 'epsilon', 'gamma', 'coeff']
    
    return hp_names

def load_results(kernel):
    saved_csv = path_data / Path(f'SVM_results_{kernel}.csv')
    if saved_csv.exists():
        return pd.read_csv(saved_csv)
    else:
        columns = get_HP_names(kernel) + ['MEE_mean', 'MEE_std', 'seconds']
        return pd.DataFrame(columns=columns)


def save_results(df, kernel):
    saved_csv = path_data / Path(f'SVM_results_{kernel}.csv')
    saved_df = load_results(kernel)
    saved_df = saved_df.append(df, ignore_index=True)
    columns = get_HP_names(kernel)
    saved_df = saved_df.drop_duplicates(columns, keep='last')
    saved_df.to_csv(saved_csv, index=False)