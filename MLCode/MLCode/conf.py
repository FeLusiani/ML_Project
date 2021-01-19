from pathlib import Path

path_prj        = Path(__file__).parents[2]
path_data       = path_prj  / Path('./data')
TR_CUP_csv      = path_data / Path('./ML_CUP/ML-CUP20-TR.csv')
TS_CUP_csv      = path_data / Path('./ML_CUP//ML-CUP20-TS.csv')