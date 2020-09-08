import logging
import os
from logging.handlers import RotatingFileHandler
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.utils import shuffle

pkl_url = '../models/precio_sucursal_producto_400.pkl'

full_ds = pd.read_pickle(pkl_url, compression='zip')
full_ds = shuffle(full_ds)

n_split = 20
train_ratio = 0.7
ds_len = len(full_ds)
fold = int(ds_len / n_split)


i = 0
for n in range(0, n_split):
    j = i + fold
    curr_ratio = i / ds_len
    if curr_ratio <= train_ratio:
        pd.to_pickle(full_ds.iloc[i:j, :], '../models/psp_train_{}.pkl'.format(n), compression="zip", protocol=4)
    else:
        pd.to_pickle(full_ds.iloc[i:, :], '../models/psp_eval.pkl', compression="zip", protocol=4)
        break
    i = j

