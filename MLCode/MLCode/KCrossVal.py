from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import RepeatedKFold
import time



# function that train the algorithm with the function passed

def kFoldCross(*args, out_scaler=False):
    start_time = time.process_time()
    if out_scaler:
        res = kFoldCross_outscaled(*args)
    else:
        res = kFoldCross_normal(*args)
    
    seconds = time.process_time()-start_time
    return res + (seconds,)


def kFoldCross_normal(trainCallback, predictCallback, X_dev, Y_dev, n_splits):
    kf = KFold(n_splits=n_splits)
    ValError = []
    TrError = []

    for train, val in kf.split(X_dev):
        trainCallback(X_dev[train], Y_dev[train])

        val_predicted = predictCallback(X_dev[val])
        val_e = MEE(val_predicted, Y_dev[val])
        ValError.append(val_e)

        tr_predicted = predictCallback(X_dev[train])
        tr_e = MEE(tr_predicted, Y_dev[train])
        TrError.append(tr_e)
    
    ValError = np.array(ValError)
    TrError = np.array(TrError)

    mean_ValError = np.mean(ValError)
    std_ValError = np.std(ValError)

    mean_TrError = np.mean(TrError)

    return mean_ValError, std_ValError, mean_TrError


def kFoldCross_outscaled(trainCallback, predictCallback, X_dev, Y_dev, n_splits):
    kf = KFold(n_splits=n_splits)
    ValError = []
    TrError = []

    out_scaler = StandardScaler()
    Y_devS = out_scaler.fit_transform(Y_dev)

    for train, val in kf.split(X_dev):
        trainCallback(X_dev[train], Y_devS[train])

        val_predicted = predictCallback(X_dev[val])
        val_predicted = out_scaler.inverse_transform(val_predicted)
        val_e = MEE(val_predicted, Y_dev[val])
        ValError.append(val_e)

        tr_predicted = predictCallback(X_dev[train])
        tr_predicted = out_scaler.inverse_transform(tr_predicted)
        tr_e = MEE(tr_predicted, Y_dev[train])
        TrError.append(tr_e)

    
    ValError = np.array(ValError)
    TrError = np.array(TrError)

    mean_ValError = np.mean(ValError)
    std_ValError = np.std(ValError)

    mean_TrError = np.mean(TrError)

    return mean_ValError, std_ValError, mean_TrError


def MEE(x,y,mean=True):
    """Mean Euclidean Error for pytorch.Tensor class.
    
    Iteratates along the first axis and
    applies the euclidean distance to each element.
    The resulting values are averaged if `mean=True` (default),
    otherwise summed."""
    assert x.shape == y.shape, 'x and y must have same shape'
    length = x.shape[0]
    r = sum([norm(x[i]-y[i]) for i in range(length)])
    return r/length if mean else r