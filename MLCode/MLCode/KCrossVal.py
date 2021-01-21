from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import RepeatedKFold


# function that train the algorithm with the function passed
#

def kFoldCross(*args, out_scaler=False):
    if out_scaler:
        return kFoldCross_outscaled(*args)
    else:
        return kFoldCross_normal(*args)


def kFoldCross_normal(trainCallback, predictcallback, X_dev, Y_dev, n_splits):
    kf = KFold(n_splits=n_splits)
    ValError = []

    for train, val in kf.split(X_dev):
        trainCallback(X_dev[train], Y_dev[train])
        y_predicted = predictcallback(X_dev[val])
        val_e = MEE(y_predicted, Y_dev[val])
        ValError.append(val_e)
    
    ValError = np.array(ValError)

    mean_ValError = np.mean(ValError)
    std_ValError = np.std(ValError)

    return mean_ValError, std_ValError


def kFoldCross_outscaled(trainCallback, predictcallback, X_dev, Y_dev, n_splits):
    kf = KFold(n_splits=n_splits)
    ValError = []

    out_scaler = StandardScaler()
    Y_devS = out_scaler.fit_transform(Y_dev)

    for train, val in kf.split(X_dev):
        trainCallback(X_dev[train], Y_devS[train])
        y_predicted = predictcallback(X_dev[val])
        y_predicted = out_scaler.inverse_transform(y_predicted)
        val_e = MEE(y_predicted, Y_dev[val])
        ValError.append(val_e)
    
    ValError = np.array(ValError)

    mean_ValError = np.mean(ValError)
    std_ValError = np.std(ValError)

    return mean_ValError, std_ValError


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