from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import RepeatedKFold
import time



# function that train the algorithm with the function passed


def kFoldCross(trainCallback, predictCallback, X_dev, Y_dev, n_splits, resetCallback=None):
    kf = KFold(n_splits=n_splits)
    ValError = []

    for train, val in kf.split(X_dev):
        start_time = time.process_time()

        if resetCallback: resetCallback()
        trainCallback(X_dev[train], Y_dev[train])
        y_predicted = predictCallback(X_dev[val])
        val_e = MEE(y_predicted, Y_dev[val])
        ValError.append(val_e)

        seconds = time.process_time()-start_time
    
    ValError = np.array(ValError)

    mean_ValError = np.mean(ValError)
    std_ValError = np.std(ValError)

    return mean_ValError, std_ValError, seconds


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