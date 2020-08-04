import numpy as np


def mdpe_score(y, y_pred):
    """
    Calculate median value of BIS performance error.

    :param y: list of 1d-numpy array
    :param y_pred: list of 1d-numpy array
    :return: 1d-numpy array
    """

    assert len(y_pred) == len(y), 'Number of records must be match'

    count = len(y_pred)
    mdpe_list = []
    for i in range(count):
        pe = (y[i] - y_pred[i]) / y_pred[i]
        pe *= 100   # make to percentage
        mdpe_list.append(np.nanmedian(pe))

    return np.mean(mdpe_list)


def mdape_score(y, y_pred):
    """
    Calculate median value of BIS absolute performance error.

    :param y: list of 1d-numpy array
    :param y_pred: list of 1d-numpy array
    :return: 1d-numpy array
    """

    assert len(y_pred) == len(y), 'Number of records must be match'

    count = len(y_pred)
    mdape_list = []
    for i in range(count):
        pe = (y[i] - y_pred[i]) / y_pred[i]
        pe *= 100  # make to percentage
        ape = np.abs(pe)
        mdape_list.append(np.nanmedian(ape))

    return np.mean(mdape_list)
