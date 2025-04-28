from collections.abc import Iterable
import warnings

import numpy as np
from sklearn.metrics import r2_score as r2_score_sklearn

def squared_error(y_true, y_pred, axis=None):
    '''MSE without mean
    if axis is provided, it will return the mean along the axis
    '''
    score = (y_true-y_pred)**2
    if axis is not None:
        score = np.mean(score, axis=axis)
    return score

def absolute_error(y_true, y_pred, axis=None):
    '''MAE without mean'''
    score = np.abs(y_true-y_pred)
    if axis is not None:
        score = np.mean(score, axis=axis)
    return score

def r2_score(y_true, y_pred, axis=None, multioutput='raw_values'):
    """
    R^2 score for multidimensional predictions.
    collapses all axes except the specified axis.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    axis: int or iterable of int, default=None
        Axis to collapse.
        It must be specified if y_true and y_pred dimensions are > 2
    multioutput : Reference to `sklearn.metrics.r2_score`
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    
    Returns
    -------
    z : np.ndarray
        if axis is specified, returns an array of shape (y_true.shape[axis],)
    """
    assert y_true.shape == y_pred.shape, f"y_true and y_pred must have the same shape, received: {y_true.shape} and {y_pred.shape}"
    if axis is not None:
        if not isinstance(axis, Iterable):
            axis = [axis]
        else:
            axis = list(axis)
        if len(axis) > y_true.ndim:
            raise ValueError("Axis is greater than the number of dimensions of y_true and y_pred")

        # shape_final = np.array(y_true.shape)[axis]
        # shape_collapse = list(set(range(y_true.ndim)).difference(axis))
        dim_collapse = axis
        dim_final = list(set(range(y_true.ndim)).difference(axis))
        shape_final = np.array(y_true.shape)[dim_final]

        y_true = np.transpose(y_true, (*dim_collapse, *dim_final)).reshape(-1, np.prod(shape_final)) # Move axis to the end, and flatten the rest
        y_pred = np.transpose(y_pred, (*dim_collapse, *dim_final)).reshape(-1, np.prod(shape_final))

        score = r2_score_sklearn(y_true, y_pred, multioutput=multioutput)

        if type(score) == float and np.isnan(score).item():
            warnings.warn("R2 score is a single NaN, shape matching with NaN")
            score = np.full(shape_final, np.nan)
            return score

        if multioutput == 'raw_values':
            return score.reshape(*shape_final)
        elif multioutput == 'uniform_average':
            return score
        elif multioutput == 'variance_weighted':
            # return np.mean(score, weights=np.var(y_true, axis=0))
            raise NotImplementedError("variance_weighted is not implemented yet")
        else:
            raise ValueError("multioutput must be one of ['raw_values', 'uniform_average', 'variance_weighted']")

    else:
        assert (y_true.ndim <= 2) and (y_pred.ndim <= 2), "If axis is None, y_true and y_pred must be smaller than 2D"
        return r2_score_sklearn(y_true, y_pred, multioutput=multioutput)

# %%