from collections.abc import Iterable
import warnings

import numpy as np
from sklearn.metrics import r2_score as r2_score_sklearn

def squared_error(y_true, y_pred, axis=None):
    """
    MSE averaging across specified axis.
    if axis=(), returns entry-by-entry squared error.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    axis : int or iterable of int, default=None
        Axis to collapse.

    Returns
    -------
    score : numpy.ndarray of shape with collapsed axis
    """
    if axis is None:
        axis = tuple(range(y_true.ndim)) # Collapse all dimensions
    score = (y_true-y_pred)**2
    score = np.mean(score, axis=axis)
    return score

def absolute_error(y_true, y_pred, axis=None):
    """
    MAE averaging across specified axis.
    if axis=(), returns entry-by-entry squared error.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    axis : int or iterable of int, default=None
        Axis to collapse.

    Returns
    -------
    score : numpy.ndarray of shape with collapsed axis
    """
    if axis is None:
        axis = tuple(range(y_true.ndim)) # Collapse all dimensions
    score = np.abs(y_true-y_pred)
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
    if axis is None:
        axis = list(range(y_true.ndim)) # Collapse all dimensions
        axis_was_none = True
    elif not isinstance(axis, Iterable):
        axis = [axis]
        axis_was_none = False
    else:
        axis = list(axis)
        axis_was_none = False

    if len(axis) > y_true.ndim:
        raise ValueError("Axis is greater than the number of dimensions of y_true and y_pred")

    dim_collapse = axis
    dim_final = list(set(range(y_true.ndim)).difference(axis))
    shape_final = np.array(y_true.shape)[dim_final] if len(dim_final)!=0 else (1,)

    y_true = np.transpose(y_true, (*dim_collapse, *dim_final)).reshape(-1, np.prod(shape_final)) # Move axis to the end, and flatten the rest
    y_pred = np.transpose(y_pred, (*dim_collapse, *dim_final)).reshape(-1, np.prod(shape_final))

    # score = r2_score_sklearn(y_true, y_pred, multioutput=multioutput)
    score = r2_score_sklearn(y_true, y_pred, multioutput='raw_values')

    if type(score) == float and np.isnan(score).item():
        warnings.warn("R2 score is a single NaN, shape matching with NaN")
        score = np.full(shape_final, np.nan)
        return score

    if multioutput == 'raw_values':
        score = score.reshape(*shape_final) if not axis_was_none else score[0]
        return score
    elif multioutput == 'uniform_average':
        return score.mean()
    elif multioutput == 'variance_weighted':
        return np.average(score, weights=np.var(y_true, axis=0))
    else:
        raise ValueError("multioutput must be one of ['raw_values', 'uniform_average', 'variance_weighted']")

# %%