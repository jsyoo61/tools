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
    if axis=(), returns entry-by-entry absolute error.

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

def r2_score(y_true, y_pred, axis=None, axis_ref=None, axis_bias=None, multioutput='raw_values', force_finite=True):
    """
    R^2 score for multidimensional predictions.
    collapses all axes except the specified axis.

    Computes 1 - RSS / TSS, where RSS is the residual sum of squares and TSS is the total sum of squares.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    axis: int or iterable of int, default=None
        Axis to collapse.
        If None, collapses all axes.
        
    multioutput : Reference to `sklearn.metrics.r2_score`
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

        Note:
        - default is 'raw_values', which is different from sklearn.
        - when multioutput is uniform_average or variance_weighted, return value is a single float even if axis is specified.
          The axis specifies the feature dimensions to average over.
    
    Returns
    -------
    z : np.ndarray
        if axis is specified, returns an array of shape (y_true.shape[axis],)
    """

    # Residual Sum of Squares (RSS) and Total Sum of Squares (TSS)
    RS = (y_true - y_pred)**2 # Residual Square (RS)
    RSS = np.sum(RS, axis=axis, keepdims=True)

    if axis_ref is None:
        axis_ref = axis
    if axis_bias is None:
        axis_bias = axis_ref

    axis_ref_set = {axis_ref} if not isinstance(axis_ref, Iterable) else set(axis_ref)
    axis_set = {axis} if not isinstance(axis, Iterable) else set(axis)
    axis_bias_set = {axis_bias} if axis_bias is not None and not isinstance(axis_bias, Iterable) else set(axis_bias) if axis_bias is not None else set()

    assert axis_bias_set.issubset(axis_ref_set), f'axis_bias ({axis_bias}) must be a subset of axis_ref ({axis_ref}) because axis_ measure variability' # If axis_bias is not a subset of axis_ref, expand axis_ref to include axis_bias

    # Dimensions of TSS must be smaller than RSS, so mean/sum over axis_ref and axis
    axis_sum = axis
    axis_mean = tuple(axis_ref_set - axis_set) # Average over axis_ref - axis

    if not axis_set.issubset(axis_ref_set): # If axis is not a subset of axis_ref, expand axis_ref to include axis
        warnings.warn(f"axis {axis} is not a subset of axis_ref {axis_ref}, TSS sums over axis {axis_sum} and averages over remaining axis_ref {axis_mean}")

    # axis_bias used to compute the mean of y_true
    y_mean = np.mean(y_true, axis=axis_bias, keepdims=True)
    TS = (y_true - y_mean)**2 # Total Square (TS)

    # axis_ref used to aggregate additional dimensions (average) additional to axis
    TSS = np.mean(TS, axis=axis_mean, keepdims=True)
    TSS = np.sum(TSS, axis=axis_sum, keepdims=True)
    
    score = 1 - RSS / TSS
    score = np.squeeze(score, axis=axis) # Collapse the axis_ref dimension

    if force_finite:
        score[np.isnan(score)] = 1
        score[np.isinf(score)] = 0 # -Inf means no fit, so set to 0

    if axis is None:
        score = score.item()

    return score

# def r2_score(y_true, y_pred, axis=None, multioutput='raw_values'):
#     """
#     R^2 score for multidimensional predictions.
#     collapses all axes except the specified axis.

#     Parameters
#     ----------
#     y_true : np.ndarray
#     y_pred : np.ndarray
#     axis: int or iterable of int, default=None
#         Axis to collapse.
#         If None, collapses all axes.
        
#     multioutput : Reference to `sklearn.metrics.r2_score`
#         https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

#         Note:
#         - default is 'raw_values', which is different from sklearn.
#         - when multioutput is uniform_average or variance_weighted, return value is a single float even if axis is specified.
#           The axis specifies the feature dimensions to average over.
    
#     Returns
#     -------
#     z : np.ndarray
#         if axis is specified, returns an array of shape (y_true.shape[axis],)
#     """
#     assert y_true.shape == y_pred.shape, f"y_true and y_pred must have the same shape, received: {y_true.shape} and {y_pred.shape}"
#     if axis is None:
#         axis = list(range(y_true.ndim)) # Collapse all dimensions
#         axis_was_none = True
#     elif not isinstance(axis, Iterable):
#         axis = [axis]
#         axis_was_none = False
#     else:
#         axis = list(axis)
#         axis_was_none = False

#     if len(axis) > y_true.ndim:
#         raise ValueError("Axis is greater than the number of dimensions of y_true and y_pred")

#     dim_collapse = axis
#     dim_final = list(set(range(y_true.ndim)).difference(axis))
#     shape_final = np.array(y_true.shape)[dim_final] if len(dim_final)!=0 else (1,)

#     y_true = np.transpose(y_true, (*dim_collapse, *dim_final)).reshape(-1, np.prod(shape_final)) # Move axis to the end, and flatten the rest
#     y_pred = np.transpose(y_pred, (*dim_collapse, *dim_final)).reshape(-1, np.prod(shape_final))

#     # score = r2_score_sklearn(y_true, y_pred, multioutput=multioutput)
#     score = r2_score_sklearn(y_true, y_pred, multioutput='raw_values')

#     if type(score) == float and np.isnan(score).item():
#         warnings.warn("R2 score is a single NaN, shape matching with NaN")
#         score = np.full(shape_final, np.nan)
#         return score

#     if multioutput == 'raw_values':
#         score = score.reshape(*shape_final) if not axis_was_none else score[0]
#         return score
#     elif multioutput == 'uniform_average':
#         return score.mean()
#     elif multioutput == 'variance_weighted':        
#         return np.average(score, weights=np.var(y_true, axis=0))
#     else:
#         raise ValueError("multioutput must be one of ['raw_values', 'uniform_average', 'variance_weighted']")

# %%