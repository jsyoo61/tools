from collections.abc import Iterable
from copy import deepcopy as dcopy
import warnings

import numpy as np
import sklearn.metrics as sk_metrics

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

def axis_fix(axis, ndim):
    """
    Convert axis into a tuple of positive integers.
    
    Parameters
    ----------
    axis : int or iterable of int
        Axis or axes to normalize.
    
    ndim : int
        The maximum number of dimensions axis can have.
        Used as reference for converting negative indices.
    
    Returns
    -------
    axis : tuple of int
        Normalized axis as a tuple of non-negative integers.
    """
    if not isinstance(axis, Iterable): # axis is a single int
        axis = (axis,)
    
    axis = tuple(ax if ax>=0 else ndim+ax for ax in axis)
    return axis

def TSS_score(y_true, axis=None, axis_ref=None, axis_bias=None):
    # Axis fixing
    axis = tuple(range(y_true.ndim)) if axis is None else axis # Default to collapsing all dimensions
    axis_ref = axis if axis_ref is None else axis_ref
    axis_bias = axis_ref if axis_bias is None else axis_bias
    axis, axis_ref, axis_bias = axis_fix(axis, y_true.ndim), axis_fix(axis_ref, y_true.ndim), axis_fix(axis_bias, y_true.ndim)

    # axis trimming operations for computing TSS
    axis_ref_set = {axis_ref} if not isinstance(axis_ref, Iterable) else set(axis_ref)
    axis_set = set(axis)
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

    return TSS

def r2_score(y_true, y_pred, axis=None, axis_ref=None, axis_bias=None, force_finite=True, TSS=None):
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
        If None, collapses all axes to yield a single number.
        
    axis_ref: reference axis to measure variability across, which normalizes r2 score.

    axis_bias: axis used to measure y_true.mean(axis_bias), when measuring reference variability (TSS) to normalize.
    
    Returns
    -------
    z : np.ndarray
        if axis is specified, returns an array of shape with remaining axes.
        if axis=None, a single number is returned.
    """

    # Axis fixing
    axis = tuple(range(y_true.ndim)) if axis is None else axis # Default to collapsing all dimensions
    axis_ref = axis if axis_ref is None else axis_ref
    axis_bias = axis_ref if axis_bias is None else axis_bias
    axis, axis_ref, axis_bias = axis_fix(axis, y_true.ndim), axis_fix(axis_ref, y_true.ndim), axis_fix(axis_bias, y_true.ndim)

    # Residual Sum of Squares (RSS) 
    RS = (y_true - y_pred)**2 # Residual Square (RS)
    RSS = np.sum(RS, axis=axis, keepdims=True)

    # Total Sum of Squares (TSS)
    if TSS is None:
        TSS = TSS_score(y_true=y_true, axis=axis, axis_ref=axis_ref, axis_bias=axis_bias)
    else: # TSS is given
        try:
            TSS = np.broadcast_to(TSS, RSS.shape)
        except ValueError as e:
            raise ValueError(f'The shape of given TSS ({TSS.shape}) must be broadcastable to shape of RSS ({RSS.shape})') from e

    # R2
    score = 1 - RSS / TSS
    score = np.squeeze(score, axis=axis) # Collapse the axis dimension

    # if nan (TSS=0, y_true no variance) or -inf (RSS/TSS=inf, very bad prediction)
    if force_finite:
        score[np.isnan(score)] = 1
        score[np.isinf(score)] = 0 # -Inf means no fit, so set to 0

    # if len(axis) == y_true.ndim:
    if score.ndim==0: # score.ndim==0 is better than len(axis) == y_true.ndim?
        score = score.item()

    return score

# Binary classification
def sensitivity_score(y_true, y_pred): # alias
    '''sensitivity == recall == tpr'''
    return sk_metrics.recall_score(y_true, y_pred)

def specificity_score(y_true, y_pred):
    (tn, fp), (fn, tp) = sk_metrics.confusion_matrix(y_true, y_pred)
    if (tn+fp)==0:
        warnings.warn('invalid value in specificity_score, setting to 0.0')
        return 0
    return tn / (tn+fp)

# Multiclass classification
def classification_report_full(y_true, y_pred=None, y_score=None, ovr=True):
    '''
    Adds additional metrics to sklearn.classification_report
    '''
    assert not (y_pred is None and y_score is None), 'either one of y_pred or y_score needs to be given'
    if y_pred is None:
        y_pred = y_score.argmax(1)

    # y_score = result['y_score'] if 'y_score' in result else None
    # y_true, y_pred = result['y_true'], result['y_pred']
    
    scores = sk_metrics.classification_report(y_true, y_pred, output_dict=True)
    scores['mcc'] = sk_metrics.matthews_corrcoef(y_true, y_pred)

    if ovr:
        scores_ovr = {k:dcopy(v) for k, v in scores.items() if k.isnumeric()}
        scores_all = {k:dcopy(v) for k, v in scores.items() if not k.isnumeric()}

        more_scorers_y_pred = {'sensitivity': sensitivity_score, 'specificity': specificity_score, 'accuracy': sk_metrics.accuracy_score} # Optimize to reduce redundant computations?
        more_scorers_y_score = {}

        # Additional metrics
        for c in scores_ovr.keys():
            c_int = int(c)
            y_true__c, y_pred_c = y_true==c_int, y_pred==c_int
            for scorer_name, scorer in more_scorers_y_pred.items():
                scores_ovr[c][scorer_name] = scorer({'y_true': y_true__c, 'y_pred': y_pred_c})

        if y_score is not None:
            more_scorers_y_score['auroc'] = sk_metrics.roc_auc_score
            for c in scores_ovr.keys():
                c_int = int(c)
                y_true_ = y_true==c_int
                y_score_ = y_score[:, c_int]
                for scorer_name, scorer in more_scorers_y_score.items():
                    scores_ovr[c][scorer_name] = scorer(y_true_, y_score_)

        # summary
        more_scorers = list(more_scorers_y_pred.keys()) + list(more_scorers_y_score.keys())
        for scorer_name in more_scorers:
            scores_all['macro avg'][scorer_name] = np.mean([scores_ovr_[scorer_name] for scores_ovr_ in scores_ovr.values()])
            scores_all['weighted avg'][scorer_name] = np.sum([scores_ovr_[scorer_name]*scores_ovr_['support'] for scores_ovr_ in scores_ovr.values()]) / scores_all['weighted avg']['support']

        scores.update(scores_ovr)
        scores.update(scores_all)

    return scores

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