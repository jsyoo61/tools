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
        assert type(axis) is int, 'type(axis) should be integer'
        axis = (axis,)
    
    axis = tuple(ax if ax>=0 else ndim+ax for ax in axis)
    return axis

def TSS_score(y_true, axis=None, axis_norm=None, axis_pool=None):

    # Axis fixing
    axis = tuple(range(y_true.ndim)) if axis is None else axis # Default to collapsing all dimensions
    axis_norm = axis if axis_norm is None else axis_norm
    axis_pool = axis_norm if axis_pool is None else axis_pool
    axis, axis_norm, axis_pool = axis_fix(axis, y_true.ndim), axis_fix(axis_norm, y_true.ndim), axis_fix(axis_pool, y_true.ndim)

    # axis trimming operations for computing TSS
    axis_set, axis_norm_set, axis_pool_set = set(axis), set(axis_norm), set(axis_pool)
    # axis_pool_set = {axis_pool} if not isinstance(axis_pool, Iterable) else set(axis_pool)
    # axis_set = set(axis)
    # axis_norm_set = {axis_norm} if axis_norm is not None and not isinstance(axis_norm, Iterable) else set(axis_norm) if axis_norm is not None else set()

    assert axis_norm_set.issubset(axis_pool_set), f'axis_norm {axis_norm} must be a subset of axis_pool {axis_pool} because axis_norm defines variability and axis_pool additionally averages'

    # Dimensions of TSS must be smaller than RSS, so mean/sum over axis_pool and axis
    axis_sum = axis
    axis_mean = tuple(axis_pool_set - axis_set) # Average over axis_pool - axis

    if not axis_set.issubset(axis_pool_set): # If axis is not a subset of axis_pool, expand axis_pool to include axis
        warnings.warn(f"axis {axis} is not a subset of axis_pool {axis_pool}, TSS sums over axis {axis_sum} and averages over remaining axis_pool {axis_mean}")

    # axis_norm used to compute the mean of y_true
    y_mean = np.mean(y_true, axis=axis_norm, keepdims=True)
    TS = (y_true - y_mean)**2 # Total Square (TS)

    # axis_pool used to aggregate additional dimensions (average) additional to axis
    TSS = np.mean(TS, axis=axis_mean, keepdims=True)
    TSS = np.sum(TSS, axis=axis_sum, keepdims=True)

    return TSS

def r2_score(y_true, y_pred, axis=None, axis_norm=None, axis_pool=None, force_finite=True, TSS=None):
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

            
    axis_norm: int or iterable of int, default=None
        axis used to measure y_true.mean(axis=axis_norm), to measure reference variability (TSS) for normalizing.
        It is recommended to keep axis_norm minimal, since a smaller axis_norm yields more localized TSS and a more detailed Dim-R2. 

    axis_pool: int or iterable of int, default=None
        axis used to additionally average TSS across for broader evaluation.
        It is recommended to keep axis_pool minimal, since a smaller axis_pool yields more localized normalization references, providing a more detailed Dim-R2.
    
    Returns
    -------
    z : np.ndarray
        if axis is specified, returns an array of shape with remaining axes.
        if axis=None, a single number is returned.
    """

    # Axis fixing
    axis = tuple(range(y_true.ndim)) if axis is None else axis # Default to collapsing all dimensions
    axis_norm = axis if axis_norm is None else axis_norm
    axis_pool = axis_norm if axis_pool is None else axis_pool
    axis, axis_norm, axis_pool = axis_fix(axis, y_true.ndim), axis_fix(axis_norm, y_true.ndim), axis_fix(axis_pool, y_true.ndim)

    # Residual Sum of Squares (RSS) 
    RS = (y_true - y_pred)**2 # Residual Square (RS)
    RSS = np.sum(RS, axis=axis, keepdims=True)

    # Total Sum of Squares (TSS)
    if TSS is None:
        TSS = TSS_score(y_true=y_true, axis=axis, axis_norm=axis_norm, axis_pool=axis_pool)
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

def ICC(data, icc_type=(2,1), per_param=False):
    """
    Intraclass correlation coefficient (ICC) via two-way ANOVA decomposition.
    Following Shrout & Fleiss (1979) notation.

    Parameters
    ----------
    data : array-like of shape (k, n)
        Rows are judges/raters (k), columns are targets (n).
    icc_type : tuple
        (1,1)    - one-way random, single measures
        (1,None) - one-way random, average of k measures
        (2,1)    - two-way random, absolute agreement, single measures
        (2,None) - two-way random, absolute agreement, average measures
        (3,1)    - two-way mixed, consistency, single measures
        (3,None) - two-way mixed, consistency, average measures
    per_param : bool
        If True, return per-target ICC of shape (n,).

    Returns
    -------
    float or np.ndarray
        ICC value(s).
    """
    data = np.asarray(data, dtype=float)
    assert data.ndim==2, f'data must be 2D (k_judges x n_targets), received shape: {data.shape}'
    k, n = data.shape  # k judges (rows), n targets (columns)
    assert len(icc_type)==2, f'icc_type must be 2D (case, avg), received: {icc_type}'
    case, avg = icc_type[0], icc_type[1]
    assert case in (1,2,3) and avg in (1, None), f'Unknown icc type: {icc_type}. Choose from (1,1), (1,None), (2,1), (2,None), (3,1), (3,None)'
    avg = icc_type[1] is None
    assert not (per_param and avg), 'per_param and icc_type=(n, None) is currently undefined'

    mu   = data.mean()
    mu_C = data.mean(axis=0)  # (n,) column means (targets)
    SSC  = k * ((mu_C - mu) ** 2).sum()
    SSW_j = ((data-mu_C)**2).sum(axis=0)
    SSW = SSW_j.sum() # == SST-SSC

    BMS  = SSC / (n - 1)
    WMS_j = SSW_j / (k - 1) if per_param else None
    WMS = (SSW) / (n * (k - 1))

    if case in (2,3):
        mu_R = data.mean(axis=1)  # (k,) row means (judges)
        SSR  = n * ((mu_R - mu) ** 2).sum()
        SSE = SSW - SSR # == SST - SSC - SSR
        JMS = SSR / (k - 1)
        EMS  = SSE / ((n - 1) * (k - 1))

    if case == 1:
        if avg:
            icc = (BMS - WMS) / BMS
        else:
            # icc = (BMS - WMS) / (BMS + (k - 1) * WMS)
            icc = 1 - k * (WMS_j if per_param else WMS) / (BMS + (k - 1) * WMS) # identical, matter of perspective
            
    elif case == 2:
        if avg:
            icc = (BMS - EMS) / (BMS + (JMS - EMS) / n)
        else:
            # icc = (BMS - EMS) / (BMS + (k - 1) * EMS + k * (JMS - EMS) / n)
            icc = 1 - k * (WMS_j if per_param else WMS) / (BMS - EMS + k * WMS) # identical, matter of perspective

    elif case == 3:
        if avg:
            icc = (BMS - EMS) / BMS
        else:
            if per_param:
                icc = 1 - ( k / (n-1) ) * ( n * WMS_j - JMS) / (BMS + (k - 1) * EMS)
            else:
                # icc = (BMS - EMS) / (BMS + (k - 1) * EMS)
                icc = 1 - k * EMS / (BMS + (k - 1) * EMS) # identical, matter of perspective

    return icc if per_param else icc.item()

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

# %%