# %%
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, train_test_split, StratifiedKFold, KFold
import numpy as np
import pandas as pd
import torch

from .. import torch as ttorch

# %%
'''
Train/(Validation)/Test Split index functions
'''
def stratified_train_test_split_i(y, test_size=0.15, random_state=None):
    ''':return: indices of train_i, test_i'''
    x = np.empty(len(y))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_i, test_i = next(sss.split(x, y))
    return train_i, test_i

def stratified_train_val_test_split_i(y, val_size=0.15, test_size=0.15, random_state=None):
    x = np.empty(len(y))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_i, test_i = next(sss.split(x, y))

    train_val_y = y[train_val_i]
    x = np.empty(len(train_val_i))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=random_state)
    train_i_, val_i_ = next(sss.split(x, train_val_y))
    train_i = train_val_i[train_i_]
    val_i = train_val_i[val_i_]

    return train_i, val_i, test_i

def train_test_split_i(y, test_size=0.15, random_state=None):
    x = np.empty(len(y))

    ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_i, test_i = next(ss.split(x, y))

    return train_i, test_i

def train_val_test_split_i(y, val_size=0.15, test_size=0.15, random_state=None):
    x = np.empty(len(y))

    ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_i, test_i = next(ss.split(x, y))

    train_val_y = y[train_val_i]
    x = np.empty(len(train_val_i))
    ss = ShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=random_state)
    train_i_, val_i_ = next(ss.split(x, train_val_y))
    train_i = train_val_i[train_i_]
    val_i = train_val_i[val_i_]

    return train_i, val_i, test_i

def stratified_kfold_split_i(y, n_splits, split_i, shuffle=True, random_state=None):
    '''
    return train, test indices of "split_i"-th split of "n_splits"-fold split
    '''
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    x = np.zeros(len(y))
    skf_ = skf.split(x, y)

    for i in range(split_i):
        next(skf_)
    train_i, test_i = next(skf_)

    return train_i, test_i

def stratified_nested_kfold_split_i(y, n_splits, m_splits, split_i, split_j, shuffle=True, random_state=None):
    '''
    return train, val, test indices of "split_i"-th split of "n_splits"-fold split
    '''
    # Outer split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    x = np.zeros(len(y))
    skf_ = skf.split(x, y)

    for i in range(split_i):
        next(skf_)
    train_val_i, test_i = next(skf_) 

    # Inner split
    skf = StratifiedKFold(n_splits=m_splits, shuffle=shuffle, random_state=random_state)
    x = np.zeros(len(y[train_val_i]))
    skf_ = skf.split(x, y[train_val_i])

    for j in range(split_j):
        next(skf_)
    train_i, val_i = next(skf_)
    train_i, val_i = train_val_i[train_i], train_val_i[val_i]

    return train_i, val_i, test_i

def kfold_split_i(y, n_splits, split_i, shuffle=True, random_state=None):
    '''
    return train, test indices of "split_i"-th split of "n_splits"-fold split
    '''
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    kf_ = kf.split(y)

    for i in range(split_i):
        next(kf_)
    train_i, test_i = next(kf_)

    return train_i, test_i

def nested_kfold_split_i(y, n_splits, m_splits, split_i, split_j, shuffle=True, random_state=None):
    '''
    return train, val, test indices of "split_i"-th split of "n_splits"-fold split
    '''
    # Outer split
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    kf_ = kf.split(y)

    for i in range(split_i):
        next(kf_)
    train_val_i, test_i = next(kf_) 

    # Inner split
    kf = KFold(n_splits=m_splits, shuffle=shuffle, random_state=random_state)
    kf_ = kf.split(y[train_val_i])

    for j in range(split_j):
        next(kf_)
    train_i, val_i = next(kf_)
    train_i, val_i = train_val_i[train_i], train_val_i[val_i]

    return train_i, val_i, test_i

# %%
'''
Train/(Validation)/Test split dataset functions
'''
def nested_kfold_split_data(data, y, n_splits, m_splits, split_i, split_j, shuffle=True, random_state=None):
    '''
    y doesn't matter since it's not stratified
    '''
    train_i, val_i, test_i = nested_kfold_split_i(y, n_splits, m_splits, split_i, split_j, shuffle, random_state)
    train_data, val_data, test_data = wrap_data(data, train_i), wrap_data(data, val_i), wrap_data(data, test_i)
    return train_data, val_data, test_data
    
def stratified_nested_kfold_split_data(data, y, n_splits, m_splits, split_i, split_j, shuffle=True, random_state=None):
    """
    Split data into train, validation, and test set

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray or torch.Tensor or torch.dat.Dataset object or iterable object (i.e. list)
        Data to split

    y: pd.Series or np.ndarray or torch.Tensor or iterable object (i.e. list)
        Target variable to be used in stratified split

    Returns
    -------
    train_data : train data of the same variable format
    val_data : validation data of the same variable format
    test_data : test data of the same variable format
    """
    train_i, val_i, test_i = stratified_nested_kfold_split_i(y, n_splits, m_splits, split_i, split_j, shuffle, random_state)
    train_data, val_data, test_data = wrap_data(data, train_i), wrap_data(data, val_i), wrap_data(data, test_i)
    return train_data, val_data, test_data

def wrap_data(data, i):
    if isinstance(data, np.ndarray):
        return data[i]
    elif isinstance(data, pd.DataFrame):
        return data.iloc[i]
    elif isinstance(data, torch.Tensor):
        return data[i]
    elif isinstance(data, torch.utils.data.Dataset):
        return ttorch.data.ProxyDataset(dataset=data, idxs=i)
    else:
        return data[i]

# %%
# def train_val_test_split(x, val_size=0.1, test_size=0.1, random_state=None):
#     if type(x)==int:
#         x = np.zeros(x)
#     ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
#     train_val_i, test_i = next(ss.split(x))
#     train_i, val_i = train_test_split(train_val_i, test_size=val_size/(1-test_size), random_state=random_state)
#
#     # assert len(set(np.concatenate((train_i, val_i, test_i)))) == len(x)
#     return train_i, val_i, test_i

class MultiGridSearchCV():
    '''Perform Grid Search over multiple models

    Parameters
    ----------
    estimators: dict
        Name of model. This name has to correspond to model name of param_grid
    param_grids: dict
        Parameter Grid to Search. It must be dict nested in dict(Double dict).
    **kwargs: other parameters from GridSearchCV
        Look GridSearchCV

    examples of **kwargs
    --------------------
    cv: number of k-fold CV
    scoring: scoring method. either can be prebuilt sklearn string or a callable function

    '''
    def __init__(self, estimators, param_grids, **kwargs):
        self.estimators = estimators
        self.param_grids = param_grids
        self.kwargs = kwargs
        self.grid_searches = {estimator_name: GridSearchCV(estimator=estimator, param_grid=self.param_grids[estimator_name], **self.kwargs) for estimator_name, estimator in self.estimators.items()}

    def fit(self, x, y):
        for grid_search in self.grid_searches.values():
            grid_search.fit(x,y)
        self.cv_results = {estimator_name: grid_search.cv_results_ for estimator_name, grid_search in self.grid_searches.items()}
        best_estimator_name_, best_grid_search_ = max(self.grid_searches.items(), key = lambda x: x[1].best_score_) # x[0]: key, x[1]: value
        self.best_score_ = best_grid_search_.best_score_
        self.best_estimator_ = best_grid_search_.best_estimator_
        self.best_params_ = best_grid_search_.best_params_
        self.best_estimator_name_ = best_estimator_name_

    def predict(self, x):
        assert hasattr(self, 'best_estimator_'), 'There is no best_estimator_. Need to call "fit" first.'
        assert hasattr(self.best_estimator_,'predict'), 'Best estimator does not support "predict" method.'
        return self.best_estimator_.predict(x)
