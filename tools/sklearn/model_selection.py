from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

def train_val_test_split(x, y):
    ss = ShuffleSplit()
    return

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
