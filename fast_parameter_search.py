import numpy as np
import itertools
from joblib import Parallel, delayed

class MyGridSearchCV():
    def __init__(self, model, param_grid, cv, n_jobs=-1):
        ''''
        model: a class of model to be used
        param_grid: a dictionary with parameters to be tested
        cv: a class of cross-validation to be used
        '''
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.best_params = dict()
        self.best_score = 0.0
        self.best_estimator = None
        self.cv_results = None
        self.n_jobs = n_jobs
    
    def fit(self, X, y):
            param_combinations = self._generate_param_combinations()

            def evaluate_params(params):
                cv_scores = np.zeros(self.cv.get_n_splits())
                for i, (train_index, test_index) in enumerate(self.cv.split(X, y)):
                    clf = self.model(**params)
                    clf.fit(X.iloc[train_index], y.iloc[train_index])
                    cv_scores[i] = clf.score(X.iloc[test_index], y.iloc[test_index])
                return np.mean(cv_scores)

            scores = Parallel(n_jobs=self.n_jobs)(delayed(evaluate_params)(params) for params in param_combinations)

            best_index = np.argmax(scores)
            self.best_params = param_combinations[best_index]
            self.best_score = scores[best_index]
            self.best_estimator = self.model(**self.best_params)
   

    def _generate_param_combinations(self):
        '''
        Generate all possible combinations of parameters to be tested

        Returns:
        List of dictionaries, where each dictionary represents a unique combination of parameters
        '''
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        param_combinations = list(itertools.product(*values))
        param_dicts = [dict(zip(keys, combination)) for combination in param_combinations]
        return param_dicts
    
    
        