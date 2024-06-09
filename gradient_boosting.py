# Implement Gradient Boosting Regressor
from my_tree import CustomDecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class CustomGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None, tree_type='custom'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree_type = tree_type
        self.f0 = None

    def fit(self, X, y):
        self.f0 = np.mean(y)
        f = np.full_like(y, self.f0)
        for i in range(self.n_estimators):
            residuals = y - f
            # print(i, np.mean(residuals))
            if self.tree_type == 'custom':
                tree = CustomDecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                                   min_samples_split=self.min_samples_split, max_features=self.max_features)
            elif self.tree_type == 'scikit':
                tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                             min_samples_split=self.min_samples_split, max_features=self.max_features)
            else:
                raise ValueError('Invalid tree type')
            tree.fit(X, residuals)
            self.trees.append(tree)
            f += self.learning_rate * tree.predict(X)

    def predict(self, X):
        f = np.full(X.shape[0], self.f0)
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)
        return f
