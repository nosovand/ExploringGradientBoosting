# Implement Gradient Boosting Regressor
from my_tree import CustomDecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
import numpy as np
from sklearn.metrics import r2_score

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

    def __call__(self, *args, **kwargs):
        '''
        Initializes the tree with the new parameters
        '''
        return CustomGradientBoostingRegressor(*args, **kwargs)

    def fit(self, X, y):
        self.trees = []
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
                raise ValueError('Invalid tree type', self.tree_type)
            tree.fit(X, residuals)
            self.trees.append(tree)
            # print('Tree', i, 'score:', self.score(X, y))
            # if self.tree_type == 'custom':
            #     tree.draw_tree()
            #     print("residuals", residuals)
            #     print("prediction", tree.predict(X))
            # if self.tree_type == 'scikit':
            #     plot_tree(tree, filled=True)
            #     print("residuals", residuals)
            #     print("prediction", tree.predict(X))

            f += self.learning_rate * tree.predict(X)

    def predict(self, X):
        f = np.full(X.shape[0], self.f0)
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)
        return f

    def score(self, X, y):
        return r2_score(y, self.predict(X))
    
    def print_params(self):
        print(f'n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, min_samples_leaf={self.min_samples_leaf}, max_features={self.max_features}, tree_type={self.tree_type}')