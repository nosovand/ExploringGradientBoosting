# Implement Gradient Boosting Regressor
from my_tree import CustomDecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
import numpy as np
from sklearn.metrics import r2_score

class CustomGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None, tree_type='custom'):
        '''
        Initializes the Gradient Boosting Regressor with the given parameters

        Parameters:
        n_estimators (int): The number of trees to use in the ensemble
        learning_rate (float): The learning rate to use for the ensemble
        max_depth (int): The maximum depth of the trees
        min_samples_split (int): The minimum number of samples required to split an internal node
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node
        max_features (int/str): The number of features to consider when looking for the best split, or 'sqrt', 'log2'
        tree_type (str): The type of tree to use ('custom' or 'scikit')    
        '''
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
        '''
        Fits the Gradient Boosting Regressor to the given data

        Parameters:
        X The input data
        y The target values
        '''

        self.trees = []
        # Initialize the prediction with the mean of the target values
        self.f0 = np.mean(y)
        f = np.full_like(y, self.f0)
        
        for i in range(self.n_estimators):
            # Calculate the residuals for the current iteration
            residuals = y - f
            # Fit the trees to the residuals
            if self.tree_type == 'custom':
                tree = CustomDecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                                   min_samples_split=self.min_samples_split, max_features=self.max_features)
            elif self.tree_type == 'scikit':
                tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                             min_samples_split=self.min_samples_split, max_features=self.max_features)
            else:
                raise ValueError('Invalid tree type', self.tree_type)
            tree.fit(X, residuals)

            # Add the tree to the ensemble
            self.trees.append(tree)

            # Update the predictions
            f += self.learning_rate * tree.predict(X)

    def predict(self, X):
        '''
        Predicts the target values for the given input data

        Parameters:
        X The input data

        Returns:
        The predicted target values
        '''
        f = np.full(X.shape[0], self.f0)
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)
        return f

    def score(self, X, y):
        '''
        Returns the R^2 score of the model

        Parameters:
        X The input data
        y The target values

        Returns:
        The R^2 score of the model
        '''
        return r2_score(y, self.predict(X))
    
    def print_params(self):
        print(f'n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, min_samples_leaf={self.min_samples_leaf}, max_features={self.max_features}, tree_type={self.tree_type}')