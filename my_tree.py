import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Any

class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=1, min_samples_split=2, min_samples_leaf=1):
        ''''
        Constructor for the DecisionTreeClassifier class.

        Parameters:
        max_depth: int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        min_samples_split: int, default=2
            The minimum number of samples required to split an internal node.
        '''
        if not isinstance(max_depth, int):
            
            raise ValueError(f"max_depth must be an integer. {max_depth} is not an integer.")
        else:
            self.max_depth = max_depth
        if not isinstance(min_samples_split, int):
            raise ValueError("min_samples_split must be an integer.")
        else:
            self.min_samples_split = min_samples_split
        if not isinstance(min_samples_leaf, int):
            raise ValueError("min_samples_leaf must be an integer.")
        else:
            self.min_samples_leaf = min_samples_leaf
    
    def __call__(self, *args, **kwargs):
        '''
        Initializes the tree with the new parameters
        '''
        return CustomDecisionTreeClassifier(*args, **kwargs)

    def fit(self, X, y):
        '''
        Build a decision tree classifier from the training set (X, y).

        Parameters:
        X: matrix of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples,)
            The target values.
        '''
        X = self._check_feature_format(X)

        y = np.array(y)
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        '''
        Recursively build the tree.

        Parameters:
        X: matrix of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples,)
            The target values.
        depth: int
            The current depth of the tree.
        '''
        # If the tree is empty, return None
        if self._empty_tree(y):
            return None

        # If the stopping condition is met, return the most common class
        if self._stop_condition(X, y, depth):
            return self._most_common_class(y)

        # Find the best split
        best_feature, best_value, best_gini = self._find_best_split(X, y)
        # Check if the best split is None
        if best_feature is None:
            return self._most_common_class(y)

        best_feature_name = X[0, best_feature]

        # Split the data based on the best split
        left_indices = X[1:, best_feature] < best_value
        right_indices = X[1:, best_feature] >= best_value

        #Add row 0 to left and right indices to keep track of the feature names
        left_indices = np.insert(left_indices, 0, True)
        right_indices = np.insert(right_indices, 0, True)

        # Delete used features
        # X = np.delete(X, best_feature, axis=1)

        # Check if number of samples in each split is bigger than min_samples_leaf
        if len(y[left_indices[1:]]) < self.min_samples_leaf or len(y[right_indices[1:]]) < self.min_samples_leaf:
            return self._most_common_class(y)

        # Recursively build the tree
        left_node = self._build_tree(X[left_indices], y[left_indices[1:]], depth+1)
        right_node = self._build_tree(X[right_indices], y[right_indices[1:]], depth+1)
        
        tree = Node(feature=best_feature, feature_name=best_feature_name, value=best_value, left=left_node, right=right_node, gini=best_gini)

        return tree
    
    def _stop_condition(self, X, y, depth) -> bool:
        ''''
        Check if the stopping condition is met.
        Stopping conditions:
        - The current depth of the tree is equal to the maximum depth.
        - The number of samples is less than the minimum number of samples required to split an internal node.
        - The node is pure, i.e., all samples belong to the same class.

        Parameters:
        X: matrix of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples,)
            The target values.
        depth: int
            The current depth of the tree.
        '''
        return depth == self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1
    
    def _empty_tree(self, y):
        '''
        Check if the tree is empty.

        Parameters:
        y: array-like of shape (n_samples,)
            The target values.
        '''
        return len(y) == 0
    
    def _most_common_class(self, y):
        '''
        Return the most common class in the target values.

        Parameters:
        y: array-like of shape (n_samples,)
            The target values.
        '''
        unique_elements, counts = np.unique(y, return_counts=True)
        max_count_index = np.argmax(counts)
        most_common_element = unique_elements[max_count_index]
        return most_common_element
    
    def _find_best_split(self, X, y):  
        '''
        Find the best split for the data by iterating over all features and values 
        to find the one that minimizes the Gini impurity.

        Parameters:
        X: matrix of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples,)
            The target values.
        '''
        best_gini = 1
        best_feature = None
        best_value = None

        # Iterate over all features and values to find the best split
        for feature in range(X.shape[1]):
            for value in np.array(list(set(X[1:, feature]))):
                # Split the data
                left_indices = X[1:, feature] < value
                right_indices = X[1:, feature] >= value
                # Check if number of samples in each split is bigger than min_samples_leaf
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    # Skip splits with less than min_samples_leaf samples
                    # print("Skipping split with less than min_samples_leaf samples")
                    continue
                # Calculate the Gini impurity
                gini = self._calculate_gini(y[left_indices], y[right_indices])
                # print("test gini")
                # print(gini)
                # Update the best split if the current split is better
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_value = value
        return best_feature, best_value, best_gini
    
    def _calculate_gini(self, left_y, right_y):
        '''
        Calculate the Gini impurity for the given split.

        Parameters:
        left_y: array-like
            The target values for the left node.
        right_y: array-like
            The target values for the right node.
        '''
        #If left or right node is empty, return 1
        if len(left_y) == 0 or len(right_y) == 0:
            return 1
        #extract number of classes from left and right nodes
        classes = np.unique(np.concatenate((left_y, right_y), axis = None))
        #2D array of size len(classes) x 2 to store gini impurity for each class and each node
        gini = np.zeros((len(classes), 2))
        for i in range(len(classes)):
            #calculate ratio of class i in left node
            gini[i, 0] = np.sum(left_y == classes[i]) / len(left_y)
            #calculate ratio of class i in right node
            gini[i, 1] = np.sum(right_y == classes[i]) / len(right_y)

        #Calculate gini impurity for left and right nodes
        left_gini = 1 - np.sum(gini[:, 0] ** 2)
        right_gini = 1 - np.sum(gini[:, 1] ** 2)

        left_size = len(left_y)
        right_size = len(right_y)
        #Calculated total size of left and right nodes
        total_size = left_size + right_size

        #Calculate weighted gini impurity
        gini = (left_size / total_size) * left_gini + (right_size / total_size) * right_gini
        return gini
    
    def predict(self, X):
        '''
        Predict class for X.

        Parameters:
        X: matrix of shape (n_samples, n_features)
            The input samples.
        '''
        X = self._check_feature_format(X)[1:]
        return np.array([self._predict_tree(x, self.tree) for x in X])
    
    def _predict_tree(self, x, node):
        '''
        Recursively predict the class for a given
        input sample x.

        Parameters:
        x: array-like of shape (n_features,)
            The input sample.
        node: Node
            The current node in the tree.
        '''
        
        # If the node is a leaf, return the predicted class
        if not isinstance(node, Node):
            return node
        # Recursively traverse the tree
        # Check the used feature is continuous or categorical
        if x[node.feature] < node.value:
            #delete the feature used to split the node
            # x = np.delete(x, node.feature)
            return self._predict_tree(x, node.left)
        else:
            #delete the feature used to split the node
            # x = np.delete(x, node.feature)
            return self._predict_tree(x, node.right)
        
    def _check_feature_format(self, data):
        """
        Converts a DataFrame to a NumPy matrix.

        Args:
        data (DataFrame): The DataFrame to be converted.

        Returns:
        numpy.ndarray: NumPy matrix containing the values of the DataFrame.
        """
        # Check if the input data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        # Add the feature names as the first row
        column_names = data.columns.tolist()
        return np.vstack([column_names, data.values])
    
    def draw_tree(self):
        '''
        Draw the tree.
        '''
        self._draw_tree(self.tree)

    def _draw_tree(self, node, depth=0):
        '''
        Recursively draw the tree using visual indentation.

        Parameters:
        node: Node
            The current node in the tree.
        depth: int
            The current depth of the tree.
        '''
        # If the node is a leaf, print the predicted class
        if not isinstance(node, Node):
            print("  " * depth, node)
            return
        # Recursively draw the tree
        # If value is continuous, print the feature and value
        print("  " * depth, f"Feature {node.feature_name} < {node.value}")
        # Print gini for the node
        # print("  " * depth, f"Gini: {node.gini}")
        self._draw_tree(node.left, depth + 1)
        print("  " * depth, f"Feature {node.feature_name} >= {node.value}")
        # Print gini for the node
        # print("  " * depth, f"Gini: {node.gini}")

    def score(self, X, y):
        predictions = self.predict(X)
        score = np.mean(predictions == y)
        return score

    def get_params(self, deep=True):
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class Node:
    '''
    A class to represent a node in the decision tree.
    '''
    def __init__(self, feature=None, feature_name=None, value=None, left=None, right=None, gini=None):
        self.feature = feature
        self.feature_name = feature_name
        self.value = value
        self.left = left
        self.right = right
        self.gini = gini
