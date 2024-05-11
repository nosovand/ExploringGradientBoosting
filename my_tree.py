import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Any

class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=1, min_samples_split=2, min_samples_leaf=1, max_features=None):
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
        if max_features is not None and not isinstance(max_features, int):
            raise ValueError("max_features must be an integer.")
        else:
            self.max_features = max_features
    
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
        # self.num_classes = len(np.unique(y))
        # self.num_samples = len(y)
        self.unique_classes = np.unique(y)
        X = self._check_feature_format(X)
        y = np.array(y)
        if self.max_features is None:
            self.max_features = X.shape[1]
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
        self.current_node_num_samples = len(y)
        self.current_node_unique_classes = np.unique(y)
        self.current_node_num_classes = len(self.current_node_unique_classes)
        # Count the number of samples in each class in a dictionary with class as key and number of samples as value
        self.current_node_num_of_samples_in_classes = {}
        for i in range(self.current_node_num_classes):
            self.current_node_num_of_samples_in_classes[self.current_node_unique_classes[i]] = np.sum(y == self.current_node_unique_classes[i])

        # Keep only max_features number of features
        if self.max_features < X.shape[1]:
            random_indices = np.random.choice(X.shape[1], self.max_features, replace=False)
            X = X[:, random_indices]
            print(X)

        # Iterate over all features and values to find the best split
        for feature in range(X.shape[1]):
            # Initialize left indices as an array of False values 
            left_indices = np.zeros(len(X[1:, feature]), dtype=bool)
            # Initialize right indices as an array of True values
            right_indices = np.ones(len(X[1:, feature]), dtype=bool)
            # Sort the data from smallest to largest
            X_sorted_idx = X[1:, feature].argsort()
            X_sorted = X[1:, feature][X_sorted_idx]
            y_sorted = y[X_sorted_idx]
            # Initialize the number of samples in each class in the left and right nodes
            self.current_split_num_samples_in_classes = {}
            self.current_split_num_samples_in_classes['left'] = {}
            self.current_split_num_samples_in_classes['right']  = {}
            for i in range(self.current_node_num_classes):
                self.current_split_num_samples_in_classes['left'][self.current_node_unique_classes[i]] = 0.0
                self.current_split_num_samples_in_classes['right'][self.current_node_unique_classes[i]] = self.current_node_num_of_samples_in_classes[self.current_node_unique_classes[i]]
            # Tmp variable to store the value of the previous value
            value_stored = None

            for idx, value in enumerate(X_sorted):
                # Split the data
                left_indices[idx] = True
                right_indices[idx] = False
                # Update the number of samples in each class
                self.current_split_num_samples_in_classes['left'][y_sorted[idx]] += 1
                self.current_split_num_samples_in_classes['right'][y_sorted[idx]] -= 1
                # Check if the value is the same as the previous value
                if value_stored == value:
                    continue
                value_stored = value
                # Check if number of samples in each split is bigger than min_samples_leaf
                if idx+1 < self.min_samples_leaf or self.current_node_num_samples-idx-1 < self.min_samples_leaf:
                    # Skip splits with less than min_samples_leaf samples
                    # print("Skipping split with less than min_samples_leaf samples")
                    continue
                # Calculate the Gini impurity
                gini = self._calculate_gini(y_sorted[left_indices], y_sorted[right_indices], idx)
                # print("test gini")
                # print(gini)
                # Update the best split if the current split is better
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_value = value
        return best_feature, best_value, best_gini
    
    def _calculate_gini(self, left_y, right_y, idx):
        '''
        Calculate the Gini impurity for the given split.

        Parameters:
        left_y: array-like
            The target values for the left node.
        right_y: array-like
            The target values for the right node.
        '''
        left_size = idx + 1
        right_size = self.current_node_num_samples - idx - 1
        #If left or right node is empty, return 1
        if left_size == 0 or right_size == 0:
            return 1
        #2D array of size len(classes) x 2 to store ratio of class i in left and right nodes
        ratios = np.zeros((self.current_node_num_classes, 2))
        for i, label in enumerate(self.current_node_unique_classes):
            #calculate ratio of class i in left node
            ratios[i, 0] = self.current_split_num_samples_in_classes['left'][label] / left_size
            #calculate ratio of class i in right node
            ratios[i, 1] = self.current_split_num_samples_in_classes['right'][label] / right_size

        #Calculate gini impurity for left and right nodes
        left_gini = 1 - np.sum(ratios[:, 0] ** 2)
        right_gini = 1 - np.sum(ratios[:, 1] ** 2)

        #Calculate weighted gini impurity
        gini = (left_size / self.current_node_num_samples) * left_gini + (right_size / self.current_node_num_samples) * right_gini
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
            return self._predict_tree(x, node.left)
        else:
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
