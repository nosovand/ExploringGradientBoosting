import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from typing import Any
import copy

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
        if max_features is not None and not isinstance(max_features, int) and not isinstance(max_features, str):
            raise ValueError("max_features must be an integer or string.")
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
        elif isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                self.max_features = int(np.sqrt(X.shape[1]))
            elif self.max_features == "log2":
                self.max_features = int(np.log2(X.shape[1]))
            else:
                raise ValueError("max_features must be an integer, 'sqrt' or 'log2'.")
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
        if best_feature is None or best_value is None:
            return self._most_common_class(y)

        best_feature_name = X[0, best_feature]

        # Split the data based on the best split
        left_indices = X[1:, best_feature] <= best_value
        right_indices = X[1:, best_feature] > best_value

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
        # print("test stop condition")
        return depth == self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1
    
    def _empty_tree(self, y) -> bool:
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
        number_of_features = X.shape[1]
        best_gini = 1
        best_feature = None
        best_value = None
        self.current_node_num_samples = len(y)
        self.current_node_unique_classes = np.unique(y)
        self.current_node_num_classes = len(self.current_node_unique_classes)

        # Count the number of samples in each class in a dictionary with class as key and number of samples as value
        current_node_num_samples_in_classes = {}
        current_node_num_samples_in_classes['left'] = {label: 0 for label in self.current_node_unique_classes}
        current_node_num_samples_in_classes['right'] = {label: 0 for label in self.current_node_unique_classes}
        for label in y:
            current_node_num_samples_in_classes['right'][label] += 1

        # Keep only max_features number of features
        if self.max_features < number_of_features:
            random_indices = np.random.choice(X.shape[1], self.max_features, replace=False)
            X = X[:, random_indices]

        # Sort the data along each feature
        sorted_indices = np.argsort(X[1:, :], axis=0)
        sorted_X = np.take_along_axis(X[1:, :], sorted_indices, axis=0)
        # sorted_y = np.take_along_axis(y, sorted_indices[:, 0], axis=0)

        for feature in range(X.shape[1]):
            sorted_y = np.take_along_axis(y, sorted_indices[:, feature], axis=0)
            left_sizes = np.arange(1, X.shape[0])
            right_sizes = self.current_node_num_samples - left_sizes

            left_class_counts = np.zeros((self.current_node_num_classes, X.shape[0]-1))
            right_class_counts = np.zeros((self.current_node_num_classes, X.shape[0]-1))

            # Construct left and right class counts for all possible splits
            for c, label in enumerate(self.current_node_unique_classes):
                left_class_counts[c] = np.cumsum(sorted_y == label, axis=0)
                right_class_counts[c] = current_node_num_samples_in_classes['right'][label] - left_class_counts[c]

            # Change first min_samples_leaf and last min_samples_leaf values of left and right sizes to 1
            left_sizes[:self.min_samples_leaf] = 1
            right_sizes[:self.min_samples_leaf] = 1
            left_sizes[-self.min_samples_leaf:] = 1
            right_sizes[-self.min_samples_leaf:] = 1

            # Calculate Gini impurity for all possible splits
            left_ginis = 1 - np.sum((left_class_counts / left_sizes) ** 2, axis=0)
            right_ginis = 1 - np.sum((right_class_counts / right_sizes) ** 2, axis=0)

            gini_values = (left_sizes / self.current_node_num_samples) * left_ginis + (right_sizes / self.current_node_num_samples) * right_ginis

            # Change the gini values of the first min_samples_leaf and last min_samples_leaf values to 1
            gini_values[:self.min_samples_leaf] = 1
            gini_values[-self.min_samples_leaf:] = 1

            # print("gini_values")
            # print(gini_values)

            # Update best split if the current split is better
            min_gini_idx = np.argmin(gini_values)
            # print("min_gini_idx")
            # print(min_gini_idx)
            if gini_values[min_gini_idx] < best_gini:
                best_gini = gini_values[min_gini_idx]
                best_feature = feature

                if min_gini_idx == len(sorted_X)-1:
                    best_value = sorted_X[min_gini_idx, feature]
                else:
                    best_value = (sorted_X[min_gini_idx, feature].astype(float) + sorted_X[min_gini_idx + 1, feature].astype(float)) / 2.0
                    best_value = str(best_value)

        # Translate the index of the feature to the original index if max_features < number_of_features
        if self.max_features < number_of_features:
            best_feature = random_indices[best_feature]

        return best_feature, best_value, best_gini

    
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
        if x[node.feature] <= node.value:
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
        print("  " * depth, f"Feature {node.feature_name} <= {node.value}")
        # Print gini for the node
        # print("  " * depth, f"Gini: {node.gini}")
        self._draw_tree(node.left, depth + 1)
        print("  " * depth, f"Feature {node.feature_name} > {node.value}")
        # Print gini for the node
        # print("  " * depth, f"Gini: {node.gini}")
        self._draw_tree(node.right, depth + 1)

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

class CustomDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth=1, min_samples_split=2, min_samples_leaf=1, max_features=None):
        '''
        Constructor for the DecisionTreeRegressor class.

        Parameters:
        max_depth: int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        min_samples_split: int, default=2
            The minimum number of samples required to split an internal node.
        min_samples_leaf: int, default=1
            The minimum number of samples required to be at a leaf node.
        max_features: int or string, default=None
            The number of features to consider when looking for the best split. If None, then all features will be considered. If 'sqrt', then max_features=sqrt(n_features). If 'log2', then max_features=log2(n_features).
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
        if max_features is not None and not isinstance(max_features, int) and not isinstance(max_features, str):
            raise ValueError("max_features must be an integer or string.")
        else:
            self.max_features = max_features
    
    def __call__(self, *args, **kwargs):
        '''
        Initializes the tree with the new parameters
        '''
        return CustomDecisionTreeRegressor(*args, **kwargs)

    def fit(self, X, y):
        '''
        Build a decision tree regressor from the training set (X, y).

        Parameters:
        X: matrix of shape (n_samples + feature_name, n_features)
            The training input samples.
        y: array-like of shape (n_samples)
            The target values.
        '''
        X = self._check_feature_format(X)
        y = np.array(y)
        if self.max_features is None:
            self.max_features = X.shape[1]
        elif isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                self.max_features = int(np.sqrt(X.shape[1]))
            elif self.max_features == "log2":
                self.max_features = int(np.log2(X.shape[1]))
            else:
                raise ValueError("max_features must be an integer, 'sqrt' or 'log2'.")
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        '''
        Recursively build the tree.

        Parameters:
        X: matrix of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples)
            The target values.
        depth: int
            The current depth of the tree.
        
        Returns:
        Node: The root node of the tree.
        '''

        # If the tree is empty, return None
        # Used only for debugging purposes
        if self._empty_tree(y):
            return None
        
        # If the stopping condition is met, return the mean of the target values
        if self._stop_condition(y, depth):
            return np.mean(y.astype(float)).astype(float)

        # Find the best split
        best_feature, best_value, best_mse = self._find_best_split(X, y)

        # Check if the best split is None
        if best_feature is None or best_value is None:
            # Best split was not found, return the mean of the target values
            return np.mean(y.astype(float)).astype(float)

        left_indices = X[1:, best_feature].astype(float) <= best_value
        right_indices = X[1:, best_feature].astype(float) > best_value
        left_indices = np.insert(left_indices, 0, True)
        right_indices = np.insert(right_indices, 0, True)

        # Check for the min_samples_leaf condition
        if len(y[left_indices[1:]]) < self.min_samples_leaf or len(y[right_indices[1:]]) < self.min_samples_leaf:
            return np.mean(y.astype(float)).astype(float)

        # Recursively build the tree
        left_node = self._build_tree(X[left_indices], y[left_indices[1:]], depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices[1:]], depth + 1)
        
        # Create a node
        best_feature_name = X[0, best_feature]
        tree = Node(feature=best_feature, feature_name=best_feature_name, value=best_value, left=left_node, right=right_node)
        return tree
    
    def _stop_condition(self, y, depth) -> bool:
        '''
        Check if the stopping condition is met.
        Stopping conditions:
        - The current depth of the tree is equal to the maximum depth.
        - The number of samples is less than the minimum number of samples required to split an internal node.

        Parameters:
        y: array-like of shape (n_samples)
            The target values.
        depth: int
            The current depth of the tree.
        
        Returns:
        bool: True if the stopping condition is met, False otherwise.
        '''
        return depth == self.max_depth or len(y) < self.min_samples_split
    
    def _empty_tree(self, y) -> bool:
        '''
        Check if the tree is empty.

        Parameters:
        y: array-like of shape (n_samples,)
            The target values.

        Returns:
        bool: True if the tree is empty, False otherwise.
        '''
        return len(y) == 0
    
    def _find_best_split(self, X, y):
        '''
        Find the best split for the data by iterating over all features and values
        to find the one that minimizes the mean squared error.

        Parameters:
        X: matrix of shape (n_samples + feature_name, n_features)
            The training input samples.
        y: array-like of shape (n_samples,)
            The target values.
        
        Returns:
        int: The index of the best feature.
        float: The best value to split the data.
        float: The best mean squared error.
        '''

        number_of_features = X.shape[1]
        best_mse = float("inf")
        best_feature = None
        best_value = None
        self.current_node_num_samples = len(y)

        # Keep only max_features number of features
        if self.max_features < number_of_features:
            random_indices = np.random.choice(X.shape[1], self.max_features, replace=False)
            X = X[:, random_indices]

        # Sort the data along each feature
        sorted_indices = np.argsort(X[1:, :].astype(float), axis=0)        
        sorted_X = np.take_along_axis(X[1:, :], sorted_indices, axis=0)

        for feature in range(X.shape[1]):
            sorted_y = np.take_along_axis(y, sorted_indices[:, feature], axis=0)
            # We want to keep only the unique values of the feature for splitting
            sorted_x_uniq, counts_x = np.unique(sorted_X[:, feature].astype(float), return_counts=True)
            if len(sorted_x_uniq) == 1:
                continue
            
            # Calculate the sizes of the left and right splits
            left_sizes = np.append([0], np.cumsum(counts_x))
            right_sizes = self.current_node_num_samples - left_sizes

            # Compute the cumulative sum of sorted_y for unique x values
            cumsum_sorted_y = np.cumsum(sorted_y)
            end_indices = np.cumsum(counts_x)
            start_indices = np.roll(end_indices, shift=1)
            start_indices[0] = 0
            sorted_y_sum_uniq_x = cumsum_sorted_y[end_indices - 1]
            rev_sorted_y_sum_uniq_x = sorted_y_sum_uniq_x[-1] - sorted_y_sum_uniq_x
            
            # Calculate the means of the left and right splits
            left_means = sorted_y_sum_uniq_x[:-1] / left_sizes[1:-1]
            right_means = rev_sorted_y_sum_uniq_x[:-1] / right_sizes[1:-1]

            # Calculate the MSE for all possible splits using only the unique values of the feature
            left_mse_array = np.zeros(len(left_means))
            right_mse_array = np.zeros(len(right_means))
            
            for i in range(len(left_means)):
                left_mse_array[i] = np.mean((sorted_y[:left_sizes[i+1]] - left_means[i]) ** 2)
                right_mse_array[i] = np.mean((sorted_y[left_sizes[i+1]:] - right_means[i]) ** 2)
            
            mse_array = (left_sizes[1:-1] / self.current_node_num_samples) * left_mse_array + \
                        (right_sizes[1:-1] / self.current_node_num_samples) * right_mse_array
            
            
            # Choose the best split while checking for the min_samples_leaf condition using the mse_array
            splits_considered = 0
            while splits_considered < len(mse_array):
                min_mse_idx = np.argmin(mse_array)
                if left_sizes[min_mse_idx+1] < self.min_samples_leaf or right_sizes[min_mse_idx+1] < self.min_samples_leaf:
                    mse_array[min_mse_idx] = float("inf")
                    splits_considered += 1
                else:
                    break
            
            if mse_array[min_mse_idx] < best_mse:
                best_mse = mse_array[min_mse_idx]
                best_feature = feature
                # Since the mse_array is calculated for the unique values of the feature,\
                #  we need to find the index of the best value in the original sorted_X
                best_value_idx = np.cumsum(counts_x)[min_mse_idx] - 1
                best_value = sorted_X[best_value_idx, feature].astype(float)

        # Translate the index of the feature to the original index if max_features < number_of_features
        if self.max_features < number_of_features:
            best_feature = random_indices[best_feature]

        return best_feature, best_value, best_mse

    def predict(self, X):
        '''
        Predict target values for X.

        Parameters:
        X: matrix of shape (n_samples + feature_name, n_features)
            The input samples.
        
        Returns:
        array-like of shape (n_samples)
            The predicted target values.
        '''
        X = self._check_feature_format(X)[1:]
        return np.array([self._predict_tree(x, self.tree) for x in X])
    
    def _predict_tree(self, x, node):
        '''
        Recursively predict the target value for a given input sample x.

        Parameters:
        x: array-like of shape (n_features,)
            The input sample.
        node: Node
            The current node in the tree.

        Returns:
        Any: The predicted target value.
        '''

        # If the node is a leaf, return the predicted target value
        if not isinstance(node, Node):
            return node
        
        # Recursively traverse the tree until a leaf is reached
        if x[node.feature].astype(float) <= node.value:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)
        
    def _check_feature_format(self, data):
        '''
        Converts a DataFrame to a NumPy matrix.

        Parameters:
        data (DataFrame): The DataFrame to be converted.

        Returns:
        numpy.ndarray: NumPy matrix containing the values of the DataFrame.
        '''

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
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

        # If the node is a leaf, print the predicted target value
        if not isinstance(node, Node):
            print("  " * depth, node)
            return
        
        # Recursively draw the tree
        print("  " * depth, f"Feature {node.feature} < {node.value}")
        self._draw_tree(node.left, depth + 1)
        print("  " * depth, f"Feature {node.feature} >= {node.value}")
        self._draw_tree(node.right, depth + 1)

    def score(self, X, y):
        '''
        Return the coefficient of determination R^2 of the prediction.

        Parameters:
        X: matrix of shape (n_samples + feature_name, n_features)
            The input samples.
        y: array-like of shape (n_samples)
            The target values.

        Returns:
        float: The coefficient of determination R^2 of the prediction.
        '''

        predictions = self.predict(X)
        u = np.sum((predictions - y) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u/v

    def get_params(self, deep=True):
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split, "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features}

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


