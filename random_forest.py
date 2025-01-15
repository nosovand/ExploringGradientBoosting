# Implement Random Forest Classifier
from tree import CustomDecisionTreeClassifier, CustomDecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import r2_score

class CustomRandomForestClassifier:
    oob_score_ = 0.0
    def __init__(self, max_depth=1, min_samples_split=2, min_samples_leaf=1, max_features=None, n_estimators=100, n_jobs=1, tree_type='custom'):
        self.n_estimators = n_estimators
        self.estimators = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.scores = np.zeros(n_estimators)
        self.n_jobs = n_jobs
        self.tree_type = tree_type

    def __call__(self, *args, **kwargs):
        '''
        Initializes the tree with the new parameters
        '''
        return CustomRandomForestClassifier(*args, **kwargs)


    def fit(self, X, y):
        def build_tree(X, y):
            # Randomly sample with replacement
            bootstrap_idx = np.random.choice(len(X), len(X), replace=True)
            oob_idx = np.setdiff1d(np.arange(len(X)), bootstrap_idx)
            X_bootstrap, y_bootstrap = X.iloc[bootstrap_idx], y.iloc[bootstrap_idx]

            # Create and train a decision tree classifier
            if self.tree_type == 'custom':
                tree = CustomDecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                                    min_samples_split=self.min_samples_split, max_features=self.max_features)
            elif self.tree_type == 'scikit':
                tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                            min_samples_split=self.min_samples_split, max_features=self.max_features, random_state=42)
            else:
                raise ValueError('Invalid tree type')

            tree.fit(X_bootstrap, y_bootstrap)
            return tree, oob_idx

        results = Parallel(n_jobs=self.n_jobs)(delayed(build_tree)(X, y) for _ in range(self.n_estimators))
        self.estimators, oob_datasets_indices = zip(*results)

        def collect_votes(tree, X, oob_indices):
            predictions = tree.predict(X.iloc[oob_indices])
            return dict(zip(oob_indices, predictions))

        # Parallelize the votes collection
        votes_list = Parallel(n_jobs=self.n_jobs)(
            delayed(collect_votes)(tree, X, oob_datasets_indices[idx])
            for idx, tree in enumerate(self.estimators)
        )

        # Aggregate the votes
        vote_counts = {}
        for vote in votes_list:
            for idx, pred in vote.items():
                if idx not in vote_counts:
                    vote_counts[idx] = {}
                if pred in vote_counts[idx]:
                    vote_counts[idx][pred] += 1
                else:
                    vote_counts[idx][pred] = 1

        # Determine the final prediction based on majority vote
        final_predictions = np.zeros(len(X), dtype=int)
        for idx, counts in vote_counts.items():
            final_predictions[idx] = max(counts, key=counts.get)

        self.oob_score_ = np.mean(final_predictions[list(vote_counts.keys())] == y.iloc[list(vote_counts.keys())])
            

    def predict(self, X):
        # Make dictionary for each class and count the number of votes
        votes = {}
        def predict_tree(tree, X):
            return tree.predict(X)
        
        predictions = Parallel(n_jobs=self.n_jobs)(delayed(predict_tree)(tree, X) for tree in self.estimators)
        for i in range(len(X)):
            votes[i] = {}
            for prediction in predictions:
                if prediction[i] in votes[i]:
                    votes[i][prediction[i]] += 1
                else:
                    votes[i][prediction[i]] = 1
        
        # Get the class with the most votes
        final_predictions = []
        for i in range(len(X)):
            final_predictions.append(max(votes[i], key=votes[i].get))   
        return final_predictions
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def oob_score(self):
        return self.oob_score_
    

class CustomRandomForestRegressor:
    oob_score_ = 0.0
    
    def __init__(self, max_depth=1, min_samples_split=2, min_samples_leaf=1, max_features=None, n_estimators=100, n_jobs=1, tree_type='custom'):
        self.n_estimators = n_estimators
        self.estimators = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.scores = np.zeros(n_estimators)
        self.n_jobs = n_jobs
        self.tree_type = tree_type

    def __call__(self, *args, **kwargs):
        '''
        Initializes the tree with the new parameters
        '''
        return CustomRandomForestRegressor(*args, **kwargs)

    def fit(self, X, y):
        def build_tree(X, y):
            # Randomly sample with replacement
            bootstrap_idx = np.random.choice(len(X), len(X), replace=True)
            oob_idx = np.setdiff1d(np.arange(len(X)), bootstrap_idx)
            X_bootstrap, y_bootstrap = X.iloc[bootstrap_idx], y.iloc[bootstrap_idx]

            # Create and train a decision tree regressor
            if self.tree_type == 'custom':
                tree = CustomDecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                                    min_samples_split=self.min_samples_split, max_features=self.max_features)
            elif self.tree_type == 'scikit':
                tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                            min_samples_split=self.min_samples_split, max_features=self.max_features, random_state=42)
            else:
                raise ValueError('Invalid tree type')

            tree.fit(X_bootstrap, y_bootstrap)
            return tree, oob_idx

        results = Parallel(n_jobs=self.n_jobs)(delayed(build_tree)(X, y) for _ in range(self.n_estimators))
        self.estimators, oob_datasets_indices = zip(*results)

        def collect_predictions(tree, X, oob_indices):
            predictions = tree.predict(X.iloc[oob_indices])
            return dict(zip(oob_indices, predictions))

        # Parallelize the predictions collection
        predictions_list = Parallel(n_jobs=self.n_jobs)(
            delayed(collect_predictions)(tree, X, oob_datasets_indices[idx])
            for idx, tree in enumerate(self.estimators)
        )

        # Aggregate the predictions
        oob_predictions = {}
        for pred_dict in predictions_list:
            for idx, pred in pred_dict.items():
                if idx not in oob_predictions:
                    oob_predictions[idx] = []
                oob_predictions[idx].append(pred)

        # Calculate the mean prediction for each OOB sample
        final_predictions = np.zeros(len(X))
        for idx, preds in oob_predictions.items():
            final_predictions[idx] = np.mean(preds)

        # Calculate the OOB score using R2 error
        oob_indices = list(oob_predictions.keys())
        # self.oob_score_ = np.mean((final_predictions[oob_indices] - y.iloc[oob_indices]) ** 2)
        self.oob_score_ = r2_score(y.iloc[oob_indices], final_predictions[oob_indices])

    def predict(self, X):
        def predict_tree(tree, X):
            return tree.predict(X)

        # Collect predictions from all estimators
        predictions = Parallel(n_jobs=self.n_jobs)(delayed(predict_tree)(tree, X) for tree in self.estimators)

        # Average the predictions for the final output
        final_predictions = np.mean(predictions, axis=0)
        return final_predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return r2_score(y, predictions)

    def oob_score(self):
        return self.oob_score_