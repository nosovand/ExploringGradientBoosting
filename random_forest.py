# Implement Random Forest Classifier
from my_tree import CustomDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from joblib import Parallel, delayed

class CustomRandomForestClassifier:
    def __init__(self, max_depth=1, min_samples_split=2, min_samples_leaf=1, max_features=None, n_estimators=100, n_jobs=1):
        self.n_estimators = n_estimators
        self.estimators = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.scores = np.zeros(n_estimators)
        self.n_jobs = n_jobs

    def fit(self, X, y):

        def build_tree(X, y, i):
            # Randomly sample with replacement
            bootstrap_idx = np.random.choice(len(X), len(X), replace=True)
            oob_idx = [i for i in range(len(X)) if i not in bootstrap_idx]
            X_bootstrap = X.iloc[bootstrap_idx]
            y_bootstrap = y.iloc[bootstrap_idx]
            X_oob = X.iloc[oob_idx]
            y_oob = y.iloc[oob_idx]
            # Create and train a decision tree classifier
            tree = CustomDecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split, max_features=self.max_features)
            tree.fit(X_bootstrap, y_bootstrap)
            oob_score = tree.score(X_oob, y_oob)
            return tree, oob_score
        
        results = Parallel(n_jobs=self.n_jobs)(delayed(build_tree)(X, y, i) for i in range(self.n_estimators))
        self.estimators, self.scores = zip(*results)
            

    def predict(self, X):
        # Make dictionary for each class and count the number of votes
        votes = {}
        def predict_tree(tree, X):
            return tree.predict(X)
        
        predictions = Parallel(n_jobs=self.n_jobs)(delayed(predict_tree)(tree, X) for tree in self.estimators)
        votes = {}
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
        return predictions
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def oob_score(self):
        return np.mean(self.scores)