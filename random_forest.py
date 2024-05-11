# Implement Random Forest Classifier
from my_tree import CustomDecisionTreeClassifier
import numpy as np

class RandomForestClassifier:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Perform bootstrapping
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            # Create and train a decision tree classifier
            tree = CustomDecisionTreeClassifier(max_features='sqrt')
            tree.fit(X_bootstrap, y_bootstrap)

            # Add the trained tree to the list of estimators
            self.estimators.append(tree)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree in self.estimators:
            predictions += tree.predict(X)
        return np.round(predictions / self.n_estimators)