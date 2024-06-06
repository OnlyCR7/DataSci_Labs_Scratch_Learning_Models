from decisiontree import DecisionTree
from collections import Counter
import numpy as np

class RandomForest :
    def __init__(self, n_trees = 2, max_depth = 10, min_sample_split = 10, n_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.trees = []
        
        
        
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth = self.max_depth, min_sample_split = self.min_sample_split, n_features = self.n_features)
            X_samples, y_samples = self._bootstrap_samples(X, y)
            tree.fit(X_samples, y_samples)
            self.trees.append(tree)
        
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        indx = np.random.choice(n_samples, n_samples, replace = True)
        return X[indx], y[indx]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
        
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
        