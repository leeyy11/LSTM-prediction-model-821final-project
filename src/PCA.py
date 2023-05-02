"""Define PCAClassifier."""
from sklearn.decomposition import TruncatedSVD
import pandas as pd


class PCAClassifier:
    """Define a PCAClassifier."""

    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca = None

    def fit(self, X, y=None):
        self.pca = TruncatedSVD(n_components=self.n_components)
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
