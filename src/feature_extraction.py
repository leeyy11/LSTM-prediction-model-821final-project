from sklearn.decomposition import TruncatedSVD
from data_prepocessing import load_data,create_diag_m1
import pandas as pd

class PCAClassifier:
    def __init__(self, n_components=1):
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

def get_feature():
    # Load data
    data = load_data()

    # Create diag_m1
    diag_m1 = create_diag_m1()

    # Create proc_code
    # proc_code = create_proc_code()

    # Perform PCA on diag_m1
    pca = PCAClassifier(n_components=10)
    features = pca.fit_transform(diag_m1)
    sample_ids = diag_m1.index
    df = pd.DataFrame(data=features, index=sample_ids)
    return df


if __name__ == '__main__':
    main()