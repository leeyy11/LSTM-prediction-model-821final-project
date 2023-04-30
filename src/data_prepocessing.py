import pandas as pd
from sklearn.decomposition import TruncatedSVD


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


def load_data():
    # Load patient data and diagnoses
    pat = pd.read_csv('data/patients.csv')
    diag = pd.read_csv('data/diagnoses_icd.csv')

    # Merge patient and diagnosis data
    data = pat[['subject_id', 'anchor_age']].merge(diag, on='subject_id', how='right')
    data = data[data['icd_version'] == 10][['subject_id', 'icd_code', 'anchor_age']]
    data = data.rename(columns={'subject_id': 'PAT_KEY', 'icd_code': 'ICD_CODE', 'anchor_age': 'age'})

    # Load diagnosis codes and merge with data
    ICD = pd.read_csv('DXCCSR.csv')
    ICD = ICD[["'ICD-10-CM CODE'",
               "'CCSR CATEGORY 1 DESCRIPTION'",
               "'CCSR CATEGORY 2 DESCRIPTION'",
               "'CCSR CATEGORY 3 DESCRIPTION'",
               "'CCSR CATEGORY 4 DESCRIPTION'",
               "'CCSR CATEGORY 5 DESCRIPTION'",
               "'CCSR CATEGORY 6 DESCRIPTION'"]]
    ICD = ICD.replace("'", "", regex=True)
    ICD = pd.melt(ICD,
                  id_vars=["'ICD-10-CM CODE'"],
                  value_vars=["'CCSR CATEGORY 1 DESCRIPTION'",
                              "'CCSR CATEGORY 2 DESCRIPTION'",
                              "'CCSR CATEGORY 3 DESCRIPTION'",
                              "'CCSR CATEGORY 4 DESCRIPTION'",
                              "'CCSR CATEGORY 5 DESCRIPTION'",
                              "'CCSR CATEGORY 6 DESCRIPTION'"])

    # data['ICD_CODE'] = data['ICD_CODE'].str.replace('.', '')
    diag_m1 = data.merge(ICD, left_on='ICD_CODE', right_on="'ICD-10-CM CODE'")
    return diag_m1


def create_diag_m1():
    diag_m1 = load_data()
    # Filter and pivot the data
    diag_m1['count'] = 1
    diag_m1 = diag_m1[["PAT_KEY", "value", "count"]]
    diag_m1 = diag_m1.pivot_table(index='PAT_KEY', columns='value', values='count', fill_value=0)
    diag_m1.to_csv("diag_m1_new.csv")
    return diag_m1



def main():
    # Load data
    data = load_data()

    # Create diag_m1
    diag_m1 = create_diag_m1()

    # Create proc_code
    # proc_code = create_proc_code()

    # Perform PCA on diag_m1
    pca = PCAClassifier(n_components=50)
    features = pca.fit_transform(diag_m1)
    sample_ids = diag_m1.index
    df = pd.DataFrame(data=features, index=sample_ids)
    print(df.head())


if __name__ == '__main__':
    main()
