"""Clinical data with ICD code preprocessing."""
import pandas as pd
from PCA import PCAClassifier
import numpy as np


class DataProcessor:
    """Define a dataprocessor."""

    def __init__(self, diag_filename="DXCCSR.csv"):
        """Define the ICD reference file."""
        self.reference = pd.read_csv(diag_filename)
        self.reference = self.reference[
            [
                "'ICD-10-CM CODE'",
                "'CCSR CATEGORY 1 DESCRIPTION'",
                "'CCSR CATEGORY 2 DESCRIPTION'",
                "'CCSR CATEGORY 3 DESCRIPTION'",
                "'CCSR CATEGORY 4 DESCRIPTION'",
                "'CCSR CATEGORY 5 DESCRIPTION'",
                "'CCSR CATEGORY 6 DESCRIPTION'",
            ]
        ]
        self.reference = self.reference.replace("'", "", regex=True)
        self.reference = pd.melt(
            self.reference,
            id_vars=["'ICD-10-CM CODE'"],
            value_vars=[
                "'CCSR CATEGORY 1 DESCRIPTION'",
                "'CCSR CATEGORY 2 DESCRIPTION'",
                "'CCSR CATEGORY 3 DESCRIPTION'",
                "'CCSR CATEGORY 4 DESCRIPTION'",
                "'CCSR CATEGORY 5 DESCRIPTION'",
                "'CCSR CATEGORY 6 DESCRIPTION'",
            ],
        )

    def diag_catergorize(
        self, n_component: int, diag_data: pd.DataFrame([[str]])
    ) -> pd.DataFrame([[int]]):
        """Catergorize and perform PCA diagnoses data."""
        # merge diagnoses data with ICD code reference
        diag = diag_data[diag_data["icd_version"] == "10"][["subject_id", "icd_code"]]
        diag = diag.merge(
            self.reference, left_on="icd_code", right_on="'ICD-10-CM CODE'"
        )
        diag = diag.set_index("subject_id")

        # pivot the catergorized diagnoses data
        diag["count"] = 1
        diag = diag[["value", "count"]]
        cat_diag = diag.pivot_table(
            index="subject_id", columns="value", values="count", fill_value=0
        )
        cat_diag = cat_diag.drop(cat_diag.columns[0], axis=1)

        # perform PCA
        pca = PCAClassifier(n_component)
        diag_features = pca.fit_transform(cat_diag.values)
        data = pd.DataFrame(data=diag_features, index=cat_diag.index)

        return data

    def load_data(self, data_filename: str) -> pd.DataFrame([[int]]):
        """Read in csv file of diagnoses data and reformat to dataframe."""
        diag_data = pd.read_csv(data_filename)
        data_final = self.diag_catergorize(diag_data)

        return data_final
