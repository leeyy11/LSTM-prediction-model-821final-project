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


    def diag_catergorize(self, diag_data: pd.DataFrame) -> pd.DataFrame:
        """Catergorize and expand the diagnoses data."""
        # merge diagnoses data with ICD code reference
        diag = diag_data[diag_data["icd_version"] == 10][["subject_id", "icd_code"]]
        diag = diag.merge(
            self.reference, left_on="icd_code", right_on="'ICD-10-CM CODE'"
        )

        # pivot the catergorized diagnoses data
        diag["count"] = 1
        diag = diag[["subject_id", "value", "count"]]
        cat_diag = diag.pivot_table(
            index="subject_id", columns = "value", values="count", fill_value=0
        )

        return cat_diag


    def data_load(
        self, patient_filename: str, diagnoses_filename: str
    ) -> pd.DataFrame:
        """Load and parsing clinical data."""
        # Load patient and diagnoses data
        pat = pd.read_csv(patient_filename)
        diag = pd.read_csv(diagnoses_filename)

        # perform PCA on catergrized diagnoses data, and merged with patient data
        cat_diag = self.diag_catergorize(diag)
        pca = PCAClassifier(n_components=10)
        diag_features = pca.fit_transform(cat_diag.values)
        diag = pd.DataFrame(data=diag_features, index=cat_diag.index)

        data = pat.merge(diag, on="subject_id", how="right")

        return data
