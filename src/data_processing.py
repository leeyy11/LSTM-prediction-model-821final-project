"""Clinical data with ICD code preprocessing."""
import pandas as pd
from PCA import (
    PCAClassifier,
)
import numpy as np


class DataProcessor:
    """Define a dataprocessor."""

    def __init__(
        self,
        diag_filename="DXCCSR.csv",
    ):
        """Define the ICD reference file."""
        self.reference = pd.read_csv(diag_filename)
        refer = self.reference
        refer = refer[
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
        refer = refer.replace(
            "'",
            "",
            regex=True,
        )
        refer = pd.melt(
            refer,
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
        refer = refer[refer["value"] != " "]
        refer = refer.drop(columns=["variable"])
        self.reference = refer

    def diag_categorize(self, diag_data):
        """Catergorize diagnoses data."""
        # merge diagnoses data with ICD code reference
        diag = diag_data[diag_data["icd_version"] == "10"][
            [
                "subject_id",
                "icd_code",
            ]
        ]
        diag = diag.merge(
            self.reference,
            left_on="icd_code",
            right_on="'ICD-10-CM CODE'",
        )

        # pivot the catergorized diagnoses data
        diag["count"] = 1
        diag = diag[
            [
                "subject_id",
                "value",
                "count",
            ]
        ]
        cat_diag = diag.pivot_table(
            index="subject_id",
            columns="value",
            values="count",
            fill_value=0,
        )

        return cat_diag

    def diag_pca(
        self,
        n_components: int,
        cat_diag,
    ):
        """Perform PCA on data."""
        pca = PCAClassifier(n_components)
        diag_features = pca.fit_transform(cat_diag.values)
        pca_diag = pd.DataFrame(
            data=diag_features,
            index=cat_diag.index,
        )

        return pca_diag

    def data_load(
        self,
        disease_name: str,
        data_filename: str,
        n_components: int,
    ):
        """Read in csv file of diagnoses data and reformat to dataframe."""
        diag_data = pd.read_csv(data_filename).astype(str)
        cat_diag = self.diag_categorize(diag_data)
        disease = cat_diag[disease_name]
        pac_diag = self.diag_pca(
            n_components, cat_diag.drop(disease_name, axis=1)
        )

        return pac_diag, disease
