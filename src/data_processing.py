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

    def diag_categorize(
        self,
        diag_data: pd.DataFrame[[[str]]],
    ) -> pd.DataFrame[[[int]]]:
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
        # cat_diag = cat_diag.drop(cat_diag.columns[0], axis=1)

        return cat_diag

    def pca_extract(
        self,
        data: pd.DataFrame[[[int]]],
        n_components: int,
    ) -> pd.DataFrame[[[int]]]:
        """Perform PCA on data."""
        pca = PCAClassifier(n_components)
        diag_features = pca.fit_transform(data.values)
        pca_data = pd.DataFrame(
            data=diag_features,
            index=data.index,
        )

        return pca_data

    def data_load(
        self,
        data_filename: str,
        n_components: int,
    ) -> pd.DataFrame[[[int]]]:
        """Read in csv file of diagnoses data and reformat to dataframe."""
        diag_data = pd.read_csv(data_filename).astype(str)
        cat_diag = self.diag_categorize(diag_data)
        data_final = self.pca_extract(
            data,
            n_components,
        )

        return data_final
