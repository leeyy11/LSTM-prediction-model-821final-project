"""Clinical data with ICD code preprocessing."""
import pandas as pd


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

    def expand_diag(self, diag: pd.DataFrame) -> pd.DataFrame:
        """Catergorize and expand the diagnoses data."""
        # merge diagnoses data with ICD code reference
        diag = diag[diag["icd_version"] == 10][["subject_id", "icd_code"]]
        diag = diag.merge(
            self.reference, left_on="icd_code", right_on="'ICD-10-CM CODE'"
        )

        # Pivot the diagnoses data
        diag["count"] = 1
        diag = diag[["subject_id", "value", "count"]]
        diag = diag.pivot_table(
            index="subject_id", columns="value", values="count", fill_value=0
        )

        return diag

    def data_load(
        self, patient_filename: str, diagnoses_filename: str
    ) -> pd.DataFrame:
        """Load and parsing clinical data."""
        # Load patient and diagnoses data
        pat = pd.read_csv(patient_filename)
        diag = pd.read_csv(diagnoses_filename)

        # Merge patient data and expanded diagnoses data
        diag = self.expand_diag(diag)
        data = pat.merge(diag, on="subject_id", how="right")

        return data
