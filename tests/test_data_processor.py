"""Testing data processor."""
import pytest
from fake_files import fake_files
from data_processing import DataProcessor
import pandas as pd


def test_ICD_reference() -> None:
    """Test ICD reference file."""
    ref = DataProcessor().reference
    assert ref.loc[0, "'ICD-10-CM CODE'"] == "A000"
    assert ref.loc[0, "value"] == "Intestinal infection"


def test_diag_categorize() -> None:
    """Test categorizing diagnoses data."""
    diagnoses = [
        ["subject_id", "icd_version", "icd_code"],
        ["123", "10", "A000"],
        ["123", "10", "A001"],
    ]
    diag_data = pd.DataFrame(diagnoses[1:], columns=diagnoses[0])
    cat_diag = DataProcessor().diag_categorize(diag_data)
    assert cat_diag.loc["123", "Intestinal infection"] == 1


def test_diag_pca() -> None:
    """Test pac on data."""
    cat = [
        ["subject_id", "cat1", "cat2"],
        ["123", 1, 0],
        ["456", 1, 1],
    ]
    cat_diag = pd.DataFrame(cat[1:], columns=cat[0]).set_index("subject_id")
    pca_diag = DataProcessor().diag_pca(1, cat_diag)
    assert round(pca_diag.loc["123"][0], 2) == 0.85


def test_data_load() -> None:
    """Test loading and parsing clinical data."""
    diagnoses_file = [
        ["subject_id", "icd_version", "icd_code"],
        ["123", "10", "A000"],
        ["456", "10", "A001"],
    ]
    with fake_files(diagnoses_file) as data_file:
        data = DataProcessor().data_load(1, data_file[0])
        assert round(data.loc["123"][0], 2) == 1.41

