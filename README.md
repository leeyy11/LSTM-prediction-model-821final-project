# MLP-for-disease-prediction-821final-project

## Group member

Qin Weng; Yaoyao Li; Xi Liang.

# BioMLP Library

This library implements a Multilayer Perceptron (MLP) model for disease prediction using patient ICD-10 codes and it consists of four main modules: data preprocessing, feature extraction, model prediction, and model evaluation/visualization.

## Data Preprocessing

In the data preprocessing phase, patients are allocated into pre-existing disease categories based on their respective ICD-10 codes. A binary matrix is constructed with patient IDs on the horizontal axis and major disease categories on the vertical axis, effectively representing the patients' morbidity information.

## Feature Extraction

The feature matrix derived from the ICD codes represents a highly sparse structure with patient IDs on the horizontal axis and overarching disease categories on the vertical axis. To address the inherent sparsity of the matrix and mitigate noise, dimensionality reduction techniques are employed to enhance the discernment of meaningful patterns within the dataset.

## Model Prediction

A Multilayer Perceptron (MLP) model is employed to predict potential diseases a patient may develop. The MLP consists of an input layer, one or more hidden layers, and an output layer. Data is input to train the model and find the optimized parameters.

## Model Evaluation & Visualization

The performance of the MLP model is evaluated using AUC and AP graphs, which are generated to visualize the model's accuracy.

## Usage

To use this library, import the required modules and call the corresponding functions:

- Check the input data file follows the formats below:
  
  * The data file is of ".csv" format
  
  * The data file should contain the first three columns as "subject_id", "icd_version" and "icd_code". The data type of each one should be string.
  
  The example data file is shown below:
  
  ```python

  diagnoses = [
        ["subject_id", "icd_version", "icd_code"],
        ["001", "10", "A000"],
        ["001", "10", "A001"],
    ]
    ```

- `DataProcessor`: classifies ICD-10 codes based on existing categories and reduces the dimensionality of the sparse matrix to extract useful information.
- `DiseasePred`:  Converts processed data into a binary vector and by inputing data to train the MLP model and learn the optimized parameters, it also shows the performance of the model using AUC and AP graphs.

## Dependencies

This library requires the following Python packages:

- numpy
- pandas
- sklearn
- torch
- matplotlib
- seaborn

## Installation

To install this library, follow these steps:

1. Clone the repository to your local machine

2. Install the required dependencies by running the following command:

```
pip install numpy pandas sklearn torch matplotlib seaborn
```

## API Reference

### DataProcessor Class

The `DataProcessor` class provides methods to categorize, expand, load and parse diagnoses data. 

To use this class, create an instance of the class with the `diag_filename` parameter (default value is "DXCCSR.csv"). Then, you can call the following methods:

#### diag_categorize(diag_data: pd.DataFrame) -> pd.DataFrame

Categorize and expand the diagnoses data. This method takes a Pandas DataFrame as input and returns a Pandas DataFrame as output.

#### pca_extract(n_components: int, data: pd.DataFrame) -> pd.DataFrame

Perform PCA on data to extract to n_components features. This method takes an interger as the number of features, and a Pandas DataFrame as data input. This method returns a Pandas DataFrame as output.

#### data_load(n_components: int, patient_filename: str, diagnoses_filename: str) -> pd.DataFrame

Load and parsing clinical data. This method takes three parameters, an interger as the number of features for PCA, `patient_filename` and `diagnoses_filename`, which are the names of the patient and diagnoses files to be loaded, respectively. It returns a Pandas DataFrame as output.

#### Example

```python
from data_processor import DataProcessor
import pandas as pd

processor = DataProcessor()
patient_file = "patient_data.csv"
diagnoses_file = "diagnoses_data.csv"

data = processor.data_load(patient_file, diagnoses_file)
```

In the example above, an instance of the `DataProcessor` class is created, and the `data_load` method is used to load and parse clinical data from two input files: `patient_data.csv` and `diagnoses_data.csv`. The resulting output is stored in the `data` variable.

### DiseasePred Class

The `DiseasePred` class initializes the model. The `run` function trains the model and returns the trained model. You can specify the following hyperparameters as arguments to the `DiseasePred` class:

* `hidden_size`: The number of units in the hidden layer (default=10).
* `num_layers`: The number of layers in the model (default=1).

You can also specify the following hyperparameters as arguments to the `run` function:

* `learning_rate`: The learning rate for the optimizer (default=0.001).
* `epochs`: The number of epochs to train the model (default=50).
* `batch_size`: The batch size for the data loader (default=7).
* `lam`: The regularization parameter for L1 regularization (default=0e-5).

The trained model returns the predicted probabilities for the test set. The `performance` function plots the ROC curve and the Precision-Recall curve, and returns the Area Under the Curve (AUC) for the ROC curve and the average precision (AP) for the Precision-Recall curve. You can use it as follows:

```python
from disease_prediction import performance

performance(labels, pred)
```

where `labels` are the true labels of the test set, and `pred` are the predicted probabilities for the test set.
