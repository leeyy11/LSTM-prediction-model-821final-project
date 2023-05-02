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

The `DataProcessor` class is designed to process and categorize diagnoses data using ICD-10-CM codes for further analysis. The class provides three methods: `__init__`, `diag_categorize`, `diag_pca`, and `data_load`.

- __init__ Method

The `__init__` method initializes the class and reads in the ICD reference file. It then selects the necessary columns and replaces the single quotes in column names with an empty string. It then converts the data from a wide format to a long format using `pd.melt` and drops the 'variable' column. The result is stored in the `self.reference` attribute.

- diag_categorize Method

The `diag_categorize` method categorizes diagnoses data by merging the diagnoses data with the ICD code reference. It then pivots the data and fills any missing values with 0. The result is returned as a pandas DataFrame.

- diag_pca Method

The `diag_pca` method performs PCA on the data using the `PCAClassifier` class from the `PCA` module. It then returns the transformed data as a pandas DataFrame.

- data_load Method

The `data_load` method reads in a CSV file of diagnoses data and calls the `diag_categorize` and `diag_pca` methods to process the data. It then returns the transformed data and the diagnoses data for the specified disease as pandas DataFrames.

Note that the `diag_data` argument passed to the `diag_categorize` method must have an 'icd_version' column with the value '10' for the method to work properly.


#### Example

```python
from data_processor import DataProcessor
import pandas as pd

processor = DataProcessor()
disease_name = "Intestinal infection"
diagnoses_file = "diagnoses_data.csv"
n_component = 10

data = processor.data_load(disease_name, diagnoses_file, n_component)
```

In the example above, an instance of the `DataProcessor` class is created, and the `data_load` method is used to load and parse clinical data from input `diagnoses_data.csv`, and the specific disease name with the dimension the user wants. The resulting output is stored in the `data` variable.

### DiseasePred Class

The `DiseasePred` class is designed for disease prediction using a multilayer perceptron (MLP) model. The MLP model is implemented using PyTorch, and performance is evaluated using the ROC AUC score and average precision score. It contains the following methods:

- `__init__(self, disease_name:str, diagnoses_filename: str, n_components=10)`:
    Initializes a `DiseasePred` object. It loads the data and creates an MLP model for the specified disease. The data is loaded using `DataProcessor().data_load()` method from `data_processing` module. `disease_name` is a string specifying the disease to be predicted, `diagnoses_filename` is the file path of diagnoses file, and `n_components` is an integer specifying the number of principal components to be used for dimensionality reduction.

- `learning(self, lam=0e-5, learning_rate=0.001, epochs=50)`:
    Trains the MLP model using the loaded data. `lam`, `learning_rate`, and `epochs` are optional parameters. `lam` specifies the L1 regularization penalty weight. `learning_rate` specifies the learning rate for the optimizer, and `epochs` specifies the number of training epochs.

- `pred(self, data_x: np.ndarray)`:
    Predicts the probability of the disease based on the provided data. `data_x` is a numpy array containing the patient data for prediction. The user could also call this function under the class to make prediction on the data they want to use.

- `performance(self, new_diag: np.ndarray, new_disease: np.ndarray)`:
    Evaluates the performance of the MLP model on new data. `new_diag` is a numpy array containing the patient data for prediction, and `new_disease` is a numpy array containing the corresponding disease labels.

#### Example

Here is an example of using the `DiseasePred` class:

```python
from DiseasePred import DiseasePred

# Initialize the DiseasePred object for predicting disease 'diabetes'
dp = DiseasePred('Intestinal infection', 'diagnoses.csv', n_components=10)

# Train the model
dp.learning(lam=0.001, learning_rate=0.01, epochs=50)

# Predict the disease probability for a new patient
new_data = np.random.rand(1, 10)
pred_prob = dp.pred(new_data)

# Evaluate the model performance on new data
new_diag = np.random.rand(100, 10)
new_disease = np.random.randint(2, size=100)
dp.performance(new_diag, new_disease)
```
