# MLP-for-disease-prediction-821final-project

## Group member

Qin Weng; Yaoyao Li; Xi Liang.

# BioMLP Library

This library implements a Multilayer Perceptron (MLP) model for disease prediction using patient ICD-10 codes. The library consists of four main modules: data preprocessing, feature extraction, model prediction, and model evaluation/visualization.

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

- `preprocess_data`: classifies ICD-10 codes based on existing categories and converts them into a binary vector.
- `extract_features`: reduces the dimensionality of the sparse matrix to extract useful information.
- `train_model`: inputs data to train the MLP model and learn the optimized parameters.
- `evaluate_model`: shows the performance of the model using AUC and AP graphs.

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

1. Install the required dependencies by running the following command:

```
pip install numpy
pip install pandas
pip install sklearn
pip install torch
pip install matplotlib
pip install seaborn
```

## API Reference

### DataProcessor Class

This is a class

