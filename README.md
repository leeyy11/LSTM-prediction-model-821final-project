# MLP-for-disease-prediction-821final-project
## Group member
Qin Weng; Yaoyao Li; Xi Liang. 

## Logistics
1. *Data Prepocessing*: Classify ICD-10 code based on existing category and convert into a binary vector.

2. *Feature extraction*: Reduce the dimension of the sparse matrix to extract useful information.

3. *Model prediction*: Input data to train the model. Learn the optimized parameters.

5. *Model evaluation & Vitualization*: Show the perfomance of the model.


## Development steps
Data Prepocessing(Classification & Code2vec):

We are aware that the ICD codes can be subdivided into numerous overarching categories corresponding to various disease types, owing to the extensive range of ICD codes available. In the data preprocessing phase, we allocate patients into the pre-existing disease categories based on their respective ICD codes. Utilizing the patients' ICD codes, we construct a binary matrix with patient IDs on the horizontal axis and major disease categories on the vertical axis, which effectively represents the patients' morbidity information.


Feature extraction:

The feature matrix in our study, derived from the ICD codes, represents a highly sparse structure with patient IDs on the horizontal axis and overarching disease categories on the vertical axis. Due to the inherent sparsity of the matrix, it is susceptible to the presence of excessive noise, which could hinder the extraction of valuable information. To address this challenge, we employ dimensionality reduction techniques on the data to mitigate the noise and enhance the discernment of meaningful patterns within the dataset.

Model prediction:

We employ a Multilayer Perceptron (MLP) to predict the potential diseases a patient may develop. The Multilayer Perceptron (MLP) is a type of artificial neural network (ANN) that can be employed for predicting potential diseases a patient may develop. The MLP consists of an input layer, one or more hidden layers, and an output layer. We input data to train the model and find the optimized parameters. 

Model evaluation & Vitualization:

We showing the perfomance of the model by AUC and AP graphs.

