# MLP-for-disease-prediction-821final-project
Group member: Qin Weng; Yaoyao Li; Xi Liang. 

Data Prepocessing: Classify ICD-10 code based on existing category. Convert classified ICD-10 vector into a binary vector of description on these codes.

Feature extraction: Reduce the dimension of the sparse matrix to extract useful information.

Model prediction: Input data to train the model and find the optimized parameters. Also showing the perfomance of the model by graphs.



Data Prepocessing(Classification & Code2vec):

We are aware that the ICD codes can be subdivided into numerous overarching categories corresponding to various disease types, owing to the extensive range of ICD codes available. In the data preprocessing phase, we allocate patients into the pre-existing disease categories based on their respective ICD codes. Utilizing the patients' ICD codes, we construct a binary matrix with patient IDs on the horizontal axis and major disease categories on the vertical axis, which effectively represents the patients' morbidity information. This matrix will be incorporated into our research paper.


Feature extraction:

The feature matrix in our study, derived from the ICD codes, represents a highly sparse structure with patient IDs on the horizontal axis and overarching disease categories on the vertical axis. Due to the inherent sparsity of the matrix, it is susceptible to the presence of excessive noise, which could hinder the extraction of valuable information. To address this challenge, we employ dimensionality reduction techniques on the data to mitigate the noise and enhance the discernment of meaningful patterns within the dataset. This refined approach will be presented in our research paper.


Model prediction:

We employ a Multilayer Perceptron (MLP) to predict the potential diseases a patient may develop. The Multilayer Perceptron (MLP) is a type of artificial neural network (ANN) that can be employed for predicting potential diseases a patient may develop. The MLP consists of an input layer, one or more hidden layers, and an output layer. We input data to train the model and find the optimized parameters. 

Model evaluation:

We showing the perfomance of the model by graphs.

