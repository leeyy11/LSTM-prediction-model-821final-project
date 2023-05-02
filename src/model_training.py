# %% ---------------set hyper parameters and libraries--------------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
import seaborn as sns
from data_processing import DataProcessor


def split_set(test_ratio=0.1, data: np.ndarray):
    """Split train set and test set by test ratio."""
    train = data[:int(test_ratio*len(data))]
    test = data[int(test_ratio*len(data)):]
    return train, test


def data_load(data_x: np.ndarray, data_y: np.ndarray):
    """Load and reformat data for torch."""
    class TransData_s():
        def __init__(self, xx, yy):
            self.X = xx
            self.y = yy

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx, :], np.array(self.y[idx])

    x_train, x_test = split_set(test_ratio=0.1, data_x: np.ndarray)
    y_train, y_test = split_set(test_ratio=0.1, data_y: np.ndarray)

    train_set = TransData_s(xx=x_train, yy=y_train)
    test_set = TransData_s(xx=x_test, yy=y_test)

    train = DataLoader(train_set, batch_size=7, shuffle=True)
    test = DataLoader(test_set, batch_size=7, shuffle=True)

    return train, test


class DiseasePred(nn.Module):
    """Define a MLP model."""
    def __init__(self, hidden_size=10, num_layers=1):
        super(DiseasePred, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(input_size=10,hidden_size=10))
        layers.append(nn.ReLU())
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size=10, 1))
        layers.append(nn.Sigmoid())
        self.linear_sigmoid_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x).to(torch.float32)
        logits = self.linear_sigmoid_stack(x)
        return logits


def model_train(train: np.ndarray, model: DiseasePred, lam=0e-5, learning_rate=0.001, epochs=50):
    """Define model training."""
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        for X, y in train:
            predict = model(X)
            batch_loss = nn.functional.binary_cross_entropy(predict, y.unsqueeze(1).float())

            # L1 regularization
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            batch_loss += lam * regularization_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    return model


def performance(labels: int, pred: int):
    """Evaluating and plotting the model performance."""
    auc = roc_auc_score(labels, pred)
    ap = average_precision_score(labels, pred)

    plt.style.use('seaborn')
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    fpr, tpr, _ = roc_curve(labels, pred)
    precision, recall, _ = precision_recall_curve(labels, pred)

    ax[0].plot(fpr, tpr, label='AUC = %.3f' % auc)
    ax[0].set_xlim([-.01, 1.01]) # type: ignore
    ax[0].set_ylim([-.01, 1.01]) # type: ignore
    ax[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    ax[0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    ax[0].plot([0, 1], [0, 1], 'k--', label='No information')
    ax[0].legend(loc='lower right', fontsize=14)

    ax[1].plot(recall, precision, label='Avg Precision = %.3f' % ap)
    ax[1].set_xlim([-.01, 1.01]) # type: ignore
    ax[1].set_ylim([-.01, 1.01]) # type: ignore
    ax[1].set_xlabel('Recall (Sensitivity)', fontsize=14)
    ax[1].set_ylabel('Precision (Positive Predictive Value)', fontsize=14)
    ax[1].plot([0, 1], [labels.mean(), labels.mean()], 'k--', label='No information')   # type: ignore
    ax[1].legend(loc='upper right', fontsize=14)

    plt.show()
    return auc, ap


def run(patient_filename: str, diagnoses_filename: str):
    """Perform the MLP for disease prediction."""
    X = DataProcessor().data_load(patient_filename, diagnoses_filename, n_components=10)
    binary_output = np.random.randint(2, size=(len(X), 1))
    df_binary_output = pd.DataFrame(binary_output, columns=['Binary Output'], index=X.index)
    X = pd.concat([pd.DataFrame(X), df_binary_output], axis=1)
    xxx = np.array(X.iloc[:, :-1])
    yyy = np.array(X.iloc[:, -1])
    train,test = load_data(xxx, yyy)

    model = DiseasePred(hidden_size=10, num_layers=1)

    model = model_train(train: np.ndarray, model: DiseasePred, lam=0e-5, learning_rate=0.001, epochs=50)

    labels = []
    pred = []
    with torch.no_grad():
        for X, y in test:
            pred += model(X).tolist()
            labels += y.tolist()
    pred = sum(pred, [])

    labels1 = np.array(labels)
    pred1 = np.array(pred)
    auc, ap = performance(labels1, pred1)

    return auc,ap
