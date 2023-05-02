import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
import seaborn as sns
from data_processing import DataProcessor


class MLP(nn.Module):
    """Define a MLP model."""
    def __init__(self, input_size, hidden_size=10, num_layers=1):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.linear_sigmoid_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x).to(torch.float32)
        logits = self.linear_sigmoid_stack(x)
        return logits


class DiseasePred():
    """Define a disease prediction library"""

    def __init__(self, disease_name:str, diagnoses_filename: str, n_components=10):
        self.disease_name = disease_name
        diag_df, disease_df = DataProcessor().data_load(disease_name, diagnoses_filename, n_components)
        self.xx = np.array(diag_df)
        self.yy = np.array(disease_df).transpose()
        self.n = n_components
        self.model = MLP(input_size=(self.n))

    def split_set(self, data: np.ndarray, test_ratio):
        """Split train set and test set by test ratio."""
        train = data[:int(test_ratio*len(data))]
        test = data[int(test_ratio*len(data)):]
        return train, test

    def torch_load(self, data_x: np.ndarray, data_y: np.ndarray, test_ratio, batch_size=7):
        """Load and reformat data for torch."""
        class TransData_s():
            def __init__(self, xx, yy):
                self.X = xx
                self.y = yy

            def __len__(self):
                return self.X.shape[0]

            def __getitem__(self, idx):
                return self.X[idx, :], np.array(self.y[idx])

        x_train, x_test = self.split_set(data_x, test_ratio)
        y_train, y_test = self.split_set(data_y, test_ratio)

        train_set = TransData_s(xx=x_train, yy=y_train)
        test_set = TransData_s(xx=x_test, yy=y_test)

        train = DataLoader(train_set, batch_size, shuffle=True)
        test = DataLoader(test_set, batch_size, shuffle=True)

        return train, test
    

    def learning(self, lam=0e-5, learning_rate=0.001, epochs=50):
        """Define and train a MLP model."""

        train, test = self.torch_load(self.xx, self.yy, test_ratio=0.8)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        for t in range(epochs):
            for X, y in train:
                predict = self.model(X)
                batch_loss = nn.functional.binary_cross_entropy(predict, y.unsqueeze(1).float())

                # L1 regularization
                regularization_loss = 0
                for param in self.model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
                batch_loss += lam * regularization_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
    
    def pred(self, data_x: np.ndarray):
        """Return the predcted probability of disease."""
        x_tensor = torch.from_numpy(data_x)
        pred = self.model(x_tensor).detach().numpy()
        return(pred)
    
    def performance(self, new_diag: np.ndarray, new_disease: np.ndarray):
        """Evaluating and plotting the model performance on new data."""
        labels = new_disease
        pred = self.pred(new_diag)
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
    
