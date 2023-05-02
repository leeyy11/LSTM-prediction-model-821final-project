# %% ---------------set hyper parameters and libraries--------------
# library
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from feature_extraction import get_feature
import seaborn as sns

# data generation
N_PATIENTS = 50000
N_FEATURES = 100
RANDOM_SEED = 2023
STEP_SIZE = 1e-3

# modeling
training_ratio = 0.8
test_ratio = 0.1

learning_rate = 0.001
batch_size = 7

input_size=10
num_layers = 1
hidden_size = 10
epochs = 50

lam = 0e-5

# %% ---------------single-label model--------------
# %% def data loader

def split_set(test_ratio, data):
    train = data[:int(test_ratio*len(data))]
    test = data[int(test_ratio*len(data)):]
    return train, test

# %% def data loader
def load_data(data_x, data_y):
    class TransData_s():
        def __init__(self, xx, yy):
            self.X = xx
            self.y = yy

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx, :], np.array(self.y[idx])

    x_train, x_test = split_set(test_ratio, data_x)
    y_train, y_test = split_set(test_ratio, data_y)

    train_set = TransData_s(xx=x_train, yy=y_train)
    test_set = TransData_s(xx=x_test, yy=y_test)

    train = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # type: ignore
    test = DataLoader(test_set, batch_size=batch_size, shuffle=True)  # type: ignore

    return train, test


# %% def single label model
class Model_s(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(Model_s, self).__init__()
        self.flatten = nn.Flatten()
        layers = []

        # input_tensor = input_tensor.to(linear_layer.weight.dtype)
        layers.append(nn.Linear(input_size,hidden_size))
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

    # %% def model training


def model_training_s(train, model, lam, learning_rate, epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # train_loss_list = []
    # valid_loss_list = []

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        # train_loss = 0
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

        #     train_loss += batch_loss.item()
        # train_loss /= len(train)
        # train_loss_list.append(train_loss)

        # valid_loss = 0
        # with torch.no_grad():
        #     for X, y in valid:
        #         X, y = X.to(device), y.to(device)
        #         loss = nn.functional.binary_cross_entropy(model(X), y.unsqueeze(1))
        #         valid_loss += loss.item()
        # valid_loss /= len(valid)
        # valid_loss_list.append(valid_loss)

    return model  # , train_loss_list, valid_loss_list


# # %% def single modeling
# def single_modeling(train, test):
#     model = Model_s(hidden_size, num_layers)
#
#     model = model_training_s(train, test, model, lam, learning_rate, epochs)
#
#     labels = []
#     pred = []
#     with torch.no_grad():
#         torch.manual_seed(2023)
#         for X, Y in test:
#             pred += model(X).tolist()
#             labels += Y[:, 0].tolist()
#     pred = sum(pred, [])
#
#     auc, ap = performance(np.array([labels[i] for i in range(len(labels))]), np.array([pred[i] for i in range(len(pred))]))
#
#     return auc, ap

# %% def performance plot
def performance(labels, pred):
    # AUROC + PR +loss
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

    # return auc, ap



# # %% def single modeling
def single_modeling(xx, yy):
    train,test = load_data(xx, yy)
    input_size=xx.shape[-1]


    model = Model_s(hidden_size, num_layers)

    model = model_training_s(train, model, lam, learning_rate, epochs)

    labels = []
    pred = []
    with torch.no_grad():
        for X, y in test:
            pred += model(X).tolist()
            labels += y.tolist()
    pred = sum(pred, [])

    # performance(np.array([labels[i] for i in range(len(labels))]),
    #                     np.array([pred[i] for i in range(len(pred))]))

    labels1 = np.array(labels)
    pred1 = np.array(pred)
    auc, ap = performance(labels1, pred1)

    return auc,ap


# # %% def single modeling
# def single_modeling(train, test):
#     model = Model_s(hidden_size, num_layers)
#
#     model = model_training_s(train, test, model, lam, learning_rate, epochs)
#
#     labels = []
#     pred = []
#     with torch.no_grad():
#         torch.manual_seed(2023)
#         for X, Y in test:
#             pred += model(X).tolist()
#             labels += Y[:, 0].tolist()
#     pred = sum(pred, [])
#
#     auc, ap = performance(np.array([labels[i] for i in range(len(labels))]), np.array([pred[i] for i in range(len(pred))]))
#
#     return auc,ap





# %% ---------------single/multi simulation--------------
X=get_feature()

# Simulate binary output
binary_output = np.random.randint(2, size=(len(X), 1))

# Create a Pandas DataFrame from the binary output
df_binary_output = pd.DataFrame(binary_output, columns=['Binary Output'],index=X.index)

# Concatenate the feature data and the binary output horizontally
# X['binary_output'] = pd.DataFrame(binary_output)

X = pd.concat([pd.DataFrame(X), df_binary_output], axis=1)

# auc_single = single_modeling(np.array(X.iloc[:,:-1]), np.array(X.iloc[:, -1]))
xxx=np.array(X.iloc[:,:-1])
yyy=np.array(X.iloc[:, -1])
auc, ap=single_modeling(xxx, yyy)

# single = pd.DataFrame(columns=["ER", "similarity", "AUC"])
#
# for event_rate in np.logspace(-3, -1, 3):
#     for similarity in np.linspace(0.01, 1, 3):
#         # x, e1, e2 = generate_data(event_rate, similarity)
#
#         np.array(X.iloc[:,:-1]), np.array(X.iloc[:, -1])
#
#         auc_single = single_modeling(np.array(X.iloc[:,:-1]), np.array(X.iloc[:, -1]))
#
#         single.loc[len(single)] = {"ER": event_rate,
#                                    "similarity": similarity,
#                                    "AUC": auc_single}
#
#
#         print("ER:", event_rate, "similarity:", similarity)
#
# single.explode("AUC").to_csv("single.csv")
# # %%
