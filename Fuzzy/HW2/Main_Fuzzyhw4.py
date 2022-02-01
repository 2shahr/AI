import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


#################### Functions ############

def normalize(X):
    min = X.min(axis=0)
    max = X.max(axis=0)
    return (X - min) / (max - min), min, max


def Composition_based_Classification(x_tst, R0, Rc):
    Forecast = []
    for k1 in range(x_tst.shape[0]):
        temp = (x_tst[k1, :]).reshape(1, -1)
        Dist = (np.abs(temp-R0)) / \
            (np.max((np.abs(temp-R0)), axis=0).reshape(1, -1))
        Similarity = 1-Dist
        Out = np.sum(Similarity+Rc, axis=1)
        Out = np.argmax(Out)
        Forecast.append(Out)

    return np.array(Forecast)


#################### Main ############

# Load data
Raw_data = pd.read_csv(r'hcvdat0.csv')
Temp = pd.DataFrame(Raw_data).to_numpy()


# Remove Nans
Data = Temp[~pd.isnull(Temp).any(axis=1)]


# Convert logical values to numerical
X = Data[:, 4::]
Y = Data[:, 1]
Y[Y == '0=Blood Donor'] = 0
Y[Y == '0s=suspect Blood Donor'] = 4
Y[Y == '1=Hepatitis'] = 1
Y[Y == '2=Fibrosis'] = 2
Y[Y == '3=Cirrhosis'] = 3

#X[X[:,1]=='f',1] = 1
#X[X[:,1]=='m',1] = 2


# Normalize
X_normalize, minX, maxX = normalize(X)


# Train Test split
X_trn, x_tst, y_trn, y_tst = train_test_split(
    X_normalize, Y, test_size=0.3, random_state=42)


##

for i in range(np.unique(y_trn).shape[0]):
    if i == 0:
        R0 = np.mean(X_trn[y_trn == i, :], axis=0).reshape(1, -1)
    else:
        R0 = np.concatenate(
            (R0, np.mean(X_trn[y_trn == i, :], axis=0).reshape(1, -1)), axis=0)


for i in range(np.unique(y_trn).shape[0]):
    if i == 0:
        InterClassVariance = np.var(
            X_trn[y_trn == i, :], axis=0).reshape(1, -1)
    else:
        InterClassVariance = np.concatenate((InterClassVariance, np.var(
            X_trn[y_trn == i, :], axis=0).reshape(1, -1)), axis=0)


BetweenClassVariance = np.var(R0, axis=0).reshape(1, -1)


Temp = BetweenClassVariance/InterClassVariance
Rc = Temp/((np.sum(Temp, axis=1)).reshape(-1, 1))


# Distances Calculation
Outputs = Composition_based_Classification(x_tst, R0, Rc)


# Evaluation
result = precision_recall_fscore_support(
    np.array(y_tst, dtype=float), Outputs)
result = np.mean(result, axis=1)
Fscore = result[2]
Precision = result[0]
Recall = result[1]
Accuracy = accuracy_score(np.array(y_tst, dtype=float), Outputs)
print("Fscore: ", Fscore)
print("Precision: ", Precision)
print("Recall: ", Recall)
print("Accuracy: ", Accuracy)


# confusion chart
cm = confusion_matrix(np.array(y_tst, dtype=float), Outputs)
print("confusion matrix: ")
print(cm)
