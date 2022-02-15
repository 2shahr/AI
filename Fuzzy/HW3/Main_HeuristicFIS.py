import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


##################### Functions #################

def CheckRules(Input, Means):
    # 1---> IF Input==MeanClass0 THEN class==0
    # 2---> IF Input==MeanClass1 THEN class==1
    # 3---> IF Input==MeanClass2 THEN class==2
    # 4---> IF Input==MeanClass3 THEN class==3
    # 5---> IF Input==MeanClass4 THEN class==4
    Dist = []
    for i in range(0, Means.shape[0]):
        Dist.append(np.linalg.norm(currentdata-Means[i, :]))

    Dist = np.array(Dist)/np.max(Dist)
    Composition = 1-Dist

    return Composition


def Tuneweights(Results, y_trn):
    Accuracies = []
    Weights = []
    for i1 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        for i2 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            for i3 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                for i4 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    for i5 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                        Weights.append([i1, i2, i3, i4, i5])
                        temp = Results * np.array([i1, i2, i3, i4, i5])
                        Out = np.argmax(temp, axis=1)
                        accuracy = (np.where(y_trn == Out)[
                                    0].shape[0]) / (y_trn.shape[0])
                        Accuracies.append(accuracy)

    Best = np.argmax(Accuracies)
    OptimizedWeights = Weights[Best]

    return np.array(OptimizedWeights)


##################### Main ####################
# Read data  from .csv
Initial = pd.read_csv(r'hcvdat0.csv')
data = pd.DataFrame(Initial).to_numpy()


# Mising value handling and handling logical features
Data = data[~pd.isnull(data).any(axis=1)]
Inputss = Data[:, 4::]
Targets = Data[:, 1]
Targets[Targets == '0=Blood Donor'] = 0
Targets[Targets == '0s=suspect Blood Donor'] = 4
Targets[Targets == '1=Hepatitis'] = 1
Targets[Targets == '2=Fibrosis'] = 2
Targets[Targets == '3=Cirrhosis'] = 3


# spliting train and test data
x_trn, x_tst, y_trn, y_tst = train_test_split(
    Inputss, Targets, test_size=0.3, random_state=42)


# calculating mean and standard deviation of each class
for i in range(np.unique(y_trn).shape[0]):
    if i == 0:
        Means = np.mean(x_trn[y_trn == i, :], axis=0).reshape(1, -1)
        Mins = np.min(x_trn[y_trn == i, :], axis=0).reshape(1, -1)
        Maxs = np.max(x_trn[y_trn == i, :], axis=0).reshape(1, -1)
    else:
        Means = np.concatenate((Means, np.mean(
            np.array(x_trn[y_trn == i, :], dtype=float), axis=0).reshape(1, -1)), axis=0)
        Mins = np.concatenate((Mins, np.min(
            np.array(x_trn[y_trn == i, :], dtype=float), axis=0).reshape(1, -1)), axis=0)
        Maxs = np.concatenate((Maxs, np.max(
            np.array(x_trn[y_trn == i, :], dtype=float), axis=0).reshape(1, -1)), axis=0)


# Show differences among classes
plt.figure(figsize=[8, 6])

for i in range(np.unique(y_trn).shape[0]):
    if i == 0:
        plt.plot(Means[i, :], '.-r', markersize=12)
    elif i == 1:
        plt.plot(Means[i, :], '.-b', markersize=12)
    elif i == 2:
        plt.plot(Means[i, :], '.-k', markersize=12)
    elif i == 3:
        plt.plot(Means[i, :], '.-g', markersize=12)
    elif i == 4:
        plt.plot(Means[i, :], '.-m', markersize=12)

plt.xlabel('Feature index', fontsize=16)
plt.ylabel('Feature value', fontsize=16)
plt.title('Difference between mean of classes')
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'])
plt.show()


# # After checking the figure, we selected Features 1,3 and 8 as discriminative features (ALP, AST, GGT)
print("#########################################################")
print('####################### Mean of feature 1 ########################')
print(Means[:, 1])
print("#########################################################")
print('####################### Mean of feature 3 ########################')
print(Means[:, 3])
print("#########################################################")
print('####################### Mean of feature 8 ########################')
print(Means[:, 8])


# With regard to dataset and mean of features in each class:
# We define our rules based on composition between value of features in new input and mean of features in each class.
# Then, we define 5 rules as:
# 1---> IF Input==MeanClass0 THEN class==0
# 2---> IF Input==MeanClass1 THEN class==1
# 3---> IF Input==MeanClass2 THEN class==2
# 4---> IF Input==MeanClass3 THEN class==3
# 5---> IF Input==MeanClass4 THEN class==4

# Determine weight for each rule in regard to training data's label
# Check rules
Results = []
for k1 in range(x_trn.shape[0]):
    currentdata = (x_trn[k1, [1, 3, 8]]).reshape(1, -1)
    Composition = CheckRules(currentdata, Means[:, [1, 3, 8]])
    Results.append(Composition)

Results = np.array(Results)

# Tune weights
OptimizedWeights = Tuneweights(Results, y_trn)


# Check rules
Results = []
for k1 in range(x_tst.shape[0]):
    currentdata = (x_tst[k1, [1, 3, 8]]).reshape(1, -1)
    Composition = CheckRules(currentdata, Means[:, [1, 3, 8]])
    temp = Composition*OptimizedWeights
    Out = np.argmax(temp)
    Results.append(Out)

Results = np.array(Results)


# accuracy
print("#########################################################")
print("######################## Accuracy #######################")
Accuracy = accuracy_score(np.array(y_tst, dtype=float), Results)
print("Accuracy: ", Accuracy)


# confusion chart
print("#########################################################")
print("############## Confusion matrix #########################")
cm = confusion_matrix(np.array(y_tst, dtype=float), Results)
print(cm)
