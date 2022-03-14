import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

#################### Functions ############


def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min)


#################### Main ############


# Load data
Raw_data = pd.read_excel(
    'Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls', sheet_name='Training_Data')
x_trn = Raw_data.values[:, [3, 4]]
y_trn = Raw_data.values[:, -1]
y_trn[y_trn == 'High'] = 0
y_trn[y_trn == 'Middle'] = 1
y_trn[y_trn == 'Low'] = 2
y_trn[y_trn == 'very_low'] = 3

Raw_data = pd.read_excel(
    'Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls', sheet_name='Test_Data')
x_tst = Raw_data.values[:, [3, 4]]
y_tst = Raw_data.values[:, -1]
y_tst[y_tst == 'High'] = 0
y_tst[y_tst == 'Middle'] = 1
y_tst[y_tst == 'Low'] = 2
y_tst[y_tst == 'Very Low'] = 3

# Show data
plt.figure(figsize=[8, 6])
plt.plot(x_trn[y_trn == 0, 0], x_trn[y_trn == 0, 1], '.b')
plt.plot(x_trn[y_trn == 1, 0], x_trn[y_trn == 1, 1], '.r')
plt.plot(x_trn[y_trn == 2, 0], x_trn[y_trn == 2, 1], '.g')
plt.plot(x_trn[y_trn == 3, 0], x_trn[y_trn == 3, 1], '.k')
plt.title('Original data', fontsize=16)
plt.show()


# GMM
Accuracies = []
for kfold in range(0, 5):
    Accuracy = []
    for k in [1, 5, 10]:
        # Train
        Y0 = np.zeros(y_trn.shape)
        Y0[y_trn != 0] = 1
        gmm0 = GaussianMixture(
            n_components=k, random_state=0).fit(x_trn[Y0 == 0])
        Y1 = np.zeros(y_trn.shape)
        Y1[y_trn != 1] = 1
        gmm1 = GaussianMixture(
            n_components=k, random_state=0).fit(x_trn[Y1 == 0])
        Y2 = np.zeros(y_trn.shape)
        Y2[y_trn != 2] = 1
        gmm2 = GaussianMixture(
            n_components=k, random_state=0).fit(x_trn[Y2 == 0])
        Y3 = np.zeros(y_trn.shape)
        Y3[y_trn != 3] = 1
        gmm3 = GaussianMixture(
            n_components=k, random_state=0).fit(x_trn[Y3 == 0])
        # Train
        likelihood0 = gmm0.score_samples(x_tst)
        likelihood1 = gmm1.score_samples(x_tst)
        likelihood2 = gmm2.score_samples(x_tst)
        likelihood3 = gmm3.score_samples(x_tst)
        likelihoods = np.concatenate((likelihood0.reshape(-1, 1), likelihood1.reshape(-1, 1),
                                     likelihood2.reshape(-1, 1), likelihood3.reshape(-1, 1)), axis=1)
        Outputs = np.argmax(likelihoods, axis=1)
        accuracy = accuracy_score(np.array(y_tst, dtype=float), Outputs)
        Accuracy.append(accuracy)
    Accuracies.append(Accuracy)


allK = [1, 5, 10]
Best = np.argmax(np.mean(np.array(Accuracies), axis=0))
Acc = np.max(np.mean(np.array(Accuracies), axis=0))
BestK = allK[Best]
print(f"Best K: {BestK}, with accuracy= {Acc} ")

#
#
