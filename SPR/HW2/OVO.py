import numpy as np
import pandas as pd
from math import e


# ################### Functions ################### 

def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min), min, max


def bias_addition(X):
    ones = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    return np.concatenate((ones, X), axis=1)


def SGDforLR(x, y, alpha, maxIter):
    # Stochastic gradient descent - logistic regression with static learning rate
    m, n = x.shape
    w_0 = np.random.rand(n).reshape(n, 1)
    w_1 = np.zeros((n, 1))
    it = 0
    while (it <= maxIter):
        for i in range(m):
            w_0 = w_1
            a = e**(np.inner(x[i][1:].reshape(x[i][1:].shape[0], 1).T, w_0[1:].T)+w_0[0])
            b = (1.+e**(np.inner(x[i][1:].reshape(x[i][1:].shape[0], 1).T, w_0[1:].T)+w_0[0]))
            logisticFunction = (a/b)
            error = (y[i]-logisticFunction)
            # Update weights using SGD based on the cross entropy
            w_1 = w_0 + (alpha*error[0]*x[i]).T.reshape(w_0.shape)
#            print(w_1.T)
            i += 1

        it += 1
    return w_1

#################### Main ############


# Load data
raw_data = pd.read_csv(r'iris.data', names=[
                       'F1', 'F2', 'F3', 'F4', 'Class'], skiprows=1, sep='\,', engine='python')
Data = raw_data.loc[:, ['F1', 'F2', 'F3', 'F4', 'Class']]


# Split to train and test
idx1 = Data['Class'] == 'Iris-virginica'
idx2 = Data['Class'] == 'Iris-setosa'
idx3 = Data['Class'] == 'Iris-versicolor'
class1data = Data.loc[idx1, ['F1', 'F2', 'F3', 'F4']]
class2data = Data.loc[idx2, ['F1', 'F2', 'F3', 'F4']]
class3data = Data.loc[idx3, ['F1', 'F2', 'F3', 'F4']]
class1target = np.zeros(Data.loc[idx1, ['Class']].shape[0]).reshape(
    Data.loc[idx1, ['Class']].shape[0], 1)
class2target = np.ones(Data.loc[idx2, ['Class']].shape[0]).reshape(
    Data.loc[idx2, ['Class']].shape[0], 1)
class3target = 2*np.ones(Data.loc[idx3, ['Class']].shape[0]
                         ).reshape(Data.loc[idx3, ['Class']].shape[0], 1)
trainIndex1 = np.round(0.8*class1data.shape[0])
trainIndex2 = np.round(0.8*class2data.shape[0])
trainIndex3 = np.round(0.8*class3data.shape[0])

x_trn = np.concatenate([class1data.values[0:trainIndex1.astype(
    int)], class2data[0:trainIndex2.astype(int)], class3data[0:trainIndex3.astype(int)]], axis=0)
x_tst = np.concatenate([class1data.values[trainIndex1.astype(int):class1data.shape[0]+1], class2data[trainIndex2.astype(
    int):class2data.shape[0]+1], class3data[trainIndex3.astype(int):class3data.shape[0]+1]], axis=0)
y_trn = np.concatenate([class1target[0:trainIndex1.astype(
    int)], class2target[0:trainIndex2.astype(int)], class3target[0:trainIndex3.astype(int)]], axis=0)
y_tst = np.concatenate([class1target[trainIndex1.astype(int):class1data.shape[0]+1], class2target[trainIndex2.astype(
    int):class2target.shape[0]+1], class3target[trainIndex3.astype(int):class3target.shape[0]+1]], axis=0)


# Normalize
x_trn, minX, maxX = normalize(x_trn)
x_tst = (x_tst - minX) / (maxX - minX)


# Add bias
XTrainBiasAdded = bias_addition(x_trn)

# Run SGD
nClass = 3
ovoModels = {}
it = -1
for i in range(nClass):
    for j in range(i+1, nClass):
        it += 1
        if i != j:
            ovo = y_trn[np.any(np.concatenate(
                (y_trn == i, y_trn == j), axis=1), axis=1), :]
            ovo[ovo == i] = 0
            ovo[ovo == j] = 1
            XTrainBiasAdded_ovo = XTrainBiasAdded[np.any(
                np.concatenate((y_trn == i, y_trn == j), axis=1), axis=1), :]
            gradient_desc_w = SGDforLR(
                XTrainBiasAdded_ovo, ovo, alpha=0.003, maxIter=1000)
            ovoModels[it] = gradient_desc_w

            # Show w
            print('################# Calculated Ws for classes: ', i+1, ' and ', j+1, '#################')
            print('Gradient Descent Solution, thetas: ',
                  gradient_desc_w.T, '\n', '----------')


# output Gradient_descent
it = -1
y_ovo_train = np.zeros((x_trn.shape[0], nClass))
for i in range(nClass):
    for j in range(i+1, nClass):
        it += 1
        if i != j:
            gradient_desc_w = ovoModels[it]
            for ii in range(x_trn.shape[0]):
                A = e**(np.inner(x_trn[ii].reshape(x_trn[ii].shape[0],
                        1).T, gradient_desc_w[1:].T)+gradient_desc_w[0])
                B = (1.+e**(np.inner(x_trn[ii].reshape(x_trn[ii].shape[0],
                     1).T, gradient_desc_w[1:].T)+gradient_desc_w[0]))
                Y_ovo = (A/B)[0]
                if Y_ovo > 0.5:
                    y_ovo_train[ii, j] += 1
                else:
                    y_ovo_train[ii, i] += 1


it = -1
y_ovo_test = np.zeros((x_tst.shape[0], nClass))
for i in range(nClass):
    for j in range(i+1, nClass):
        it += 1
        if i != j:
            gradient_desc_w = ovoModels[it]
            for ii in range(x_tst.shape[0]):
                A = e**(np.inner(x_tst[ii].reshape(x_tst[ii].shape[0],
                        1).T, gradient_desc_w[1:].T)+gradient_desc_w[0])
                B = (1.+e**(np.inner(x_tst[ii].reshape(x_tst[ii].shape[0],
                     1).T, gradient_desc_w[1:].T)+gradient_desc_w[0]))
                Y_ovo = (A/B)[0]
                if Y_ovo > 0.5:
                    y_ovo_test[ii, j] += 1
                else:
                    y_ovo_test[ii, i] += 1

# Find argmax
y_ovo_train_out = np.argmax(y_ovo_train, axis=1).reshape(
    np.argmax(y_ovo_train, axis=1).shape[0], 1)
y_ovo_test_out = np.argmax(y_ovo_test, axis=1).reshape(
    np.argmax(y_ovo_test, axis=1).shape[0], 1)

# Print accuracy
accuracyTrain = (np.where(y_trn == y_ovo_train_out)
                 [0].shape[0]) / (y_trn.shape[0])
print('Train accuracy: ', accuracyTrain, '\n', '----------')
accuracyTest = (np.where(y_tst == y_ovo_test_out)
                [0].shape[0]) / (y_tst.shape[0])
print('Test accuracy:  ', accuracyTest, '\n', '----------')
