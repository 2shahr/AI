import numpy as np
import pandas as pd
from math import e


#################### Functions ############

def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min), min, max


def bias_addition(X):
    ones = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    return np.concatenate((ones, X), axis=1)


def logicFunction(y, classidx):
    out = np.zeros(y.shape)
    out[y == classidx] = 1
    return out


def SGDforSoftmaxLR(x, y, alpha, maxIter, nClass):
    # Stochastic gradient descent for softmax logistic regression with static learning rate
    m, n = x.shape
    w_0 = np.random.rand(n, nClass)
    w_1 = np.zeros((n, nClass))
    for classidx in range(nClass):
        it = 0
        while (it <= maxIter):
            for i in range(m):
                temp1 = w_1
                w_0 = temp1
                A = e**(np.inner(x[i][1:].reshape(x[i][1:].shape[0], 1).T, w_0.T[classidx].reshape(
                    w_0.T[classidx].shape[0], 1)[1:].T)+w_0.T[classidx].reshape(w_0.T[classidx].shape[0], 1)[0])

                B = 0
                for ii in range(nClass):
                    temp = B = (e**(np.inner(x[i][1:].reshape(x[i][1:].shape[0], 1).T, w_0.T[ii].reshape(
                        w_0.T[ii].shape[0], 1)[1:].T)+w_0.T[ii].reshape(w_0.T[ii].shape[0], 1)[0]))
                    B += temp

                probability = (A/B)[0]

                temp = w_0[:, classidx].reshape(w_0.shape[0], 1) - alpha*(-1 * ((logicFunction(y, classidx)[
                    i]-probability)*x[i])).reshape(w_0.shape[0], 1)  # .T.reshape(w_0.shape)   # Update weights using SGD
                for qq in range(n):
                    w_1[qq][classidx] = temp[qq][0]

            it += 1
    return w_1


# ################### Main ############
# Load data
raw_data = pd.read_csv(r'iris.data', names=['F1', 'F2', 'F3', 'F4', 'Class'], skiprows=1, sep='\,', engine='python')
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
gradient_desc_w_Mat = SGDforSoftmaxLR(
    XTrainBiasAdded, y_trn, alpha=0.008, maxIter=1000, nClass=3)


# Show w
print('################# Calculated Ws######################')
print('Gradient Descent Solution, thetas: ', gradient_desc_w_Mat, '\n', '----------')

# output Gradient_descent
nClass = 3
for it in range(nClass):
    y_softmax = []
    Gradient_descent_w = gradient_desc_w_Mat[:, it]
    for i in range(x_trn.shape[0]):
        A = e**(np.inner(x_trn[i].reshape(x_trn[i].shape[0], 1).T,
                Gradient_descent_w[1:].T)+Gradient_descent_w[0])
        B = 0
        for ii in range(nClass):
            Gradient_descent_w_others = gradient_desc_w_Mat[:, ii]
            temp = e**(np.inner(x_trn[i].reshape(x_trn[i].shape[0], 1).T,
                       Gradient_descent_w_others[1:].T)+Gradient_descent_w_others[0])
            B += temp
        y_softmax.append((A/B)[0])
    if it == 0:
        y_softmax_train = np.array(y_softmax).reshape(
            np.array(y_softmax).shape[0], 1)
    else:
        y_softmax_train = np.concatenate((y_softmax_train, np.array(
            y_softmax).reshape(np.array(y_softmax).shape[0], 1)), axis=1)


for it in range(nClass):
    y_softmax = []
    Gradient_descent_w = gradient_desc_w_Mat[:, it]
    for i in range(x_tst.shape[0]):
        A = e**(np.inner(x_tst[i].reshape(x_tst[i].shape[0], 1).T,
                Gradient_descent_w[1:].T)+Gradient_descent_w[0])
        B = 0
        for ii in range(nClass):
            Gradient_descent_w_others = gradient_desc_w_Mat[:, ii]
            temp = e**(np.inner(x_tst[i].reshape(x_tst[i].shape[0], 1).T,
                       Gradient_descent_w_others[1:].T)+Gradient_descent_w_others[0])
            B += temp
        y_softmax.append((A/B)[0])

    if it == 0:
        y_softmax_test = np.array(y_softmax).reshape(
            np.array(y_softmax).shape[0], 1)
    else:
        y_softmax_test = np.concatenate((y_softmax_test, np.array(
            y_softmax).reshape(np.array(y_softmax).shape[0], 1)), axis=1)


# Find argmax
y_ova_train_out = np.argmax(y_softmax_train, axis=1).reshape(
    np.argmax(y_softmax_train, axis=1).shape[0], 1)
y_ova_test_out = np.argmax(y_softmax_test, axis=1).reshape(
    np.argmax(y_softmax_test, axis=1).shape[0], 1)

# Print accuracy
accuracyTrain = (np.where(y_trn == y_ova_train_out)
                 [0].shape[0]) / (y_trn.shape[0])
print('Train accuracy: ', accuracyTrain, '\n', '----------')
accuracyTest = (np.where(y_tst == y_ova_test_out)
                [0].shape[0]) / (y_tst.shape[0])
print('Test accuracy:  ', accuracyTest)
