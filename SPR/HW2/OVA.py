import numpy as np
import pandas as pd
from math import e
import matplotlib.pyplot as plt


#################### Functions ############

def normalize(X):
    min = X.min()
    Max = X.max()
    return (X - min) / (Max - min), min, Max


def bias_addition(X):
    ones = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    return np.concatenate((ones, X), axis=1)


def SGDforLR(x, y, alpha, MaxIter):
    # Stochastic gradient descent - logistic regression with static learning rate
    m, n = x.shape
    w_0 = np.random.rand(n).reshape(n, 1)
    w_1 = np.zeros((n, 1))
    i = 0
    it = 0
    errors = []
    while (it <= MaxIter):
        Out = []
        for i in range(m):
            w_0 = w_1
            A = e**(np.inner(x[i][1:].reshape(x[i][1:].shape[0], 1).T, w_0[1:].T)+w_0[0])
            B = (1+e**(np.inner(x[i][1:].reshape(x[i][1:].shape[0], 1).T, w_0[1:].T)+w_0[0]))
            LogisticFunction = A/B
            error = (y[i]-LogisticFunction)
            # Update weights using SGD based on the cross entropy
            w_1 = w_0 + (alpha*error[0]*x[i]).T.reshape(w_0.shape)
            Out.append(LogisticFunction[0][0])
            # print(w_1.T)
            i += 1
        Out = np.array(Out)
        MSE = np.mean((y-Out.reshape(Out.shape[0], 1))**2)
        errors.append(MSE)
        it += 1
    return w_1, np.array(errors)

# ################## Main ############
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
x_trn, minX, MaxX = normalize(x_trn)
x_tst = (x_tst - minX) / (MaxX - minX)


# Add bias
x_trnBiasAdded = bias_addition(x_trn)

# Run SGD
nClass = 3
ovaModels = []
for i in range(nClass):
    ova = np.zeros(y_trn.shape)
    ova[y_trn == i] = 1
    gradient_desc_w, errors = SGDforLR(
        x_trnBiasAdded, ova, alpha=0.003, MaxIter=1000)
    ovaModels.append(gradient_desc_w)

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(errors, 'm', linewidth=3.0)
    plt.title(str(i+1) + ' versus others', fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('error', fontsize=16)
    plt.show()

    # Show w
    print('################# Calculated Ws for class: ', i+1, '#################')
    print('Gradient Descent Solution, thetas: \n',
          gradient_desc_w.T, '\n', '-----------')


# Output Gradient_descent
for it in range(nClass):
    y_ova = []
    gradient_desc_w = ovaModels[it]
    for i in range(x_trn.shape[0]):
        A = e**(np.inner(x_trn[i].reshape(x_trn[i].shape[0],
                1).T, gradient_desc_w[1:].T)+gradient_desc_w[0])
        B = (1.+e**(np.inner(x_trn[i].reshape(x_trn[i].shape[0],
             1).T, gradient_desc_w[1:].T)+gradient_desc_w[0]))
        y_ova.append((A/B)[0])

    if it == 0:
        y_ova_train = np.array(y_ova)
    else:
        y_ova_train = np.concatenate((y_ova_train, np.array(y_ova)), axis=1)


for it in range(nClass):
    y_ova = []
    gradient_desc_w = ovaModels[it]
    for i in range(x_tst.shape[0]):
        A = e**(np.inner(x_tst[i].reshape(x_tst[i].shape[0], 1).T,
                gradient_desc_w[1:].T)+gradient_desc_w[0])
        B = (1.+e**(np.inner(x_tst[i].reshape(x_tst[i].shape[0],
             1).T, gradient_desc_w[1:].T)+gradient_desc_w[0]))
        y_ova.append((A/B)[0])

    if it == 0:
        y_ova_test = np.array(y_ova)
    else:
        y_ova_test = np.concatenate((y_ova_test, np.array(y_ova)), axis=1)


# Find argmax
y_ova_train_out = np.argmax(y_ova_train, axis=1).reshape(
    np.argmax(y_ova_train, axis=1).shape[0], 1)
y_ova_test_out = np.argmax(y_ova_test, axis=1).reshape(
    np.argmax(y_ova_test, axis=1).shape[0], 1)

# Print accuracy
accuracy_trn = (np.where(y_trn == y_ova_train_out)
                [0].shape[0]) / (y_trn.shape[0])
print('Train accuracy: ', accuracy_trn, '\n', '----------')
accuracy_tst = (np.where(y_tst == y_ova_test_out)
                [0].shape[0]) / (y_tst.shape[0])
print('Test accuracy:  ', accuracy_tst)
