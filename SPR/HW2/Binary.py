import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import e


#################### Functions ############

def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min), min, max


def bias_addition(X):
    ones = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    return np.concatenate((ones, X), axis=1)


def SGDforLR(x, y, alpha, maxIter):
    # Stochastic gradient descent for two class (0,1) logistic regression with static learning rate
    m, n = x.shape
    w_0 = np.random.rand(n).reshape(n, 1)
    w_1 = np.zeros((n, 1))
    it = 0                      # iterations
    while (it <= maxIter):
        for i in range(m):
            w_0 = w_1
            a = e**(np.inner(x[i][1:].reshape(x[i]
                    [1:].shape[0], 1).T, w_0[1:].T)+w_0[0])
            b = (
                1.+e**(np.inner(x[i][1:].reshape(x[i][1:].shape[0], 1).T, w_0[1:].T)+w_0[0]))
            logisticFunction = a/b
            error = (y[i]-logisticFunction)
            # Update weights using SGD based on the cross entropy
            w_1 = w_0 + (alpha*error[0]*x[i]).T.reshape(w_0.shape)
#            print(w_1.T)
        it += 1
    return w_1

#################### Main ############


# Load data
raw_data = pd.read_csv(r'iris.data', names=[
                       'F1', 'F2', 'F3', 'F4', 'Class'], skiprows=1, sep='\,', engine='python')
Data = raw_data.loc[raw_data['Class'] !=
                    'Iris-versicolor', ['F1', 'F2', 'Class']]

# Split to train and test
idx1 = Data['Class'] == 'Iris-virginica'
idx2 = Data['Class'] == 'Iris-setosa'
class1data = Data.loc[idx1, ['F1', 'F2']]
class2data = Data.loc[idx2, ['F1', 'F2']]
class1target = np.zeros(Data.loc[idx1, ['Class']].shape[0]).reshape(
    Data.loc[idx1, ['Class']].shape[0], 1)
class2target = np.ones(Data.loc[idx2, ['Class']].shape[0]).reshape(
    Data.loc[idx2, ['Class']].shape[0], 1)
trainIndex1 = np.round(0.8*class1data.shape[0])
trainIndex2 = np.round(0.8*class2data.shape[0])

x_trn = np.concatenate([class1data.values[0:trainIndex1.astype(
    int)], class2data[0:trainIndex2.astype(int)]], axis=0)
x_test = np.concatenate([class1data.values[trainIndex1.astype(
    int):class1data.shape[0]+1], class2data[trainIndex2.astype(int):class2data.shape[0]+1]], axis=0)
y_trn = np.concatenate([class1target[0:trainIndex1.astype(
    int)], class2target[0:trainIndex2.astype(int)]], axis=0)
y_tst = np.concatenate([class1target[trainIndex1.astype(int):class1data.shape[0]+1],
                       class2target[trainIndex2.astype(int):class2target.shape[0]+1]], axis=0)

# Normalize
x_trn, minX, maxX = normalize(x_trn)
x_test = (x_test - minX) / (maxX - minX)

# Add bias
x_trnBiasAdded = bias_addition(x_trn)

# Run SGD
gradient_desc_w = SGDforLR(x_trnBiasAdded, y_trn, alpha=0.003, maxIter=1000)

# Show w
print('################# Calculated Ws ######################')
print('Gradient Descent Solution, thetas: \n',
      gradient_desc_w, '\n', '-----------')


# Output Gradient_descent
def y_gradient_desc(x):
    y_gd = []
    for i in range(x.shape[0]):
        a = e**(np.inner(x[i].reshape(x[i].shape[0], 1).T,
                gradient_desc_w[1:].T)+gradient_desc_w[0])
        b = (1.+e**(np.inner(x[i].reshape(x[i].shape[0],
             1).T, gradient_desc_w[1:].T)+gradient_desc_w[0]))
        y_gd.append((a/b)[0])
    y_gd = np.round(y_gd)
    return y_gd


y_gradient_desc_train = y_gradient_desc(x_trn)
y_gradient_desc_test = y_gradient_desc(x_test)


# Decision boundary plot
decision_boundary = - \
    (gradient_desc_w[0] + gradient_desc_w[1]
     * x_test[:, 0]) / (gradient_desc_w[2])
print('###########################')
print('Decision boundary formula:')
print('-(' + str(gradient_desc_w[0][0]) + ' + ' + str(
    gradient_desc_w[0][0]) + '*X1) / (' + str(gradient_desc_w[2][0]) + ')')
print('###########################')
plt.figure(figsize=[8, 6])
plt.plot(x_test[np.where(y_tst == 0)[0], 0],
         x_test[np.where(y_tst == 0)[0], 1], 'bo')  # blue circles
plt.plot(x_test[np.where(y_tst == 1)[0], 0],
         x_test[np.where(y_tst == 1)[0], 1], 'r*')  # red stars
plt.plot(x_test[:, 0], decision_boundary, 'g-', linewidth=3.0)  # green line
plt.xlabel('Feature 1 ', fontsize=16)
plt.ylabel('Feature 2 ', fontsize=16)
plt.legend(['Iris-virginica', 'Iris-setosa', 'Decision boundary'], fontsize=18)
plt.show()


# print accuracy
accuracy_trn = (np.where(y_trn == y_gradient_desc_train)
                [0].shape[0]) / (y_trn.shape[0])
print('Train accuracy:', accuracy_trn, '\n', '-----------')
accuracy_tst = (np.where(y_tst == y_gradient_desc_test)
                [0].shape[0]) / (y_tst.shape[0])
print('Test accuracy:', accuracy_tst)
