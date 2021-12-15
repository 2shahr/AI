import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt

########## Function Definitions ##########
def normalize(X):
    Min = X.min()
    Max = X.max()
    return (X - Min) / (Max - Min), Min, Max

def biasAddition(X):
    ones = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    return np.concatenate((ones, X.reshape(X.shape[0], 1)), axis=1)


def leastSquared(X, y):
    theta = inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


def mseError(y,y_hat):
    mse = np.mean(np.power(y - y_hat, 2))
    return mse

################# Main Code #################

# Load data
train_data = pd.read_csv('Data-Train.csv')
test_data = pd.read_csv('Data-Test.csv')

x_trn = train_data['x'].values
y_trn = train_data['y'].values

x_tst = test_data['x'].values
y_tst = test_data['y'].values

x_trn, trnMinX, trnMaxX = normalize(x_trn)
x_tst, tstMinX, tstMaxX = normalize(x_tst)

# Add bias
x_trnBiasAdded = biasAddition(x_trn)
x_tstBiasAdded = biasAddition(x_tst)

# Run Least Squared
leastSquared_theta = leastSquared(x_trnBiasAdded, y_trn)

# Output closed form
yHat_trn = leastSquared_theta[0] + leastSquared_theta[1] * x_trn
yHat_tst = leastSquared_theta[0] + leastSquared_theta[1] * x_tst


# Mean Squared Error
mseTrain = mseError(y_trn,yHat_trn)
mseTest = mseError(y_tst,yHat_tst)


# Print theta
print('################# Calculated Thetas ######################')
print(' Closed Form Solution, theta0, theta1 :\n ', leastSquared_theta, '\n', '-')
print('MSE Train:\n', mseTrain, '\n', '-')
print('MSE Test:\n', mseTest, '\n', '-')


# Plot Data Predicts
plt.figure(figsize=[8, 6])
plt.scatter(x_trn,y_trn, color='lightblue')
plt.scatter(x_tst,y_tst, color='lightcoral')
plt.legend(['Train data', 'Test Data'], fontsize=18)
plt.xlabel('Samples ', fontsize=16)
plt.ylabel('Errors ', fontsize=16)
plt.title('Least Squared Data Predicts', fontsize=16)
plt.show()


# Plot regression line
theta1, theta0 = np.polyfit(yHat_tst, y_tst, 1)
plt.figure(figsize=[8, 6])
plt.scatter(yHat_tst, y_tst, color='lightblue')
plt.xlabel('X ', fontsize=16)
plt.ylabel('y ', fontsize=16)
plt.plot(yHat_tst, theta1 * yHat_tst + theta0, color='lightcoral')
plt.text(1, 100, 'Y =' + np.array2string(theta1) + '*X + ' + np.array2string(theta0), fontsize=14)
plt.title('LS Plot regression line', fontsize=16)
plt.show()

