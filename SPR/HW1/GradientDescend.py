import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt

########## Function Definitions ##########
def normalize(X):
    Min = X.min()
    Max = X.max()
    return (X - Min) / (Max - Min), Min, Max

def biasAddition(X):
    ones = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    return np.concatenate((ones, X.reshape(X.shape[0], 1)), axis=1)


def ErrorCalculation(X, y, theta):
    return (y.reshape(y.shape[0], 1) - X.dot(theta)).T.dot(y.reshape(y.shape[0], 1) - X.dot(theta))[0][0]


def mseError(y,y_hat):
    mse = np.mean(np.power(y - y_hat, 2))
    return mse


def gradientDescent(X, y, alpha=0.001, epsilon=0.0001):
    m, n = X.shape
    theta_0 = np.random.rand(n).reshape(n, 1)
    theta_1 = np.zeros((n, 1))
    i = 0
    errors = []
    while norm(theta_1 - theta_0) > epsilon:
        theta_0 = theta_1
        rss = ErrorCalculation(X, y, theta_0)
        temp = (X.T.dot(y))
        grad_rss = (np.matmul(X.T.dot(X), (theta_0)) -
                    temp.reshape(temp.shape[0], 1))
        theta_1 = theta_0 - alpha * grad_rss
        #print(i, np.squeeze(rss))
        errors.append(np.squeeze(rss))
        i += 1

    errors = np.array(errors)
    return theta_1, errors


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

# Run
gradientDescent_theta, Errors = gradientDescent(
    x_trnBiasAdded, y_trn, alpha=0.00001)

# Output gradientDescent
y_GradientDescent_Train = gradientDescent_theta[0] + \
    gradientDescent_theta[1]*x_trn
y_GradientDescent_Test = gradientDescent_theta[0] + \
    gradientDescent_theta[1]*x_tst


mseTrain = mseError(y_trn,y_GradientDescent_Train)
mseTest = mseError(y_tst,y_GradientDescent_Test)


# Print Theta
print('################# Calculated Thetas ######################')
print('Gradient Descent Solution, theta0, theta1: \n',
      gradientDescent_theta, '\n', '-')
print('MSE Train:\n', mseTrain , '\n', '-')
print('MSE Test:\n', mseTest , '\n', '-')

# Plot Gradient Descent Errors
plt.figure(figsize=[8, 6])
plt.plot(Errors, 'g.', linewidth=3.0)
plt.xlabel('Iteration ', fontsize=16)
plt.ylabel('Errors', fontsize=16)
plt.title('Gradient Descent Errors')
plt.show()

# Plot train data predicts
plt.figure(figsize=[8, 6])
plt.scatter(x_trn,y_trn, color='#458B74')
plt.scatter(x_tst,y_tst, color='#E3CF57')
plt.legend(['Train data', 'Test data'], fontsize=18)
plt.xlabel('Samples ', fontsize=16)
plt.ylabel('Errors ', fontsize=16)
plt.title('GD Data Predicts')
plt.show()


# Plot regression line
theta1, theta0 = np.polyfit(y_GradientDescent_Test, y_tst, 1)
plt.figure(figsize=[8, 6])
plt.scatter(y_GradientDescent_Test, y_tst, color='#458B74')
plt.xlabel('Outputs ', fontsize=16)
plt.ylabel('Targets ', fontsize=16)
plt.plot(y_GradientDescent_Test, theta1*y_GradientDescent_Test + theta0, color='#E3CF57')
plt.text(1, 100, 'Y =' + np.array2string(theta1) + '*X + ' + np.array2string(theta0), fontsize=14)
plt.title('GD plot regression line')
plt.show()
