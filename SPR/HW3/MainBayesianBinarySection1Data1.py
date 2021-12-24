import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#################### Functions ############

def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min), min, max


def gaussian_linear_discriminant_analysis(X, Y):
    P = Y.mean()
    temp_mu = []
    Mu_Yj = np.zeros(X.shape)
    for classes in [0, 1]:
        temp_mu.append(X[Y == classes].mean(axis=0))
        Mu = np.array(temp_mu)

        Mu_Yj[Y == classes, :] = Mu[classes, :]

    temp = np.zeros([Mu.shape[1], Mu.shape[1]])
    for j in range(X.shape[0]):
        temp += (X[j, :]-Mu_Yj[j, :]).reshape(Mu.shape[1],
                                              1).dot((X[j, :]-Mu_Yj[j, :]).reshape(Mu.shape[1], 1).T)

    Sigma = temp/X.shape[0]

    return Mu, Sigma, P


def Prediction(X, Mu, Sigma, P, Nclass):
    Probabilities = []
    for i in range(Nclass):
        mu = Mu[i, :]
        invSigma = np.linalg.pinv(Sigma)
        Phi = ((P)**i * (1 - P)**(1 - i))
        Probability = Phi * \
            (np.exp(-1.0 * np.sum((X-mu).dot(invSigma)*(X-mu), axis=1)))
        Probabilities.append(Probability)

    Classes = np.argmax(Probabilities, axis=0)
    return Classes, Probabilities


#################### Main ############


# Load data
Raw_data = pd.read_csv(r'BC-Train1.csv')
x_trn = Raw_data.values[:, 0:-1]
y_trn = Raw_data.values[:, -1]

Raw_data = pd.read_csv(r'BC-Test1.csv')
x_tst = Raw_data.values[:, 0:-1]
y_tst = Raw_data.values[:, -1]


# Normalize
All, minX, maxX = normalize(np.concatenate((x_trn, x_tst), axis=0))
x_trn = (x_trn - minX) / (maxX - minX)
x_tst = (x_tst - minX) / (maxX - minX)

# Train
Mu, Sigma, P = gaussian_linear_discriminant_analysis(x_trn, y_trn)
Nclass = Mu.shape[0]


# Test
Y_GLDA_Train, ProbabilitiesTrain = Prediction(x_trn, Mu, Sigma, P, Nclass)
Y_GLDA_Test, ProbabilitiesTest = Prediction(x_tst, Mu, Sigma, P, Nclass)

# Evaluation
TP = np.where((y_trn == 0) == (Y_GLDA_Train == 0))[0].shape[0]
FP = np.where((y_trn == 1) == (Y_GLDA_Train == 0))[0].shape[0]
FN = np.where((y_trn == 0) == (Y_GLDA_Train == 1))[0].shape[0]
TN = np.where((y_trn == 1) == (Y_GLDA_Train == 1))[0].shape[0]
Accuracy_trn = (np.where(y_trn == Y_GLDA_Train)[0].shape[0]) / (y_trn.shape[0])
PrecisionTrain = TP/(TP+FP)
RecallTrain = TP/(TP+FN)
F1Train = 2*((PrecisionTrain*RecallTrain)/(PrecisionTrain+RecallTrain))
print('Train accuracy: ', Accuracy_trn, 'Train precision: ',
      PrecisionTrain, 'Train recall: ', RecallTrain, 'Train F1: ', F1Train)

TP = np.where((y_tst == 0) == (Y_GLDA_Test == 0))[0].shape[0]
FP = np.where((y_tst == 1) == (Y_GLDA_Test == 0))[0].shape[0]
FN = np.where((y_tst == 0) == (Y_GLDA_Test == 1))[0].shape[0]
TN = np.where((y_tst == 1) == (Y_GLDA_Test == 1))[0].shape[0]
Accuracy_tst = (np.where(y_tst == Y_GLDA_Test)[0].shape[0]) / (y_tst.shape[0])
PrecisionTest = TP/(TP+FP)
RecallTest = TP/(TP+FN)
F1Test = 2*((PrecisionTest*RecallTest)/(PrecisionTest+RecallTest))
print('Test accuracy: ', Accuracy_tst, 'Test precision: ',
      PrecisionTest, 'Test recall: ', RecallTest, 'Test F1: ', F1Test)


# Decision boundary plot
invSigma = np.linalg.pinv(Sigma)
A = invSigma.dot(Mu[0, :]-Mu[1, :])
P0 = len(np.where(y_trn == 0)[0])/y_trn.shape[0]
P1 = len(np.where(y_trn == 1)[0])/y_trn.shape[0]
B = (((Mu[0, :].T.dot(invSigma)).dot(Mu[0, :]))*0.5) - \
    (((Mu[1, :].T.dot(invSigma)).dot(Mu[1, :]))*0.5) + np.log10((P0/P1))
Decision_boundary = -(-B + A[0]*x_tst[:, 0]) / (A[1])
print('###########################')
print('Decision boundary formula:')
print('-( -' + str(B) + ' + ' + str(A[0]) + '*X1) / (' + str(A[1]) + ')')
print('###########################')

plt.figure(figsize=[8, 6])
plt.plot(x_tst[np.where((y_tst == 0) & (Y_GLDA_Test == 0))[0], 0], x_tst[np.where(
    (y_tst == 0) & (Y_GLDA_Test == 0))[0], 1], 'm.', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 1) & (Y_GLDA_Test == 1))[0], 0], x_tst[np.where(
    (y_tst == 1) & (Y_GLDA_Test == 1))[0], 1], 'c.', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 0) & (Y_GLDA_Test == 1))[0], 0], x_tst[np.where(
    (y_tst == 0) & (Y_GLDA_Test == 1))[0], 1], 'yx', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 1) & (Y_GLDA_Test == 0))[0], 0], x_tst[np.where(
    (y_tst == 1) & (Y_GLDA_Test == 0))[0], 1], 'rx', markersize=10.0)
plt.plot(x_tst[:, 0], Decision_boundary, 'b-', linewidth=3.0)
plt.legend(['True class 0', 'True class 1', 'False class 0',
            'False class 1', 'Decision boundary'], fontsize=12)
plt.xlabel('Feature 1 ', fontsize=16)
plt.ylabel('Feature 2 ', fontsize=16)
plt.show()


# Plot PDF 3D
F0 = np.inner(invSigma[0, :], np.exp(-1.0 * (x_trn[np.where((y_trn == 0))[0]] -
                                             Mu[0, :]).dot(invSigma)*(x_trn[np.where((y_trn == 0))[0]]-Mu[0, :])))
F1 = np.inner(invSigma[1, :], np.exp(-1.0 * (x_trn[np.where((y_trn == 1))[0]] -
                                             Mu[1, :]).dot(invSigma)*(x_trn[np.where((y_trn == 1))[0]]-Mu[1, :])))
fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.plot(x_trn[np.where(y_trn == 0)[0], 0],
        x_trn[np.where(y_trn == 0)[0], 1], F0, '.m')
ax.plot(x_trn[np.where(y_trn == 1)[0], 0],
        x_trn[np.where(y_trn == 1)[0], 1], F1, '.c')
plt.title('3D PDFs ', fontsize=16)
plt.show()


# Contour with DB
plt.figure(figsize=[8, 6])
plt.plot(x_tst[np.where((y_tst == 0) & (Y_GLDA_Test == 0))[0], 0], x_tst[np.where(
    (y_tst == 0) & (Y_GLDA_Test == 0))[0], 1], 'm.', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 1) & (Y_GLDA_Test == 1))[0], 0], x_tst[np.where(
    (y_tst == 1) & (Y_GLDA_Test == 1))[0], 1], 'c.', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 0) & (Y_GLDA_Test == 1))[0], 0], x_tst[np.where(
    (y_tst == 0) & (Y_GLDA_Test == 1))[0], 1], 'yx', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 1) & (Y_GLDA_Test == 0))[0], 0], x_tst[np.where(
    (y_tst == 1) & (Y_GLDA_Test == 0))[0], 1], 'rx', markersize=10.0)
plt.plot(x_tst[:, 0], Decision_boundary, 'b-', linewidth=3.0)
plt.xlabel('Feature 1 ', fontsize=16)
plt.ylabel('Feature 2 ', fontsize=16)

XX0 = x_tst[np.where((y_tst == 0))[0], 0]
XY0 = x_tst[np.where((y_tst == 0))[0], 1]
XX, XY = np.meshgrid(XX0, XY0)
ZZ = np.array([np.inner(invSigma[0, :], np.exp(-1.0 * (np.array([xx, yy])-Mu[0, :]).dot(invSigma)
                                               * (np.array([xx, yy])-Mu[0, :]))) for xx, yy in zip(np.ravel(XX), np.ravel(XY))])
zz = ZZ.reshape(XX.shape)
plt.contour(XX, XY, zz, 15, alpha=.3)

XX0 = x_tst[np.where((y_tst == 1))[0], 0]
XY0 = x_tst[np.where((y_tst == 1))[0], 1]
XX, XY = np.meshgrid(XX0, XY0)
ZZ = np.array([np.inner(invSigma[1, :], np.exp(-1.0 * (np.array([xx, yy])-Mu[1, :]).dot(invSigma)
                                               * (np.array([xx, yy])-Mu[1, :]))) for xx, yy in zip(np.ravel(XX), np.ravel(XY))])
zz = ZZ.reshape(XX.shape)
plt.contour(XX, XY, zz, 15, alpha=.3)

plt.show()
