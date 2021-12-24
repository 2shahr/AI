import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
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
    for classes in [0, 1, 2]:
        temp_mu.append(X[Y == classes].mean(axis=0))
        Mu = np.array(temp_mu)

        Mu_Yj[Y == classes, :] = Mu[classes, :]

    temp = np.zeros([Mu.shape[1], Mu.shape[1]])
    X0 = X[Y == 0]
    for j in range(X0.shape[0]):
        temp += (X0[j, :]-Mu_Yj[j, :]).reshape(Mu.shape[1],
                                               1).dot((X0[j, :]-Mu_Yj[j, :]).reshape(Mu.shape[1], 1).T)

    Sigma0 = temp/X0.shape[0]

    X0 = X[Y == 1]
    for j in range(X0.shape[0]):
        temp += (X0[j, :]-Mu_Yj[j, :]).reshape(Mu.shape[1],
                                               1).dot((X0[j, :]-Mu_Yj[j, :]).reshape(Mu.shape[1], 1).T)

    Sigma1 = temp/X0.shape[0]

    X0 = X[Y == 2]
    for j in range(X0.shape[0]):
        temp += (X0[j, :]-Mu_Yj[j, :]).reshape(Mu.shape[1],
                                               1).dot((X0[j, :]-Mu_Yj[j, :]).reshape(Mu.shape[1], 1).T)

    Sigma2 = temp/X0.shape[0]

    return Mu, Sigma0, Sigma1, Sigma2, P


def Prediction(X, Mu, Sigma0, Sigma1, Sigma2, P, Nclass):
    Probabilities = []
    Sigmas = []
    Sigmas.append(Sigma0)
    Sigmas.append(Sigma1)
    Sigmas.append(Sigma2)
    for i in range(Nclass):
        Sigma = Sigmas[i]
        mu = Mu[i, :]
        invSigma = np.linalg.pinv(Sigma)
        Phi = ((P)**i * (1 - P)**(1 - i))
        Probability = Phi * \
            (np.exp(-1.0 * np.sum((X-mu).dot(invSigma)*(X-mu), axis=1)))
        Probabilities.append(Probability)

    Classes = np.argmax(Probabilities, axis=0)
    return Classes, Probabilities


def Roots_of_equation(a, b, c):
    # Finding the value of Discriminant
    D = b*b - 4*a*c
    # other way, D = b**2 - 4*a*c

    sqrt_D = math.sqrt(abs(D))

    # checking Discriminant condition
    if D > 0:
        #        print("Roots are Real and Different ")
        #        print((-b + sqrt_D)/(2*a))
        #        print((-b - sqrt_D)/(2*a))
        Out = np.abs((-b + sqrt_D)/(2*a))

    elif D == 0:
        Out = (-b / (2*a))

    else:
        Out = 0

    return Out
#################### Main ############


# Load data
# class0
mean = [3, 6]
cov = [[1.5, 0], [0, 1.5]]
Inputs0 = np.random.multivariate_normal(mean, cov, 500)
Targets0 = np.zeros([Inputs0.shape[0], 1])
# class0
mean = [5, 4]
cov = [[2, 0], [0, 2]]
Inputs1 = np.random.multivariate_normal(mean, cov, 500)
Targets1 = np.ones([Inputs1.shape[0], 1])
# class0
mean = [6, 6]
cov = [[1, 0], [0, 1]]
Inputs2 = np.random.multivariate_normal(mean, cov, 500)
Targets2 = 2*np.ones([Inputs2.shape[0], 1])
# Concatenate
Inputs = np.concatenate((Inputs0, Inputs1, Inputs2), axis=0)
Targets = np.concatenate((Targets0, Targets1, Targets2), axis=0)


# Normalize
Inputs, minX, maxX = normalize(Inputs)


# Spliting
x_trn, x_tst, y_trn, y_tst = train_test_split(
    Inputs, Targets, test_size=0.2, random_state=42)
y_trn = y_trn.reshape(y_trn.shape[0])
y_tst = y_tst.reshape(y_tst.shape[0])


# Train
Mu, Sigma0, Sigma1, Sigma2, P = gaussian_linear_discriminant_analysis(
    x_trn, y_trn)
Nclass = Mu.shape[0]


# Test
Y_GLDA_Train, ProbabilitiesTrain = Prediction(
    x_trn, Mu, Sigma0, Sigma1, Sigma2, P, Nclass)
Y_GLDA_Test, ProbabilitiesTest = Prediction(
    x_tst, Mu, Sigma0, Sigma1, Sigma2, P, Nclass)


# Evaluation
Accuracy_trn = (np.where(y_trn == Y_GLDA_Train)[0].shape[0]) / (y_trn.shape[0])
print('Train accuracy: ', Accuracy_trn)


Accuracy_tst = (np.where(y_tst == Y_GLDA_Test)[0].shape[0]) / (y_tst.shape[0])
print('Test accuracy: ', Accuracy_tst)


# Decision boundary plot
P0 = len(np.where(y_trn == 0)[0])/y_trn.shape[0]
P1 = len(np.where(y_trn == 1)[0])/y_trn.shape[0]
P2 = len(np.where(y_trn == 2)[0])/y_trn.shape[0]
invSigma0 = np.linalg.pinv(Sigma0)
invSigma1 = np.linalg.pinv(Sigma1)
invSigma2 = np.linalg.pinv(Sigma2)
invSigmaDiff = invSigma0-invSigma1

# -0.5*np.log(Sigma0.dot(np.linalg.pinv(Sigma1)))
temp = (-1*((Mu[0, :].T.dot(invSigma0)).dot(Mu[0, :]))) + \
    (((Mu[1, :].T.dot(invSigma1)).dot(Mu[1, :])))
C = 2*(np.log((P0/P1))) - \
    np.log(np.linalg.norm(Sigma0)/np.linalg.norm(Sigma1)) + temp
temp2 = (invSigma0.dot(Mu[0, :]))-(invSigma1.dot(Mu[1, :]))


H = C-(invSigmaDiff[0, 0]*(x_tst[:, 0]**x_tst[:, 0]))-2*(x_tst[:, 0]*temp2[0])
V = (x_tst[:, 0]*invSigmaDiff[0, 0]) + \
    (x_tst[:, 0]*invSigmaDiff[1, 0]) - 2 * temp2[1]
Q = invSigmaDiff[1, 1]

Decision_boundary01 = np.zeros(V.shape[0])
for i in range(V.shape[0]):
    # Q*X^2 + V*X - H ==0
    Sol = Roots_of_equation(Q, V[i], -H[i])
    Decision_boundary01[i] = Sol

#import pdb; pdb.set_trace()
#Decision_boundary01 = -( -B + A[0]*x_tst[:,0]) / (A[1])
# print('###########################')
#print('Decision boundary formula:')
#print('-( -' + str(B) + ' + ' + str(A[0]) + '*X1) / (' + str(A[1]) + ')')
# print('###########################')
#
plt.figure(figsize=[8, 6])
plt.plot(x_tst[np.where((y_tst == 0) & (Y_GLDA_Test == 0))[0], 0], x_tst[np.where(
    (y_tst == 0) & (Y_GLDA_Test == 0))[0], 1], 'b.', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 1) & (Y_GLDA_Test == 1))[0], 0], x_tst[np.where(
    (y_tst == 1) & (Y_GLDA_Test == 1))[0], 1], 'r.', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 2) & (Y_GLDA_Test == 2))[0], 0], x_tst[np.where(
    (y_tst == 2) & (Y_GLDA_Test == 2))[0], 1], 'g.', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 0) & (Y_GLDA_Test != 0))[0], 0], x_tst[np.where(
    (y_tst == 0) & (Y_GLDA_Test != 0))[0], 1], 'bx', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 1) & (Y_GLDA_Test != 1))[0], 0], x_tst[np.where(
    (y_tst == 1) & (Y_GLDA_Test != 1))[0], 1], 'rx', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 2) & (Y_GLDA_Test != 2))[0], 0], x_tst[np.where(
    (y_tst == 2) & (Y_GLDA_Test != 2))[0], 1], 'gx', markersize=10.0)
#plt.plot(x_tst[:,0], Decision_boundary01, 'k-',linewidth=3.0)
#plt.plot(x_tst[:,0], Decision_boundary02, 'k-',linewidth=3.0)
#plt.plot(x_tst[:,0], Decision_boundary12, 'k-',linewidth=3.0)
plt.legend(['True class 0', 'True class 1', 'True class 2',
           'False class 0', 'False class 1', 'False class 2'], fontsize=12)
plt.xlabel('Feature 1 ', fontsize=16)
plt.ylabel('Feature 2 ', fontsize=16)
plt.show()


# Plot PDF 3D
F0 = np.inner(invSigma0[0, :], np.exp(-1.0 * (x_trn[np.where((y_trn == 0))[0]] -
              Mu[0, :]).dot(invSigma0)*(x_trn[np.where((y_trn == 0))[0]]-Mu[0, :])))
F1 = np.inner(invSigma1[0, :], np.exp(-1.0 * (x_trn[np.where((y_trn == 1))[0]] -
              Mu[1, :]).dot(invSigma1)*(x_trn[np.where((y_trn == 1))[0]]-Mu[1, :])))
F2 = np.inner(invSigma2[0, :], np.exp(-1.0 * (x_trn[np.where((y_trn == 2))[0]] -
              Mu[2, :]).dot(invSigma2)*(x_trn[np.where((y_trn == 2))[0]]-Mu[2, :])))
fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.plot3D(x_trn[np.where(y_trn == 0)[0], 0],
          x_trn[np.where(y_trn == 0)[0], 1], F0, '.g')
ax.plot3D(x_trn[np.where(y_trn == 1)[0], 0],
          x_trn[np.where(y_trn == 1)[0], 1], F1, '.r')
ax.plot3D(x_trn[np.where(y_trn == 2)[0], 0],
          x_trn[np.where(y_trn == 2)[0], 1], F2, '.b')
plt.title('3D PDFs ', fontsize=16)
plt.show()


# Contour with DB
plt.figure(figsize=[8, 6])
plt.plot(x_tst[np.where((y_tst == 0) & (Y_GLDA_Test == 0))[0], 0], x_tst[np.where(
    (y_tst == 0) & (Y_GLDA_Test == 0))[0], 1], 'b.', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 1) & (Y_GLDA_Test == 1))[0], 0], x_tst[np.where(
    (y_tst == 1) & (Y_GLDA_Test == 1))[0], 1], 'r.', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 2) & (Y_GLDA_Test == 2))[0], 0], x_tst[np.where(
    (y_tst == 2) & (Y_GLDA_Test == 2))[0], 1], 'g.', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 0) & (Y_GLDA_Test != 0))[0], 0], x_tst[np.where(
    (y_tst == 0) & (Y_GLDA_Test != 0))[0], 1], 'bx', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 1) & (Y_GLDA_Test != 1))[0], 0], x_tst[np.where(
    (y_tst == 1) & (Y_GLDA_Test != 1))[0], 1], 'rx', markersize=10.0)
plt.plot(x_tst[np.where((y_tst == 2) & (Y_GLDA_Test != 2))[0], 0], x_tst[np.where(
    (y_tst == 2) & (Y_GLDA_Test != 2))[0], 1], 'gx', markersize=10.0)
plt.xlabel('Feature 1 ', fontsize=16)
plt.ylabel('Feature 2 ', fontsize=16)
XX0 = x_tst[np.where((y_tst == 0))[0], 0]
XY0 = x_tst[np.where((y_tst == 0))[0], 1]
XX, XY = np.meshgrid(XX0, XY0)
ZZ = np.array([np.inner(invSigma0[0, :], np.exp(-1.0 * (np.array([xx, yy])-Mu[0, :]).dot(invSigma0)
              * (np.array([xx, yy])-Mu[0, :]))) for xx, yy in zip(np.ravel(XX), np.ravel(XY))])
zz = ZZ.reshape(XX.shape)
plt.contour(XX, XY, zz, 15, alpha=.3,)
XX0 = x_tst[np.where((y_tst == 1))[0], 0]
XY0 = x_tst[np.where((y_tst == 1))[0], 1]
XX, XY = np.meshgrid(XX0, XY0)
ZZ = np.array([np.inner(invSigma1[0, :], np.exp(-1.0 * (np.array([xx, yy])-Mu[1, :]).dot(invSigma1)
              * (np.array([xx, yy])-Mu[1, :]))) for xx, yy in zip(np.ravel(XX), np.ravel(XY))])
zz = ZZ.reshape(XX.shape)
plt.contour(XX, XY, zz, 15, alpha=.3)
XX0 = x_tst[np.where((y_tst == 2))[0], 0]
XY0 = x_tst[np.where((y_tst == 2))[0], 1]
XX, XY = np.meshgrid(XX0, XY0)
ZZ = np.array([np.inner(invSigma2[0, :], np.exp(-1.0 * (np.array([xx, yy])-Mu[2, :]).dot(invSigma2)
              * (np.array([xx, yy])-Mu[2, :]))) for xx, yy in zip(np.ravel(XX), np.ravel(XY))])
zz = ZZ.reshape(XX.shape)
plt.contour(XX, XY, zz, 15, alpha=.3)
plt.show()
