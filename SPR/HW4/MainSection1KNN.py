import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


#################### Functions ############

def normalpdf(X, mu, sigma):

    det = np.linalg.det(sigma)
    const = 1/(((np.pi*2)**2)*((det)**0.5))
    invSigma = np.linalg.pinv(sigma)
    PDF = const*np.exp(-0.5 * np.sum((X-mu).dot(invSigma)*(X-mu), axis=1))

    return PDF


def KNNPDF(X00, X01, Inputs, k):
    Xs = []
    Ys = []
    Ps = []
    for i in range(0, X00.shape[0]):
        for j in range(0, X01.shape[0]):
            Xs.append(X00[i])
            Ys.append(X01[j])
            n = X00.shape[0]
            d = [np.linalg.norm(np.array([X00[i], X01[j]])-y) for y in Inputs]
            d.sort()
            v = math.pi*d[k-1]**2
            if v == 0:
                Ps.append(1)
            else:
                Ps.append(k/(n*v))

    return np.array(Ps), np.array(Xs), np.array(Ys)


#################### Main ############


# Load data
# class0
mean = [2, 5]
cov = [[2, 0], [0, 2]]
Inputs0 = np.random.multivariate_normal(mean, cov, 100)
Targets0 = np.zeros([Inputs0.shape[0], 1])

# class0
mean = [8, 1]
cov = [[3, 1], [1, 3]]
Inputs1 = np.random.multivariate_normal(mean, cov, 100)
Targets1 = np.ones([Inputs1.shape[0], 1])

# class0
mean = [5, 3]
cov = [[2, 1], [1, 2]]
Inputs2 = np.random.multivariate_normal(mean, cov, 100)
Targets2 = 2*np.ones([Inputs2.shape[0], 1])


# For dataset 1
PX = {}
Ps = {}
PsReal = {}
for h in range(1, 99, 9):
    PX[h], Xs, Ys = KNNPDF(Inputs0[:, 0], Inputs0[:, 1], Inputs0, h)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PX[h], '.g')
    plt.title('KNN-based PDF, dataset1, k= '+str(h))
    plt.show()

    PsReal[h] = normalpdf(np.concatenate(
        (Xs.reshape(Xs.shape[0], 1), Ys.reshape(Ys.shape[0], 1)), axis=1), mean, cov)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PsReal[h], '.g')
    plt.title('Normal PDF, dataset1, k= '+str(h))
    plt.show()


# For dataset 2
PX = {}
Ps = {}
PsReal = {}
for h in range(1, 99, 9):
    PX[h], Xs, Ys = Inputs0(Inputs1[:, 0], Inputs1[:, 1], Inputs1, h)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PX[h], '.g')
    plt.title('KNN-based PDF, dataset2, k= '+str(h))
    plt.show()

    PsReal[h] = normalpdf(np.concatenate(
        (Xs.reshape(Xs.shape[0], 1), Ys.reshape(Ys.shape[0], 1)), axis=1), mean, cov)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PsReal[h], '.g')
    plt.title('Normal PDF, dataset2, k= '+str(h))
    plt.show()


#import pdb; pdb.set_trace()


# For dataset 2
PX = {}
Ps = {}
PsReal = {}
for h in range(1, 99, 9):
    PX[h], Xs, Ys = KNNPDF(Inputs2[:, 0], Inputs2[:, 1], Inputs2, h)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PX[h], '.g')
    plt.title('KNN-based PDF, dataset3, k= '+str(h))
    plt.show()

    PsReal[h] = normalpdf(np.concatenate(
        (Xs.reshape(Xs.shape[0], 1), Ys.reshape(Ys.shape[0], 1)), axis=1), mean, cov)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PsReal[h], '.g')
    plt.title('Normal PDF, dataset3, k= '+str(h))
    plt.show()
