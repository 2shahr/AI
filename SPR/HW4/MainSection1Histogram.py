import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#################### Functions ############

def normalpdf(X, mu, sigma):

    det = np.linalg.det(sigma)
    const = 1/(((np.pi*2)**2)*((det)**0.5))
    invSigma = np.linalg.pinv(sigma)
    PDF = const*np.exp(-0.5 * np.sum((X-mu).dot(invSigma)*(X-mu), axis=1))

    return PDF


def HistogramPDF(X00, X01, Inputs, h):
    Xs = []
    Ys = []
    Ps = []
    for i in range(1, X00.shape[0]):
        for j in range(1, X01.shape[0]):
            Xs.append(X00[i])
            Ys.append(X01[j])
            temp1 = (np.logical_and(
                Inputs[:, 0] >= X00[i-1], Inputs[:, 0] < X00[i]))
            temp2 = (np.logical_and(
                Inputs[:, 1] >= X01[j-1], Inputs[:, 1] < X01[j]))
            P = np.where(np.logical_and(temp1, temp2))
            try:
                Ps.append(len(P[0])/(h*(Inputs.shape[0]**2)))
            except:
                Ps.append(0)

    return np.array(Ps), np.array(Xs), np.array(Ys)


#################### Main ############

# Load data
# class0
mean = [2, 5]
cov = [[2, 0], [0, 2]]
Inputs0 = np.random.multivariate_normal(mean, cov, 1000)
Targets0 = np.zeros([Inputs0.shape[0], 1])

# class0
mean = [8, 1]
cov = [[3, 1], [1, 3]]
Inputs1 = np.random.multivariate_normal(mean, cov, 1000)
Targets1 = np.ones([Inputs1.shape[0], 1])

# class0
mean = [5, 3]
cov = [[2, 1], [1, 2]]
Inputs2 = np.random.multivariate_normal(mean, cov, 1000)
Targets2 = 2*np.ones([Inputs2.shape[0], 1])


# For dataset 1
PX = {}
Ps = {}
PsReal = {}
for h in [0.09, 0.3, 0.6]:
    Min = Inputs0.min(axis=0)
    Min0 = Min[0]
    Min1 = Min[1]
    Max = Inputs0.max(axis=0)
    Max0 = Max[0]
    Max1 = Max[1]
    X00 = np.arange(Min0, Max0, h)
    X01 = np.arange(Min1, Max1, h)
    PX[h], Xs, Ys = HistogramPDF(X00, X01, Inputs0, h)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PX[h], '.g')
    plt.title('Histogram-based PDF, dataset1, h= '+str(h))
    plt.show()

    PsReal[h] = normalpdf(np.concatenate(
        (Xs.reshape(Xs.shape[0], 1), Ys.reshape(Ys.shape[0], 1)), axis=1), mean, cov)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PsReal[h], '.g')
    plt.title('Normal PDF, dataset1, h= '+str(h))
    plt.show()


# For dataset 2
PX = {}
Ps = {}
PsReal = {}
for h in [0.09, 0.3, 0.6]:
    Min = Inputs1.min(axis=0)
    Min0 = Min[0]
    Min1 = Min[1]
    Max = Inputs1.max(axis=0)
    Max0 = Max[0]
    Max1 = Max[1]
    X00 = np.arange(Min0, Max0, h)
    X01 = np.arange(Min1, Max1, h)
    PX[h], Xs, Ys = HistogramPDF(X00, X01, Inputs1, h)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PX[h], '.g')
    plt.title('Histogram-based PDF, dataset2, h= '+str(h))
    plt.show()

    PsReal[h] = normalpdf(np.concatenate(
        (Xs.reshape(Xs.shape[0], 1), Ys.reshape(Ys.shape[0], 1)), axis=1), mean, cov)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PsReal[h], '.g')
    plt.title('Normal PDF, dataset2, h= '+str(h))
    plt.show()


#import pdb; pdb.set_trace()


# For dataset 2
PX = {}
Ps = {}
PsReal = {}
for h in [0.09, 0.3, 0.6]:
    Min = Inputs2.min(axis=0)
    Min0 = Min[0]
    Min1 = Min[1]
    Max = Inputs2.max(axis=0)
    Max0 = Max[0]
    Max1 = Max[1]
    X00 = np.arange(Min0, Max0, h)
    X01 = np.arange(Min1, Max1, h)
    PX[h], Xs, Ys = HistogramPDF(X00, X01, Inputs2, h)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PX[h], '.g')
    plt.title('Histogram-based PDF, dataset3, h= '+str(h))
    plt.show()

    PsReal[h] = normalpdf(np.concatenate(
        (Xs.reshape(Xs.shape[0], 1), Ys.reshape(Ys.shape[0], 1)), axis=1), mean, cov)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PsReal[h], '.g')
    plt.title('Normal PDF, dataset3, h= '+str(h))
    plt.show()
