import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#################### Functions ############

def normalpdf(X, mu, sigma):

    det = np.linalg.det(sigma)
    const = 1/(((np.pi*2)**2)*((det)**0.5))
    invSigma = np.linalg.pinv(sigma)
    try:
        PDF = const*np.exp(-0.5 * np.sum((X-mu).dot(invSigma)*(X-mu), axis=1))
    except:
        PDF = const*np.exp(-0.5 * np.sum((X-mu).dot(invSigma)*(X-mu)))

    return PDF


def gaussian_kernel_windowPDF(X00, X01, h, KernelSigma):
    Xs = []
    Ys = []
    Ps = []
    for i in range(0, X00.shape[0]):
        for j in range(0, X01.shape[0]):
            Xs.append(X00[i])
            Ys.append(X01[j])
            temp1 = (np.logical_and(X00 > (X00[i]-h), X00 < (X00[i]+h)))
            temp2 = (np.logical_and(X01 > (X01[j]-h), X01 < (X01[j]+h)))
            temp = np.where(np.logical_or(temp1, temp2))
            if len(temp[0]) > 0:
                Sum = 0
                for kk in temp[0]:
                    for ii in temp[0]:
                        Prob = normalpdf(np.array([X00[i], X01[j]]), np.array(
                            [X00[kk], X01[ii]]), KernelSigma)
                        Sum += Prob

                Ps.append((1/(X00.shape[0]*X01.shape[0]*h))*(Sum))
            else:
                Ps.append(0)

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

KernelSigma = [[0.9, 0], [0, 0.9]]
# For dataset 1
PX = {}
Ps = {}
PsReal = {}
for h in [0.09, 0.3, 0.6]:
    PX[h], Xs, Ys = gaussian_kernel_windowPDF(
        Inputs0[:, 0], Inputs0[:, 1], h, KernelSigma)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PX[h], '.g')
    plt.title('GaussianKernel-based PDF, dataset1, h= '+str(h))
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
    PX[h], Xs, Ys = gaussian_kernel_windowPDF(
        Inputs1[:, 0], Inputs1[:, 1], h, KernelSigma)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PX[h], '.g')
    plt.title('GaussianKernel-based PDF, dataset2, h= '+str(h))
    plt.show()

    PsReal[h] = normalpdf(np.concatenate(
        (Xs.reshape(Xs.shape[0], 1), Ys.reshape(Ys.shape[0], 1)), axis=1), mean, cov)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PsReal[h], '.g')
    plt.title('Normal PDF, dataset2, h= '+str(h))
    plt.show()


#import pdb; pdb.set_trace()


# For dataset 3
PX = {}
Ps = {}
PsReal = {}
for h in [0.09, 0.3, 0.6]:
    PX[h], Xs, Ys = gaussian_kernel_windowPDF(
        Inputs2[:, 0], Inputs2[:, 1], h, KernelSigma)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PX[h], '.g')
    plt.title('GaussianKernel-based PDF, dataset3, h= '+str(h))
    plt.show()

    PsReal[h] = normalpdf(np.concatenate(
        (Xs.reshape(Xs.shape[0], 1), Ys.reshape(Ys.shape[0], 1)), axis=1), mean, cov)
    fig = plt.figure(figsize=[8, 6])
    ax = Axes3D(fig)
    ax.plot3D(Xs, Ys, PsReal[h], '.g')
    plt.title('Normal PDF, dataset3, h= '+str(h))
    plt.show()
