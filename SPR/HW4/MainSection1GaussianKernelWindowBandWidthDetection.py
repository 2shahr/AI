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
Inputs0 = np.random.multivariate_normal(mean, cov, 50)
Targets0 = np.zeros([Inputs0.shape[0], 1])


KernelSigma = [[0.6, 0], [0, 0.6]]
MeanErrors = []
for h in np.arange(0.2, 4, 0.2):
    Errors = []
    for fold in range(1, 6):
        Range0 = (fold-1)*10
        Range1 = (fold)*10 - 1
        PX, Xs, Ys = gaussian_kernel_windowPDF(
            Inputs0[Range0:Range1, 0], Inputs0[Range0:Range1, 1], h, KernelSigma)
        PsReal = normalpdf(np.concatenate(
            (Xs.reshape(Xs.shape[0], 1), Ys.reshape(Ys.shape[0], 1)), axis=1), mean, cov)
        MSE = np.mean((PX-PsReal)**2)
        Errors.append(MSE)

    MeanErrors.append(np.mean(Errors))


ArgMin = np.argmin(np.array(MeanErrors))

fig = plt.figure(figsize=[8, 6])
plt.plot(np.arange(0.2, 4, 0.2), np.array(MeanErrors), '.-b')
plt.plot(np.arange(0.2, 4, 0.2)[ArgMin], np.array(
    MeanErrors)[ArgMin], '*r', markersize=10)
plt.xlabel('h')
plt.ylabel('Mean 5-fold MSE')
plt.title('GaussianKernel-based PDF, mean MSE errors per different h')
plt.show()
