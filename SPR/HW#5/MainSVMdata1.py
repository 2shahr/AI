import scipy.io as sio
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
xrange = range

#################### Functions ############


def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min)


def matrix_of_kernel(train_x):
    numSamples = np.shape(train_x)[0]
    kernealMat = np.mat(np.zeros((numSamples, numSamples)))
    for i in xrange(numSamples):
        kernealMat[:, i] = train_x * (train_x[i]).reshape(1, -1).T

    return kernealMat


class templateSVM(object):
    def __init__(self, trainX, trainY, c, tolerance, maxIteration):
        self.train_x = trainX
        self.train_y = trainY
        self.C = c
        self.toler = tolerance
        self.maxIter = maxIteration
        self.numSamples = np.shape(trainX)[0]
        self.alphas = np.mat(np.zeros((self.numSamples, 1)))
        self.b = 0
        self.errorCache = np.mat(np.zeros((self.numSamples, 2)))
        self.kernelMat = matrix_of_kernel(self.train_x)


def calcError(svm, alpha_i):
    func_i = np.multiply(svm.alphas, svm.train_y).T * \
        svm.kernelMat[:, alpha_i] + svm.b
    erro_i = func_i - svm.train_y[alpha_i]
    return erro_i


def updateError(svm, alpha_j):
    error = calcError(svm, alpha_j)
    svm.errorCache[alpha_j] = [1, error]


def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i]
    alpha_index = np.nonzero(svm.errorCache[:, 0])[0]
    maxstep = float("-inf")
    alpha_j, error_j = 0, 0
    if len(alpha_index) > 1:
        for alpha_k in alpha_index:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_i - error_k) > maxstep:
                maxstep = abs(error_i - error_k)
                alpha_j = alpha_k
                error_j = error_k
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = random.randint(0, svm.numSamples - 1)
        error_j = calcError(svm, alpha_j)
    return alpha_j, error_j


def innerLoop(svm, alpha_i):
    error_i = calcError(svm, alpha_i)
    error_i_ago = copy.deepcopy(error_i)
    if (svm.train_y[alpha_i] * error_i < -svm.toler and svm.alphas[alpha_i] < svm.C) or \
            (svm.train_y[alpha_i] * error_i > svm.toler and svm.alphas[alpha_i] > 0):
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
        alpha_i_ago = copy.deepcopy(svm.alphas[alpha_i])
        alpha_j_ago = copy.deepcopy(svm.alphas[alpha_j])
        error_j_ago = copy.deepcopy(error_j)
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] - \
            svm.kernelMat[alpha_j, alpha_j]

        svm.alphas[alpha_j] = alpha_j_ago - \
            svm.train_y[alpha_j] * (error_i - error_j) / eta
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        elif svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L
        svm.alphas[alpha_i] = alpha_i_ago + svm.train_y[alpha_i] * svm.train_y[alpha_j] * \
            (alpha_j_ago - svm.alphas[alpha_j])
        if abs(alpha_j_ago - svm.alphas[alpha_j]) < 10 ** (-5):
            return 0

        b1 = svm.b - error_i_ago - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_ago) * \
            svm.kernelMat[alpha_i, alpha_i] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_ago) * \
            svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j_ago - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_ago) * \
            svm.kernelMat[alpha_i, alpha_j] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_ago) * \
            svm.kernelMat[alpha_j, alpha_j]
        if (svm.alphas[alpha_i] > 0) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (svm.alphas[alpha_j] > 0) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2

        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        return 1
    else:
        return 0


def trainSVM(train_x, train_y, c, toler, maxIter):
    svm = templateSVM(train_x, train_y, c, toler, maxIter)
    entire = True
    alphaPairsChanged = 0
    iter = 0
    while (iter < svm.maxIter) and ((alphaPairsChanged > 0) or entire):
        alphaPairsChanged = 0
        if entire:
            for i in xrange(svm.numSamples):
                alphaPairsChanged += innerLoop(svm, i)
            iter += 1
        else:
            nonBound_index = np.nonzero(
                (svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            for i in nonBound_index:
                alphaPairsChanged += innerLoop(svm, i)
            iter += 1
        if entire:
            entire = False
        elif alphaPairsChanged == 0:
            entire = True
    return svm


def testSVM(svm, test_x, test_y):
    numTest = np.shape(test_x)[0]
    supportVect_index = np.nonzero(svm.alphas.A > 0)[0]
    supportVect = svm.train_x[supportVect_index]
    supportLabels = svm.train_y[supportVect_index]
    supportAlphas = svm.alphas[supportVect_index]
    labelpredict = []
    for i in xrange(numTest):
        kernelValue = supportVect * (test_x[i, :]).reshape(1, -1).T
        predict = kernelValue.T * \
            np.multiply(supportLabels, supportAlphas) + svm.b
        labelpredict.append(int(np.sign(predict)))

    return labelpredict


def visualize(Data, Y, w, b):
    plt.figure(figsize=[8, 6])
    plt.plot(Data[np.where(Y == 1), 0],
             Data[np.where(Y == 1), 1], '.b', markersize=15)
    plt.plot(Data[np.where(Y == -1), 0],
             Data[np.where(Y == -1), 1], '.r', markersize=15)
    min_feature_value = Data.min(axis=0)
    max_feature_value = Data.max(axis=0)

    # hyperplane = x.w+b (actually its a line)
    def hyperplane(x, w, b, v):
        # returns a x2 value on line when given x1
        return (-w[0]*x-b+v)/w[1]

    hyp_x_min = min_feature_value*0.9
    hyp_x_max = max_feature_value*1.1

    # (w.x+b)=1
    # positive support vector hyperplane
    pav1 = hyperplane(hyp_x_min[0, 0], w, b, 1)
    pav2 = hyperplane(hyp_x_max[0, 0], w, b, 1)
    plt.plot([hyp_x_min[0, 0], hyp_x_max[0, 0]],
             [pav1[0, 0], pav2[0, 0]], '-k')

    # (w.x+b)=-1
    # negative support vector hyperplane
    nav1 = hyperplane(hyp_x_min[0, 0], w, b, -1)
    nav2 = hyperplane(hyp_x_max[0, 0], w, b, -1)
    plt.plot([hyp_x_min[0, 0], hyp_x_max[0, 0]],
             [nav1[0, 0], nav2[0, 0]], '-k')

    # (w.x+b)=0
    # db support vector hyperplane
    db1 = hyperplane(hyp_x_min[0, 0], w, b, 0)
    db2 = hyperplane(hyp_x_max[0, 0], w, b, 0)
    plt.plot([hyp_x_min[0, 0], hyp_x_max[0, 0]], [db1[0, 0], db2[0, 0]], 'y--')
    plt.show()

#################### Main ############


# Load data
mat_contents = sio.loadmat('Dataset1.mat')
Inputs = mat_contents['X']
Targets = mat_contents['y']


# Train and test split
x_trn, x_tst, y_trn, y_tst = train_test_split(
    Inputs, Targets, test_size=0.2, random_state=42)
y_trn = np.array(y_trn == 0, dtype=float)
y_tst = np.array(y_tst == 0, dtype=float)
y_trn[y_trn == 0] = -1
y_tst[y_tst == 0] = -1

x_trn = np.matrix(x_trn)
x_tst = np.matrix(x_tst)
y_trn = np.matrix(y_trn.reshape(-1, 1))
y_tst = np.matrix(y_tst.reshape(-1, 1))


# Train SVM
Accuracies = []
for kfold in range(0, 5):
    Accuracy = []
    for C in [1, 100]:
        # Train
        toler = 0.001
        maxIter = 150
        svmClassifier = trainSVM(x_trn, y_trn, C, toler, maxIter)

        # Test
        Outputs = testSVM(svmClassifier, x_tst, y_tst)

        accuracy = accuracy_score(np.array(y_tst, dtype=float), Outputs)
        Accuracy.append(accuracy)
    Accuracies.append(Accuracy)


allC = [1, 100]
Best = np.argmax(np.mean(np.array(Accuracies), axis=0))
Acc = np.max(np.mean(np.array(Accuracies), axis=0))
BestC = allC[Best]
print(f"Best C: {BestC}, with accuracy= {Acc} ")


# Accuracies
MeanOF5Fold = np.mean(np.array(Accuracies), axis=0)
plt.figure(figsize=[8, 6])
plt.plot(allC, MeanOF5Fold, '.r', linewidth=3.0)
plt.xlabel('C ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)

# Boundaries import pdb; pdb.set_trace()
svmClassifier = trainSVM(x_trn, y_trn, BestC, toler, maxIter)
supportVect_index = np.nonzero(svmClassifier.alphas.A > 0)[0]
supportLabels = svmClassifier.train_y[supportVect_index]
supportAlphas = svmClassifier.alphas[supportVect_index]
supportVect = svmClassifier.train_x[supportVect_index]
w = supportVect.T*np.multiply(supportLabels, supportAlphas)
visualize(x_tst, y_tst, w, svmClassifier.b)
