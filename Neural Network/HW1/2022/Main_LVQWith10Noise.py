
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from skimage.util import random_noise
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import SimpSOM as sps

########### Functions ###########


def getFileList(directory=os.path.dirname(os.path.realpath(__file__))):
    list = os.listdir(directory)
    return list


def TrainLVQ(X, y, a, b, max_ep):
    c, train_idx = np.unique(y, True)
    model = sps.somNet(np.unique(y, True)[0].shape[0], 1, X_train, PBC=True)
    model.train(0.01, 200)    
    W = []
    for it in range(0, np.unique(y, True)[0].shape[0]):
        W.append(model.nodeList[it].weights)
    W = np.array(W)
#    W = X[train_idx].astype(np.float64)
    train = np.array([e for i, e in enumerate(
        zip(X, y)) if i not in train_idx])
    X = train[:, 0]
    y = train[:, 1]
    ep = 0

    while ep < max_ep:
        for i, x in enumerate(X):
            d = [sum((w-x)**2) for w in W]
            min = np.argmin(d)
            s = 1 if y[i] == c[min] else -1
            W[min] += s * a*(x - W[min])
        a *= b
        ep += 1
    return W, c


def TestLVQ(Inp, W):
    W, c = W
    d = [sum((w-Inp)**2) for w in W]
    Out = c[np.argmin(d)]
    return Out


########### Main code ###########
# Load dataset
Path = 'Dataset'
Temp1 = getFileList('Dataset')
Idx = -1
Labels = []
Inputs = []
for i in range(len(Temp1)):
    Idx += 1
    temp = getFileList(os.path.join(Path, Temp1[i]))
    for j in range(len(temp)):
        Labels.append(Idx)
        im0 = cv2.imread(os.path.join(Path, Temp1[i], temp[j]))
        im1 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        dim = (100, 100)
        Temp2 = cv2.resize(im1, dim)
        Temp2 = np.reshape(Temp2, (10000))
        Inputs.append(Temp2)

Inputs = np.array(Inputs)
Targets = np.array(Labels)

# Spliting dataset to train, validation and test
X_train, X_test, y_train, y_test = train_test_split(
    Inputs, Targets, test_size=0.2, random_state=42)


# Add Noise
for it in range(0, X_train.shape[0]):
    Inp = np.reshape(X_train[it, :, ], (100, 100))
    Inp = random_noise(Inp, mode='s&p', amount=0.1)
    X_train[it, :] = np.reshape(Inp, (10000))


# Train Model
epoch = 25
tingkatAkurasi1 = []
a = .1
b = .5
w = TrainLVQ(X_train, y_train, a, b, epoch)

# Test Model
Out_test = []
for i in range(y_test.shape[0]):
    Out_test.append(TestLVQ(X_test[i], w))
Out_test = np.array(Out_test)

# Results evaluation
result1 = precision_recall_fscore_support(y_test, Out_test)
Precision = np.mean(result1[0])
print("Precision: ", Precision)
Recall = np.mean(result1[1])
print("Recall: ", Recall)
Fscore = np.mean(result1[2])
print("Fscore: ", Fscore)
Accuracy = accuracy_score(y_test, Out_test)
print("Accuracy: ", Accuracy)
