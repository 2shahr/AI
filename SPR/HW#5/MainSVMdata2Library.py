
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import scipy.io as sio

#################### Functions ############


def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min)


#################### Main ############

# Load data
mat_contents = sio.loadmat('Dataset2.mat')
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
    for C in [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]:
        for Sigma in [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]:
            # Train
            clf = svm.SVC(kernel='rbf', C=C, gamma=Sigma)
            clf.fit(x_trn, y_trn)

            # Test
            Outputs = clf.predict(x_tst)

            accuracy = accuracy_score(np.array(y_tst, dtype=float), Outputs)
            Accuracy.append(accuracy)
    Accuracies.append(Accuracy)


allC = [1, 100]
Best = np.argmax(np.mean(np.array(Accuracies), axis=0))
Acc = np.max(np.mean(np.array(Accuracies), axis=0))


# Accuracies import pdb; pdb.set_trace()
MeanOF5Fold = np.mean(np.array(Accuracies), axis=0)
X = []
Y = []
Z = []
qq = -1
for x in [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]:
    for y in [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]:
        qq += 1
        X.append(x)
        Y.append(y)
        Z.append(MeanOF5Fold[qq])
        if qq == Best:
            BestC = x
            BestSigma = y

fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig)
ax.plot3D(X, Y, Z, '.g', markersize=15)
plt.xlabel('C', fontsize=18)
plt.ylabel('Sigma', fontsize=18)
plt.show()
print(f"Best C: {BestC}, Best Sigma: {BestSigma}, with accuracy= {Acc} ")
