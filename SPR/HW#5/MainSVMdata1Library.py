import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.io as sio
from sklearn import svm

#################### Functions ############


def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min)


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
        clf = svm.SVC(kernel='linear', C=C)
        clf.fit(x_trn, y_trn)

        # Test
        Outputs = clf.predict(x_tst)

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
plt.plot(allC, MeanOF5Fold, '.-r', linewidth=3.0)
plt.xlabel('C ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)

# Boundaries
clf = svm.SVC(kernel='linear', C=BestC)
plt.figure(figsize=[8, 6])
plt.plot(x_tst[np.where(y_tst == -1), 0],
         x_tst[np.where(y_tst == -1), 1], '.r', markersize=12)
plt.plot(x_tst[np.where(y_tst == 1), 0],
         x_tst[np.where(y_tst == 1), 1], '.b', markersize=12)
clf.fit(x_trn, y_trn)
x_min, x_max = x_tst.min() - 1, x_tst.max() + 1
w = clf.coef_[0]
b = clf.intercept_[0]
x_points = np.linspace(x_min, x_max)    # generating x-points from -1 to 1
y_points = -(w[0] / w[1]) * x_points - b / w[1]
w_hat = clf.coef_[0] / (np.sqrt(np.sum(clf.coef_[0] ** 2)))
margin = 1 / np.sqrt(np.sum(clf.coef_[0] ** 2))
decision_boundary_points = np.array(list(zip(x_points, y_points)))
points_of_line_above = decision_boundary_points + w_hat * margin
points_of_line_below = decision_boundary_points - w_hat * margin
# Plotting a red hyperplane
plt.plot(x_points, y_points, c='r')
plt.plot(points_of_line_above[:, 0],
         points_of_line_above[:, 1],
         'b--',
         linewidth=2)
plt.plot(points_of_line_below[:, 0],
         points_of_line_below[:, 1],
         'g--',
         linewidth=2)
