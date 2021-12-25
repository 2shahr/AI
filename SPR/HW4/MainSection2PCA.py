import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2

#################### Functions ############


def get_file_list(directory=os.path.dirname(os.path.realpath(__file__))):
    list = os.listdir(directory)
    return list


def PCA(X, Ncomponents):

    # Subtract the mean of each variable
    X_meanreferenced = X - np.mean(X, axis=0)

    # Calculate the Covariance Matrix
    covariance_mat = np.cov(X_meanreferenced, rowvar=False)

    # Compute the Eigenvalues and Eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_mat)

    # Sort Eigenvalues in descending order
    index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[index]
    sorted_eigenvectors = eigen_vectors[:, index]

    # Calculate explained variance
    total_egnvalues = sum(sorted_eigenvalue)
    ExplainedVariance = [(i/total_egnvalues) for i in sorted_eigenvalue]

    # Select a subset from the rearranged Eigenvalue matrix
    PrincipalComponents = sorted_eigenvectors[:, 0:Ncomponents]

    # Transform the data
    X_new = np.dot(PrincipalComponents.transpose(),
                   X_meanreferenced.transpose()).transpose()

    # Reconstruct transformed data
    if Ncomponents == 1:
        Proj = X_new.reshape(-1, 1)
    else:
        Proj = X_new

    X_reconstructed = np.dot(
        Proj, PrincipalComponents.transpose()) + (np.mean(X, axis=0))

    return X_new, np.array(ExplainedVariance), PrincipalComponents, X_reconstructed


def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min), min, max

#################### Main ############


# Load dataset
Path = 'jaffe'
Temp1 = get_file_list('jaffe')
Inputs = []
# show 9 image
plt.figure(figsize=(1.8 * 3, 2.4 * 3))
for i in range(0, 9):
    im = cv2.imread(os.path.join(Path, Temp1[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, [64, 64])
    plt.subplot(3, 3, i + 1)
    plt.imshow(im, cmap=plt.cm.gray)

plt.show()

for i in range(len(Temp1)):
    im = cv2.imread(os.path.join(Path, Temp1[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, [64, 64])
    Inputs.append(im.reshape((64*64, 1)))

Inputs = np.array(Inputs)
Inputs = Inputs.reshape(Inputs.shape[0], Inputs.shape[1])


# normalize
Inputs, minX, maxX = normalize(Inputs)
#x_trn = (x_trn - minX) / (maxX - minX)


# PCA
X_new, ExplainedVariance, PrincipalComponents, X_Reconstruncted = PCA(
    Inputs, Ncomponents=3)

# 2d plot
fig = plt.figure(figsize=(10, 10))
plt.plot(X_new[:, 0], X_new[:, 1], '.g')
plt.title('Reduced dataset', fontsize=16)
plt.show()

# 3D plot
fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig)
ax.plot3D(X_new[:, 0], X_new[:, 1], X_new[:, 2], '.g')
plt.title('Reduced dataset', fontsize=16)
plt.show()


# Reconstruct the original data using K principle components
for k in [1, 40, 120]:
    X_new, ExplainedVariance, PrincipalComponents, X_Reconstruncted = PCA(
        Inputs, Ncomponents=k)
    # show 9 image
    plt.figure(figsize=(1.8 * 3, 2.4 * 3))
    for i in range(0, 9):
        im = X_Reconstruncted[i, :]
        plt.subplot(3, 3, i + 1)
        plt.imshow(im.reshape(im.shape[0], 1).reshape(
            64, 64), interpolation='nearest')
        plt.title('Reconstructed with K= ' + str(k), fontsize=16)


# Plot the MSE between the original and reconstructed images in terms of the number
# of eigenvectors
MSE_mat = []
for k in range(0, 10):
    X_new, ExplainedVariance, PrincipalComponents, X_Reconstruncted = PCA(
        Inputs, Ncomponents=k)
    MSE = ((Inputs.reshape(Inputs.shape[0]*Inputs.shape[1], 1) -
           X_Reconstruncted.reshape(Inputs.shape[0]*Inputs.shape[1], 1))**2).mean()
    MSE_mat.append(MSE)

fig = plt.figure(figsize=(10, 10))
plt.plot(range(0, 10), np.array(MSE_mat), '.-g')
plt.xlabel('K')
plt.ylabel('MSE')
plt.title('MSE between the original and reconstructed images', fontsize=16)
plt.show()


# Visualize some of the first principal components.
# 2d plot
fig = plt.figure(figsize=(10, 10))
plt.plot(PrincipalComponents[:, 0], PrincipalComponents[:, 1], '.g')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA 2D plot', fontsize=16)
plt.show()

# 3D plot
fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig)
ax.plot3D(PrincipalComponents[:, 0],
          PrincipalComponents[:, 1], PrincipalComponents[:, 2], '.g')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.clabel('PC 3')
plt.title('PCA 3D plot', fontsize=16)
plt.show()

# How many principal components are enough so that you have acceptable reconstruction? How do you select them?
# Solution:using explained variance, as more than 90 %
X_new, ExplainedVariance, PrincipalComponents = PCA(
    Inputs, Ncomponents=Inputs.shape[1])

DefinesVar = 0
i = -1
while DefinesVar <= .90:
    i += 1
    DefinesVar = DefinesVar+ExplainedVariance[i]
print('Based on our best knowledge, Number of PCs= ' + str(i+1) +
      ' With total explained variance= ' + str(DefinesVar) + ' is best')

# import pdb; pdb.set_trace()
