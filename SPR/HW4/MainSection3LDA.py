import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


#################### Functions ############

def get_file_list(directory=os.path.dirname(os.path.realpath(__file__))):
    list = os.listdir(directory)
    return list


def LDA(X, y, Ncomponents):

    height, width = X.shape
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)

    scatter_t = np.cov(X.T)*(height - 1)
    scatter_w = 0
    for i in range(num_classes):
        class_items = np.flatnonzero(y == unique_classes[i])
        scatter_w = scatter_w + np.cov(X[class_items].T) * (len(class_items)-1)

    scatter_b = scatter_t - scatter_w
    _, eig_vectors = np.linalg.eigh(np.linalg.pinv(scatter_w).dot(scatter_b))
    pc = X.dot(eig_vectors[:, ::-1][:, 0:Ncomponents])
#
#    if Ncomponents == 2:
#        if y is None:
#            plt.scatter(pc[:,0],pc[:,1])
#        else:
#            colors = ['r','g','b']
#            labels = np.unique(y)
#            for color, label in zip(colors, labels):
#                class_data = pc[np.flatnonzero(y==label)]
#                plt.scatter(class_data[:,0],class_data[:,1],c=color)
#        plt.show()

    X_reconstructed = np.dot(
        pc, eig_vectors[:, ::-1][:, 0:Ncomponents].transpose()) + (np.mean(X, axis=0))

    return pc, X_reconstructed


def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min), min, max

#################### Main ############


# Load dataset
Path = 'jaffe'
Temp1 = get_file_list('jaffe')
Inputs = []
Classes = []
labels = {1: 'AN', 2: 'DI', 3: 'FE', 4: 'HA', 5: 'NE', 6: 'SA', 7: 'SU'}
for i in range(len(Temp1)):
    im = cv2.imread(os.path.join(Path, Temp1[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, [64, 64])
    temm = Temp1[i]
    if temm[3:5] == 'AN':
        Class = 0
    elif temm[3:5] == 'DI':
        Class = 1
    elif temm[3:5] == 'FE':
        Class = 2
    elif temm[3:5] == 'HA':
        Class = 3
    elif temm[3:5] == 'NE':
        Class = 4
    elif temm[3:5] == 'SA':
        Class = 5
    elif temm[3:5] == 'SU':
        Class = 6
    else:
        import pdb
        pdb.set_trace()

    Inputs.append(im.reshape((64*64, 1)))
    Classes.append(Class)

Inputs = np.array(Inputs)
Classes = np.array(Classes)
Inputs = Inputs.reshape(Inputs.shape[0], Inputs.shape[1])


# normalize
Inputs, minX, maxX = normalize(Inputs)
#x_trn = (x_trn - minX) / (maxX - minX)


# What is the problem of applying Fisher LDA to the dataset?
# Answer: Out of memory error
#X_new = LDA(Inputs, Classes, Ncomponents=3)


# Reconstruct the original data by using K basis vectors obtained from LDA. (Show
# reconstructed images of one person for k=1, 6, 29).
for k in [1, 40, 120]:
    X_new, X_Reconstruncted = LDA(
        Inputs[0:10, :], Classes[0:10], Ncomponents=k)
    # show 1 image
    plt.figure(figsize=(1.8 * 3, 2.4 * 3))
    im = X_Reconstruncted[0, :]
    plt.imshow(im.reshape(im.shape[0], 1).reshape(
        64, 64), interpolation='nearest')
    plt.title('Reconstructed with K = ' + str(k), fontsize=16)


# Plot the MSE between the original and reconstructed images in terms of the number
# of eigenvectors
MSE_mat = []
for k in range(0, 4096):
    X_new, X_Reconstruncted = LDA(
        Inputs[0:10, :], Classes[0:10], Ncomponents=k)
    MSE = ((Inputs[0:10, :].reshape(Inputs[0:10, :].shape[0]*Inputs[0:10, :].shape[1], 1) -
           X_Reconstruncted.reshape(Inputs[0:10, :].shape[0]*Inputs[0:10, :].shape[1], 1))**2).mean()
    MSE_mat.append(MSE)

fig = plt.figure(figsize=(10, 10))
plt.plot(range(0, 4096), np.array(MSE_mat), '.-g')
plt.xlabel('K')
plt.ylabel('MSE')
plt.title('MSE between the original and reconstructed images', fontsize=16)
plt.show()


# import pdb; pdb.set_trace()
