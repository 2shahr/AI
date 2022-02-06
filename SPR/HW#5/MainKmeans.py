import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2

#################### Functions ############


def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min)


def determine_cluster(data, centers):
    """
    Given all datapoints, dedicate each datapoints to each clusters.
    """
    Idxs = []
    for datapoint in data:
        Distance = np.sqrt(np.sum((centers - datapoint)**2, axis=1))
        Idx = np.argmin(Distance)
        Idxs.append(Idx)

    return Idxs


#################### Main ############

# Load data
im = cv2.imread('bird.tiff')
im = cv2.resize(im, (128, 128))
data = im.reshape(im.shape[0]*im.shape[1], 3)
data = normalize(data)
k = 16
Epsilon = 0.001


# Initialize cluster centers randomly
ClusterCenters = np.array(random.sample(list(data), k))
Initial_ClusterCenters = np.copy(ClusterCenters)


# Update cluster centers until stops changing
Iteration = -1
Check = True
temp = 1000000
while Check | Iteration < 500:
    New_Idxs = determine_cluster(data, ClusterCenters)

    PastClusterCenters = ClusterCenters.copy()
    Iteration += 1

    # Recalculate centers
    for id_ in range(k):
        ClusterIDXs = np.where(np.array(New_Idxs) == id_)
        ClusterDatas = data[ClusterIDXs]
        Initial_ClusterCenters[id_] = ClusterDatas.mean(axis=0)

    Distance = np.sum(
        np.sqrt(np.sum((PastClusterCenters - Initial_ClusterCenters)**2, axis=1)))

    if Distance <= Epsilon:
        Check = False
        ClusterCenters = Initial_ClusterCenters
        IDXs = New_Idxs
    elif Distance < temp:
        ClusterCenters = Initial_ClusterCenters
        temp = Distance.copy()
        IDXs = New_Idxs
        print(f"Iteration: {Iteration}, Distance: {Distance} ")


Colors = np.array([[0.46663701, 0.19818511, 0.918568],
                   [0.7296014, 0.82003624, 0.64035005],
                   [0.18459927, 0.14383448, 0.41897129],
                   [0.14435445, 0.79165082, 0.44222553],
                   [0.21371182, 0.79249431, 0.81066021],
                   [0.81845007, 0.59614111, 0.41948179],
                   [0.4607801, 0.07012321, 0.61808313],
                   [0.26835536, 0.61204746, 0.76828199],
                   [0.03277759, 0.95405671, 0.99732531],
                   [0.41166301, 0.12939992, 0.50887978],
                   [0.79803232, 0.31153822, 0.80284813],
                   [0.18133617, 0.85467717, 0.22001079],
                   [0.96114245, 0.15384556, 0.03829935],
                   [0.49977452, 0.28331044, 0.75198352],
                   [0.62176266, 0.07626428, 0.26679827],
                   [0.2388419, 0.23650574, 0.67443637]])

Reconstructed = np.zeros((len(IDXs), 3))
for i in range(len(IDXs)):
    Reconstructed[i, :] = Colors[IDXs[i], :]

Reconstructed = Reconstructed.reshape(im.shape[0], im.shape[1], 3)

plt.figure(figsize=[8, 6])
plt.subplot(2, 1, 1)
plt.imshow(im)
plt.title('Original image', fontsize=16)
plt.subplot(2, 1, 2)
plt.imshow(Reconstructed)
plt.title('Reconstructed image', fontsize=16)
plt.show()
