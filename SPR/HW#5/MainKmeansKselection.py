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


def Kmeans(data, k, Epsilon):
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
        WCSS = 0    # (Within cluster sum of square)
        for id_ in range(k):
            ClusterIDXs = np.where(np.array(New_Idxs) == id_)
            ClusterDatas = data[ClusterIDXs]
            Initial_ClusterCenters[id_] = ClusterDatas.mean(axis=0)
            WCSS += np.sum(
                np.sqrt(np.sum((Initial_ClusterCenters[id_] - ClusterDatas)**2, axis=1)))

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

    return IDXs, WCSS/k

#################### Main ############


# Load data
im = cv2.imread('bird.tiff')
im = cv2.resize(im, (128, 128))
data = im.reshape(im.shape[0]*im.shape[1], 3)
data = normalize(data)
Epsilon = 0.001


AllIDXs = {}
AllWCSS = []
it = -1
for k in range(2, 40):
    it += 1
    IDXs, WCSS = Kmeans(data, k, Epsilon)
    AllIDXs[it] = IDXs
    AllWCSS.append(WCSS)


Best = np.argmin(np.array(AllWCSS))
Ns = range(2, 40)
plt.figure(figsize=[8, 6])
plt.plot(Ns, AllWCSS, '.-b')
plt.plot(Ns[Best], AllWCSS[Best], '*r')
plt.title('Original image', fontsize=16)
plt.xlabel('N cluster')
plt.ylabel('WCSS')
plt.show()
#
#
