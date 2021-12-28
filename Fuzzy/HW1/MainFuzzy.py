import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import pair_confusion_matrix


#################### Functions ############

def normalize(X):
    min = X.min(axis=0)
    max = X.max(axis=0)
    return (X - min) / (max - min), min, max


def distances_calculation(Inputs):
    distance_matrix = []
    for k1 in range(len(Inputs)):
        matrix_line = []
        for k2 in range(len(Inputs)):
            if k1 >= k2:
                matrix_line.append(0.0)
            elif k1 < k2:
                Dis = np.sqrt(
                    sum((Inputs[k1][i] - Inputs[k2][i]) ** 2 for i in range(len(Inputs[k1]))))
                matrix_line.append(Dis)
        distance_matrix.append(matrix_line)

    final_distance_matrix = np.array(
        distance_matrix) + np.array(distance_matrix).T
    return final_distance_matrix


def r_from_distances_calculation(Dist_Mat):
    R_matrix = []
    for k in range(len(Dist_Mat)):
        max_d = max(Dist_Mat[k])
        R_matrix.append(
            [1 - Dist_Mat[k][i] / max_d for i in range(len(Dist_Mat[k]))])
    return np.array(R_matrix)


#################### Main ############

# Load data
Raw_data = pd.read_csv(r'hcvdat0.csv')
Temp = pd.DataFrame(Raw_data).to_numpy()


# Remove Nans
Data = Temp[~pd.isnull(Temp).any(axis=1)]


# Convert logical values to numerical
X = Data[:, 2::]
Y = Data[:, 1]
Y[Y == '0=Blood Donor'] = 0
Y[Y == '0s=suspect Blood Donor'] = 0
Y[Y == '1=Hepatitis'] = 1
Y[Y == '2=Fibrosis'] = 2
Y[Y == '3=Cirrhosis'] = 3

X[X[:, 1] == 'f', 1] = 1
X[X[:, 1] == 'm', 1] = 2


# Normalize
x_normal, minX, maxX = normalize(X)


# Distances Calculation
Distance_Matrix = distances_calculation(X)


# Construct R matrix
normalized_distance_matrix = r_from_distances_calculation(Distance_Matrix)


# Clustering
with pd.ExcelWriter(r'Results.xlsx') as writer:

    AlphaCuts = np.arange(0, 1, 0.05)
    Accuracies = []
    for AlphaCut in AlphaCuts:

        UsedObservations = []
        m, n = np.shape(normalized_distance_matrix)
        Clusts = []
        Iter = -1
        for i in range(0, m):
            for j in range(0, n):
                if (i not in UsedObservations) & (j not in UsedObservations) & (i != j):
                    Iter += 1
                    Clusts.append([])
                    if normalized_distance_matrix[i][j] >= AlphaCut:
                        Clusts[Iter].append(i)
                        Clusts[Iter].append(j)
                        UsedObservations.append(i)
                        UsedObservations.append(j)
                    else:
                        Clusts[Iter].append(i)
                        UsedObservations.append(i)

                elif (i in UsedObservations) & (j not in UsedObservations) & (i != j):
                    if normalized_distance_matrix[i][j] >= AlphaCut:
                        Clusts[Iter].append(j)
                        UsedObservations.append(j)

        NCluster = len(Clusts)
        FER_Clustering = np.zeros(np.shape(Y))
        for IterClust in range(len(Clusts)):
            FER_Clustering[Clusts[IterClust]] = IterClust

        temp = pair_confusion_matrix(Y, np.array(FER_Clustering))
        Accuracies.append(np.sum(np.diag(temp))/np.sum(temp))

        # Save results
        df1 = pd.DataFrame([Y, FER_Clustering])
        df1 = df1.rename(index={0: "Real Class", 1: "Our Class"})
        df1.to_excel(writer, sheet_name=str(AlphaCut))

writer.save()
writer.close()

df2 = pd.DataFrame([AlphaCuts, Accuracies])
df2 = df2.rename(index={0: "Alpha Cuts", 1: "Accuracies"})
with pd.ExcelWriter(r'Accuracies.xlsx') as writer:
    df2.to_excel(writer, sheet_name='Accuracies')

# Plot accuracies per different alpha cuts
plt.figure(figsize=[8, 6])
plt.plot(AlphaCuts, Accuracies, '.-k', markersize=10.0)
plt.xlabel('Alpha Cuts ', fontsize=16)
plt.ylabel('Accuracies ', fontsize=16)
plt.show()
