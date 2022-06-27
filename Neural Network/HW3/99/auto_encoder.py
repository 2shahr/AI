import pandas as pd
from keras.layers import Dense, Input, BatchNormalization
from keras import Model
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from mpl_toolkits import mplot3d

def load_data(xl_file_name, trn_ratio):
    df = pd.read_excel(xl_file_name)
    data = df.values[:, 1:].astype('float32')
    np.random.shuffle(data)

    n_samples = data.shape[0]
    n_trn = int(np.round(trn_ratio * n_samples))

    x_trn = data[:n_trn]
    m = x_trn.mean(axis=0)
    s = x_trn.std(axis=0)

    x_trn = (x_trn - m) / s

    x_tst = data[n_trn:]
    x_tst = (x_tst - m) / s

    return x_trn, x_tst

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9999

def train_model(x_trn, z):
    n_features = x_trn.shape[1]
    input_feat = Input(shape=(n_features,))
    encode_1 = Dense(16, activation='relu')(input_feat)
    encode_2 = Dense(8, activation='relu')(encode_1)
    encode_3 = Dense(z, activation='relu')(encode_2)
    decode_1 = Dense(8, activation='relu')(encode_3)
    decode_2 = Dense(16, activation='relu')(decode_1)
    decode_3 = Dense(n_features)(decode_2)

    autoencoder = Model(input_feat, decode_3)
    encoder = Model(input_feat, encode_3)

    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    callback = keras.callbacks.LearningRateScheduler(scheduler)
    autoencoder.fit(x_trn, x_trn,
                    epochs=1000,
                    batch_size=100,
                    shuffle=True, 
                    callbacks=[callback])
    return encoder

xl_file_name = 'Gordon-2002_LungCancer.xlsx'
trn_ratio = 0.7
x_trn, x_tst= load_data(xl_file_name, trn_ratio)

dbi = {}
for z in range(2, 6):
    encoder = train_model(x_trn, z)
    prd = encoder.predict(x_tst)
    kmeans = KMeans(n_clusters=2).fit(prd)
    labels = kmeans.predict(prd)
    dbi[z] = davies_bouldin_score(prd, labels)
    if z == 2:
        plt.figure()
        ind = labels == 0
        plt.plot(prd[ind , 0], prd[ind, 1], '*r')
        plt.plot(prd[np.logical_not(ind), 0], prd[np.logical_not(ind), 1], '*k')
        plt.savefig('2D.png')
    if z == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ind = labels == 0
        ax.plot3D(prd[ind , 0], prd[ind, 1], prd[ind, 2], '*r')
        ax.plot3D(prd[np.logical_not(ind), 0], prd[np.logical_not(ind), 1], prd[np.logical_not(ind), 2], '*k')
        plt.savefig('3D.png')

for z in dbi:
    print('Z = {}, DBI = {}'.format(z, dbi[z]))