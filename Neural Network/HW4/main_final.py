import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from sklearn import preprocessing
import matplotlib.pyplot as plt

def train_generator(x, y, batch_size):
    y = to_categorical(y)
    l = [i.shape[1] for i in x]
    num_batches = len(x) // batch_size
    batches = []
    for i in range(num_batches):
        batch = np.zeros((batch_size, 13, x[(i + 1) * batch_size - 1].shape[1]))
        for j in range(batch_size):
            batch[j, :x[i * batch_size + j].shape[0], :x[i * batch_size + j].shape[1]] = x[i * batch_size + j]
        batches.append((batch, y[i * batch_size : (i + 1) * batch_size, :]))
        
    rem = len(x) - num_batches * batch_size
    if rem != 0:
        i += 1
        batch = np.zeros((rem, 13, l[-1]))
        for j in range(rem):
            batch[j, :x[i * batch_size + j].shape[0], :x[i * batch_size + j].shape[1]] = x[i * batch_size + j]
        batches.append((batch, y[i * batch_size:, :]))
    
    i = -1
    while True:
        i = (i + 1) % len(batches)
        yield batches[i][0].swapaxes(1, 2), batches[i][1]

def load_data(file_adr, samples_per_digit, scaler=None):
    df = pd.read_csv(file_adr, header=None, sep=' ')
    x = df.values #returns a numpy array
    if scaler is None:
        scaler = preprocessing.StandardScaler()
        scaler.fit(x)
    x_scaled = scaler.transform(x)
    df = pd.DataFrame(x_scaled)
    ind = np.where(df[0].isnull().values)[0]
    if ind[0] != 0:
        ind = np.insert(ind, 0, -1)
    ind = np.insert(ind, ind.shape[0], df.shape[0])
    mfcs = [df.iloc[ind[i] + 1 : ind[i + 1]].values.T for i in range(ind.shape[0] - 1)]
    
    label = 0
    labels = np.zeros((len(mfcs),))
    for i in range(len(mfcs)):
        if i % samples_per_digit == 0:
            label += 1
        labels[i] = label
    labels = labels - 1
    l = [mfcs[i].shape[1] for i in range(len(mfcs))]
    order = np.argsort(l)
    labels = labels[order]
    
    list.sort(mfcs, key=np.shape)
    return mfcs, labels, scaler

trn_file_adr = 'Data/Train_Arabic_Digit.txt'
trn_mfcs, trn_y, scaler = load_data(trn_file_adr, 660)

tst_file_adr = 'Data/Test_Arabic_Digit.txt'
tst_mfcs, tst_y, _ = load_data(tst_file_adr, 220, scaler=scaler)

model = Sequential()
model.add(LSTM(50, input_shape = (None, 13), return_sequences=False, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

print(model.summary())

opt = RMSprop(learning_rate=0.001, rho=0.9)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

batch_size = 550
h = model.fit_generator(train_generator(trn_mfcs, trn_y, batch_size), 
            steps_per_epoch=6600//batch_size, epochs=1000, 
            validation_data=train_generator(tst_mfcs, tst_y, batch_size), validation_steps=2200/batch_size)

print('test accuracy: {}'.format(h.history['val_accuracy'][-1]))
plt.figure()
plt.plot(h.history['accuracy'])
plt.title('train accuracy')
plt.show(block=False)

plt.figure()
plt.plot(h.history['loss'])
plt.title('train loss')
plt.show(block=False)

plt.figure()
plt.plot(h.history['val_accuracy'])
plt.title('test accuracy')
plt.show(block=False)

plt.figure()
plt.plot(h.history['val_loss'])
plt.title('test loss')
plt.show(block=True)