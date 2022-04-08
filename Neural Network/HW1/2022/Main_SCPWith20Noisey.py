
import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.util import random_noise
import cv2
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

########### Functions ###########


def getFileList(directory=os.path.dirname(os.path.realpath(__file__))):
    list = os.listdir(directory)
    return list


def getModel(x_train, train, x_validation, validation):

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(100, 100)))
    model.add(Flatten())
    model.add(Dense(19, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, train, batch_size=16, epochs=80,
                        verbose=1, validation_data=(x_validation, validation))

    return model, history


def getPrediction(model, x_train, train, x_test):
    model.evaluate(x_train, train, verbose=False)
    predictions = model.predict(x_test)
    return predictions


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
#        Temp2 = np.reshape(Temp2, (10000))
        Inputs.append(Temp2)

Inputs = np.array(Inputs)
Targets = np.array(Labels)


# Spliting dataset to train, validation and test
X_temp, X_test, y_temp, y_test = train_test_split(
    Inputs, Targets, test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_temp, y_temp, test_size=0.3, random_state=42)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_validation = to_categorical(y_validation)


# Add Noise
for it in range(0, X_train.shape[0]):
    X_train[it, :, :] = random_noise(X_train[it, :, :], mode='s&p', amount=0.2)

# Train RNN Model
model, history = getModel(X_train, y_train, X_validation, y_validation)

# Prediction
Out_test = getPrediction(model, X_train, y_train, X_test)
Out_test = np.round(Out_test)


# Results evaluation
result1 = precision_recall_fscore_support(
    np.argmax(y_test, axis=1), np.argmax(Out_test, axis=1))
Precision = np.mean(result1[0])
print("Precision: ", Precision)
Recall = np.mean(result1[1])
print("Recall: ", Recall)
Fscore = np.mean(result1[2])
print("Fscore: ", Fscore)
Accuracy = accuracy_score(np.argmax(y_test, axis=1),
                          np.argmax(Out_test, axis=1))
print("Accuracy: ", Accuracy)
