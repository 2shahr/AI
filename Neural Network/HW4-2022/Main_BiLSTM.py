
import os
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
import pandas as pd
from sklearn import preprocessing


########### Functions ###########

def getFileList(directory = os.path.dirname(os.path.realpath(__file__))):
    list = os.listdir(directory)
    return list


def DoingSlidingWindow(sample, annotation, X_temp, y_temp, WindowLenth):
    sample1 = sample[0:5000].astype(np.float64)
    sample2 = sample[5000:10000].astype(np.float64)
    sample3 = sample[10000:15000].astype(np.float64)
    annotation1 = annotation[0:5000]
    annotation2 = annotation[5000:10000]
    annotation3 = annotation[10000:15000]  
     
    for it in np.arange(0, sample1.shape[0], WindowLenth):
        temp = []
        SignalPart = sample1[it:it+WindowLenth]
        AnnotationPart = annotation1[it:it+WindowLenth]
        for jj in range(0, WindowLenth):
            if jj==(WindowLenth-1):
                temp.append(SignalPart[jj])
                yy = AnnotationPart[jj][0]
            else:
                temp.append(SignalPart[jj])                
                temp.append(AnnotationPart[jj][0])                
           
        X_temp.append(temp)
        y_temp.append(yy)                
    
    for it in np.arange(0, sample2.shape[0], WindowLenth):
        temp = []
        SignalPart = sample2[it:it+WindowLenth]
        AnnotationPart = annotation2[it:it+WindowLenth] 
        for jj in range(0, WindowLenth):
            if jj==(WindowLenth-1):
                temp.append(SignalPart[jj])
                yy = AnnotationPart[jj][0]
            else:
                temp.append(SignalPart[jj])
                temp.append(AnnotationPart[jj][0])                
            
        X_temp.append(temp)
        y_temp.append(yy)
        
    for it in np.arange(0, sample3.shape[0], WindowLenth):
        temp = []
        SignalPart = sample3[it:it+WindowLenth]
        AnnotationPart = annotation3[it:it+WindowLenth]        
        for jj in range(0, WindowLenth):
            if jj==(WindowLenth-1):
                temp.append(SignalPart[jj])
                yy = AnnotationPart[jj][0]
            else:
                temp.append(SignalPart[jj])
                temp.append(AnnotationPart[jj][0])                
            
        X_temp.append(temp)
        y_temp.append(yy)        
       
    return X_temp, y_temp


def getModel(x_train, x_validation, train, validation):

    # building a linear stack of layers with the sequential model
    model = Sequential()   
      

    # RNN layer
    model.add(layers.Bidirectional(layers.LSTM(20, return_sequences=True, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))) 
    
    # flatten output of conv
    model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
 
    
    ## Fully conected layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(4, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, train, epochs = 8, verbose = False, validation_data = (x_validation, validation), batch_size = 8)
    return model, history
    


def getPrediction(model, x_train, x_test, train):
    model.evaluate(x_train, train, verbose = False)
    predictions = model.predict(x_test)
    return predictions

    
########### Main code ###########

# Load dataset   
WindowLenth = 5
Path1 = 'samples'
Temp1 = getFileList('samples')
Path2 = 'annotations'
Temp2 = getFileList('annotations')
X_temp = []
y_temp = []
for j in range(7, 60):    
    sample = pd.read_csv(os.path.join(Path1,Temp1[j]))
    sample = sample.values[1::]
    sample = sample[:,1]
    annotation = pd.read_csv(os.path.join(Path2,Temp2[j]))
    annotation = annotation.values
    X_temp, y_temp = DoingSlidingWindow(sample, annotation, X_temp, y_temp, WindowLenth)
      
X_temp = np.array(X_temp)
y_temp = np.array(y_temp)


X_test = []
y_test = []
for j in range(60, len(Temp1)):
    sample = pd.read_csv(os.path.join(Path1,Temp1[j]))
    sample = sample.values[1::]
    sample = sample[:,1]
    annotation = pd.read_csv(os.path.join(Path2,Temp2[j]))
    annotation = annotation.values
    X_test, y_test = DoingSlidingWindow(sample, annotation, X_test, y_test, WindowLenth)
       
X_test = np.array(X_test)
y_test = np.array(y_test)


## Standardization 
#min_max_scaler = preprocessing.MinMaxScaler()
#Inputs = min_max_scaler.fit_transform(Inputs)

# Spliting dataset to train, validation and test
X_train, X_validation, y_train, y_validation = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)


X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_validation = X_validation.reshape(X_validation.shape[0], 1, X_validation.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_validation = to_categorical(y_validation)


# Train RNN(LSTM) Model
model, history = getModel(X_train, X_validation, y_train, y_validation)
print(model.summary())

# Prediction
Temp1 = getPrediction(model, X_train, X_test, y_train)
Temp1 = np.round(Temp1)


# Results evaluation
Out_test = [np.argmax(y, axis=None, out=None) for y in Temp1]
Temp2 = y_test
y_test = [np.argmax(y, axis=None, out=None) for y in Temp2]
result1 = precision_recall_fscore_support(y_test, Out_test)
print("Fscore: ", result1[2])
print("Precision: ", result1[0])
print("Recall: ", result1[1])
Accuracy = accuracy_score(y_test, Out_test)
print("Accuracy: ", Accuracy)


# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)



#import pdb; pdb.set_trace()
