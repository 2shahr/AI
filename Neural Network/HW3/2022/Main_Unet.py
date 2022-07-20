
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import pandas as pd
from keras_unet.models import custom_unet
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance
#from keras.callbacks import ModelCheckpoint
from keras_unet.utils import plot_segm_history
#from keras_unet.utils import plot_imgs
#from keras import backend as K
from sklearn.cluster import KMeans
#from keras.utils import to_categorical


########### Functions ###########

def getFileList(directory = os.path.dirname(os.path.realpath(__file__))):
    list = os.listdir(directory)
    return list


def getModel(x_train, train, x_validation, validation):

    # building a Unet model
    input_shape = x_train[0].shape
    model = custom_unet(
        input_shape,
        num_classes=10,
        filters=32,
        use_batch_norm=True,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        num_layers=4
    )
    
    model.compile(
        optimizer=Adam(), 
#        optimizer=SGD(lr=0.01, momentum=0.99),
#        loss='binary_crossentropy',
        loss=jaccard_distance,
        metrics=[iou, iou_thresholded]
    )    

    history = model.fit(
        x_train, train,
        steps_per_epoch=30,
        epochs=5,
        
        validation_data=(x_validation, validation)
    )    
    
    
    return model, history
    


def getPrediction(model, x_test):
    predictions = model.predict(x_test)
    return predictions


########### Main code ###########

# Load dataset    
Path = 'DatasetTest'
Temp1 = getFileList('Dataset')

# Train data
X_train = []
y_train = []
i = 1
temp = getFileList(os.path.join(Path,Temp1[i]))
for j in range(len(temp)):
   Temp2 = cv2.imread(os.path.join(Path,Temp1[i], temp[j]))
   dim = (64, 64)
   Temp3 = Temp2[:,0:256,:]
   Temp4 = Temp2[:,256:512,:]
   X_train.append(cv2.resize(Temp3, dim))
   y_train.append(cv2.resize(Temp4, dim))   
       
       
X = np.array(X_train)
y = np.array(y_train)


# Validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=0)


# Test data
X_test = []
y_test = []
i = 0
temp = getFileList(os.path.join(Path,Temp1[i]))
for j in range(len(temp)):
   Temp2 = cv2.imread(os.path.join(Path,Temp1[i], temp[j]))
   dim = (64, 64)
   Temp3 = Temp2[:,0:256,:]
   Temp4 = Temp2[:,256:512,:]
   X_test.append(cv2.resize(Temp3, dim))
   y_test.append(cv2.resize(Temp4, dim))
       
       
X_test = np.array(X_test)
y_test = np.array(y_test)


AllData = np.concatenate((y_train, y_val, y_test), axis=0)
unique, counts = np.unique(AllData.reshape(-1, AllData.shape[-1]), axis=0, return_counts=True)
print(counts.shape)


## Clustering
kmeans = KMeans(init="random",
                n_clusters=10,
                max_iter=300,
                random_state=42)
kmeans.fit(unique)
Cluster_centers = kmeans.cluster_centers_
Labels = kmeans.labels_


New_y_train = y_train.copy()
New_y_train_Class = np.zeros((y_train.shape[0], y_train.shape[1], y_train.shape[1], 10))
for i in range(y_train.shape[0]):
    for j in range(y_train.shape[1]):
        for k in range(y_train.shape[2]):
            label = kmeans.predict(y_train[i,j,k,:].reshape(1,-1))
            New_y_train_Class[i,j,k,label[0]] = 1
            New_y_train[i,j,k,:] = Cluster_centers[label[0]] 
          
New_y_val = y_val.copy()
New_y_val_Class = np.zeros((y_val.shape[0], y_val.shape[1], y_val.shape[1], 10))
for i in range(y_val.shape[0]):
    for j in range(y_val.shape[1]):
        for k in range(y_val.shape[2]):
            label = kmeans.predict(y_val[i,j,k,:].reshape(1,-1))
            New_y_val_Class[i,j,k,label[0]] = 1
            New_y_val[i,j,k,:] = Cluster_centers[label[0]]            

New_y_test = y_test.copy()
New_y_test_Class = np.zeros((y_test.shape[0], y_test.shape[1], y_test.shape[1], 10))
for i in range(y_test.shape[0]):
    for j in range(y_test.shape[1]):
        for k in range(y_test.shape[2]):
            label = kmeans.predict(y_test[i,j,k,:].reshape(1,-1))
            New_y_test_Class[i,j,k,label[0]] = 1
            New_y_test[i,j,k,:] = Cluster_centers[label[0]]

# Show clustering results            
f, axarr = plt.subplots(3, 2)
axarr[0,0].imshow(y_test[0])
axarr[0,0].set_title("Before clustering")
axarr[0,1].imshow(New_y_test[0])
axarr[0,1].set_title("After clustering")
axarr[1,0].imshow(y_test[1])
axarr[1,1].imshow(New_y_test[1])
axarr[2,0].imshow(y_test[2])
axarr[2,1].imshow(New_y_test[2])
plt.show()        


# Normalization
X_train = np.asarray(X_train, dtype=np.float32)/255
X_val = np.asarray(X_val, dtype=np.float32)/255
X_test = np.asarray(X_test, dtype=np.float32)/255
y_train_f = New_y_train_Class
y_val_f = New_y_val_Class
y_test_f = New_y_test_Class



# Train  Model
model, history = getModel(X_train, y_train_f, X_val, y_val_f)
print(model.summary())


# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


# Plot training history
plot_segm_history(history)
plt.show()


# Prediction
y_pred = getPrediction(model, X_test)
New_y_pred = np.zeros((y_pred.shape[0], y_pred.shape[1], y_pred.shape[1], 3))
for i in range(y_pred.shape[0]):
    for j in range(y_pred.shape[1]):
        for k in range(y_pred.shape[2]):
            label = np.argmax(y_pred[i,j,k,:])
            New_y_pred[i,j,k,:] = Cluster_centers[label]
New_y_pred = np.asarray(New_y_pred, dtype=np.float32)/255

            
# Plot original, ground truth, and pred
f, axarr = plt.subplots(4, 3)
axarr[0,0].imshow(X_test[0])
axarr[0,0].set_title("Original")
axarr[0,1].imshow(New_y_test[0])
axarr[0,1].set_title("Ground truth")
axarr[0,2].imshow(New_y_pred[0])
axarr[0,2].set_title("Prediction")
axarr[1,0].imshow(X_test[1])
axarr[1,1].imshow(New_y_test[1])
axarr[1,2].imshow(New_y_pred[1])
axarr[2,0].imshow(X_test[2])
axarr[2,1].imshow(New_y_test[2])
axarr[2,2].imshow(New_y_pred[2])
axarr[3,0].imshow(X_test[3])
axarr[3,1].imshow(New_y_test[3])
axarr[3,2].imshow(New_y_pred[3])


#import pdb; pdb.set_trace()