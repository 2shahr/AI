from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import preprocess_input
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPool2D, GlobalAvgPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import numpy as np

batch_size = 16
input_shape = (100, 100)
n_classes = 2
ens_size = 20
n_epochs = 20


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, 
    vertical_flip=True, 
    horizontal_flip=True,
    width_shift_range=20,
    height_shift_range=20,
    rotation_range=10,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=input_shape,
    batch_size=batch_size,
    class_mode="binary",
    subset='training'
)

test_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=input_shape,
    batch_size=batch_size,
    class_mode="binary",
    subset='validation'
)

def get_model():
    model = DenseNet121(include_top=False, input_shape=(input_shape[0], input_shape[1], 3), pooling='max')
    global_max = (model.layers[-1].output)
    class1 = Dense(10, activation='relu')(global_max)
    output = Dense(2, activation='softmax')(class1)
    model = Model(inputs=model.inputs, outputs=output)
    return model

models = []
for i in range(ens_size):
    model = get_model()
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=n_epochs)
    models.append(model)

prd = models[0].predict(test_generator)
acc = np.zeros((ens_size,))
prd_label = np.argmax(prd, axis=1)
acc[0] = np.mean(prd_label == test_generator.labels)
for i in range(1, ens_size):
    prd_ = models[i].predict(test_generator)
    prd += prd_
    prd_label = np.argmax(prd_, axis=1)
    acc[i] = np.mean(prd_label == test_generator.labels)

prd_label = np.argmax(prd, axis=1)
acc_all = np.mean(prd_label == test_generator.labels)

print(f"ensemble accuracy = {acc_all}")
for i in range(ens_size):
    print(f"accuracy of CNN{i}: {acc[i]}")



