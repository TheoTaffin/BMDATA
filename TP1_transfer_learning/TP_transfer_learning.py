# Data manipulation libraries
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Data visualization library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Native libraries
import os
import random
import gc

# Tensorflow libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.applications import vgg16, inception_v3, resnet50, mobilenet 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard


## weights : to specify which weight checkpoint to initialize the model from
## include_top : refers to whether or not include densely connected classifier on top of the 
# network. By default, this would correspond to the 1000 classes from imageNet. Since we intend 
# to use our own classifier (with only cats and dogs(edit : apparently we will use the second
# dataset)), we don't need it
## input_shape ! the shape of the image tensors that we will feed to the network. This is purely
# optional : if we don't, then the network will be able to process inputs of any size. In our 
# case, the image size is 224
conv_base = vgg16.VGG16(weights='imagenet',  include_top=False, input_shape=(224, 224, 3))

conv_base.summary()

# Setting batch size
batch_size = 32

### preparing data (again)
# rescale 1./255 normalizes the pixel value (m=0, std=1), helps with neural network efficiency
# Second set of option apply some transformation (randomly) to the dataset, this will augment it
# and improve generalization
train_datagen = ImageDataGenerator(rescale=1./255,  # Scale the image between 0 and 1
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)  # we do not augment validation data,
                                                  # we only rescale

### Creating image generators
train_generator = train_datagen.flow_from_directory(
    "C:/Users/Theo/PycharmProjects/BMDATA/TP1_Exo/data/train", batch_size=batch_size,
    target_size=(224, 224))
val_generator = val_datagen.flow_from_directory(
    "C:/Users/Theo/PycharmProjects/BMDATA/TP1_Exo/data/valid", batch_size=batch_size,
    target_size=(224, 224))

### 1rst technique of feature extraction
nTrain = 608
nVal = 160

# The shape's value are directly connected to the last layer's shape of conv_base model
train_features = np.zeros(shape=(nTrain, 7, 7, 512))
val_features = np.zeros(shape=(nVal, 7, 7, 512))
# This is 8 bcs we are apparently using the second dataset and not the first one. Thanks for the
# random change Wanny
train_labels = np.zeros(shape=(nTrain, 8))
val_labels = np.zeros(shape=(nVal, 8))


for count, (inputs_batch, labels_batch) in enumerate(train_generator):
    if count * batch_size >= nTrain:
        break

    features_batch = conv_base.predict(inputs_batch)
    train_features[count * batch_size: (count + 1) * batch_size] = features_batch
    train_labels[count * batch_size: (count + 1) * batch_size] = labels_batch


for count, (inputs_batch, labels_batch) in enumerate(val_generator):
    if count * batch_size >= nVal:
        break

    features_batch = conv_base.predict(inputs_batch)
    val_features[count * batch_size: (count + 1) * batch_size] = features_batch
    val_labels[count * batch_size: (count + 1) * batch_size] = labels_batch


train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))
val_features = np.reshape(val_features, (nVal, 7 * 7 * 512))

### Create model with our densely connected classifier
model = models.Sequential()
model.add(layers.Dense(units=256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation="softmax"))

### compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

### Making a TensorBoard callback object
NAME = "CNN_FromVGG16"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

history = model.fit(x=train_features, y=train_labels,
                    steps_per_epoch=len(train_features) // batch_size,
                    epochs=50, validation_data=(val_features, val_labels),
                    validation_steps=len(val_features) // batch_size,
                    callbacks=[tensorboard]
                    )

# To test the model on some images from the test set :

test_datagen = ImageDataGenerator(rescale=1./255)

text_labels = []
plt.figure(figsize=(30, 20))

for count, batch in enumerate(test_datagen.flow_from_directory(
        "C:/Users/Theo/PycharmProjects/BMDATA/TP1_Exo/data/test", target_size=(224, 224),
        batch_size=1)):

    if count >= 8:
        break
    features_batch = conv_base.predict(batch[0])
    reshape = np.reshape(features_batch, (1, 7 * 7 * 512))
    pred = model.predict(reshape)
    text_labels.append(np.argmax(pred))

    plt.subplot(2, 5, count + 1)
    plt.title(f"this is a {text_labels[count]}")
    imgplot = plt.imshow(batch[0][0])

plt.show()
