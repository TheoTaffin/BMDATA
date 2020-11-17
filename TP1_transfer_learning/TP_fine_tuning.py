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
    target_size=(224, 224), class_mode='categorical')
val_generator = val_datagen.flow_from_directory(
    "C:/Users/Theo/PycharmProjects/BMDATA/TP1_Exo/data/valid", batch_size=batch_size,
    target_size=(224, 224), class_mode='categorical')


conv_base = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

### We freeze the layer we don't want to train, we will just fine tune the last 4 layers
for layer in conv_base.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in conv_base.layers:
 print(layer, layer.trainable)

### Create model with our densely connected classifier
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(units=256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation="softmax"))

model.summary()

### compiling the model
model.compile(optimizer="adam",
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


n_train = train_generator.n
n_val = val_generator.n

## Making a TensorBoard callback object
NAME = "CNN_FineTuning"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

history = model.fit(train_generator,
                    steps_per_epoch=n_train // batch_size,
                    epochs=50, validation_data=val_generator,
                    validation_steps=n_val // batch_size,
                    callbacks=[tensorboard]
                    )

