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

# Keras libraries
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.applications import vgg16, inception_v3, resnet50, mobilenet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard

### ImageNet Large Scale Visual Recognition Challenge (ILSVRC) (1.2 M images, 1000 classes)
# Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')

# Load the inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')

# Load the ResNet50 model
resnet_model = resnet50.ResNet50(weights='imagenet')

# Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

### Load the image and convert its format to a 4-dimensional Tensor as an input of the form
# (batchsize, height, width, channels) requested by the Network.

filename = 'C:/Users/Theo/PycharmProjects/BMDATA/TP1/data/train/cat.1.jpg'
# Load an image in a PIL format
original = load_img(filename, target_size=(224, 224))
numpy_image = img_to_array(original)
# We add the extra dimension to the axis 0
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
plt.imshow(np.uint8(image_batch[0]))


### Now we can convert the image for the model and try and predict it
# preparing image
processed_img = vgg16.preprocess_input(image_batch.copy())
# get the predicted proba
predictions = vgg_model.predict(processed_img)


# decode and print prediction
def get_label(predicted_labels):
    prob = 0
    res = ''

    for count, prediction in enumerate(predicted_labels[0]):
        if prediction[2] > prob:
            prob = prediction[2]
            res = prediction[1]

    return res


label = vgg16.decode_predictions(predictions)
print(f"this is a {get_label(label)}")
