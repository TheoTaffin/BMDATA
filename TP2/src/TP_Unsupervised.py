from TP2.src.data_io import stl_load_stl10

import numpy as np

# Data visualization library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Native libraries
import os
import random
import gc

# Tensorflow libraries
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import TensorBoard


stl10_folder = "C:/Users/Theo/PycharmProjects/BMDATA/TP2/data"
root_folder = "C:/Users/Theo/PycharmProjects/"
X_train, y_train, X_test, y_test = stl_load_stl10(stl10_folder)


X_train_n = (X_train/255)
X_test_n = (X_test/255)

del X_train
del X_test
gc.collect()

mean_x_train = X_train_n.mean()
std_x_train = X_train_n.std()

X_train_c = (X_train_n - mean_x_train) / std_x_train
X_test_c = (X_test_n - mean_x_train) / std_x_train

del X_train_n
del X_test_n
gc.collect()

mini = X_train_c.min()
maxi = X_train_c.max() - mini

X_train_c_r_n = (X_train_c - mini) / maxi
X_test_c_r_n = (X_test_c - mini) / maxi

del X_train_c
del X_test_c
gc.collect()

conv1_weights = np.load(os.path.join(root_folder, "BMDATA/TP2/model/conv1_weights.npy"))
conv1_bias = np.load(os.path.join(root_folder, "BMDATA/TP2/model/conv1_bias.npy"))

conv2_weights = np.load(os.path.join(root_folder, "BMDATA/TP2/model/conv2_weights.npy"))
conv2_bias = np.load(os.path.join(root_folder, "BMDATA/TP2/model/conv2_bias.npy"))

tr_conv1_weights = np.load(os.path.join(root_folder, "BMDATA/TP2/model/tr_conv1_weights.npy"))
tr_conv1_bias = np.load(os.path.join(root_folder, "BMDATA/TP2/model/tr_conv1_bias.npy"))

tr_conv2_weights = np.load(os.path.join(root_folder, "BMDATA/TP2/model/tr_conv2_weights.npy"))
tr_conv2_bias = np.load(os.path.join(root_folder, "BMDATA/TP2/model/tr_conv2_bias.npy"))

conv1 = layers.Conv2D(filters=128, kernel_size=(5, 5), input_shape=(96, 96, 3),
                      activation='sigmoid', weights=[conv1_weights, conv1_bias])
maxpool1 = layers.MaxPooling2D(pool_size=(2, 2))

conv2 = layers.Conv2D(filters=128, kernel_size=(5, 5), input_shape=(96, 96, 3),
                      activation='sigmoid', weights=[conv2_weights, conv2_bias])
maxpool2 = layers.MaxPooling2D(pool_size=(2, 2))

tr_conv1 = layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                  weights=[tr_conv1_weights, tr_conv1_bias])
tr_conv2 = layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                  weights=[tr_conv2_weights, tr_conv2_bias])

model = models.Sequential([conv1, maxpool1, conv2, maxpool2, tr_conv1, tr_conv2])
model.compile()
model.summary()


### filter visualization
# retrieve weights from the first hidden layer
filters, biases = model.layers[0].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(3):
        # specify subplot and turn of axis
        ax = plt.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
# show the figure
plt.show()



