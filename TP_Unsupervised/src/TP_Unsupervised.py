from TP2.src.data_io import stl_load_stl10

# Data manipulation library
import numpy as np
import pandas as pd

# Data visualization library
import matplotlib.pyplot as plt

# Native libraries
import os
import gc

# sklearn
from sklearn.linear_model import LogisticRegression

# Tensorflow libraries
from tensorflow.keras import layers
from tensorflow.keras import models

# to avoid tf crashes
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# local variable
stl10_folder = "C:/Users/Theo/PycharmProjects/BMDATA/TP2/data"
root_folder = "C:/Users/Theo/PycharmProjects/"

# data extraction
X_train, y_train, X_test, y_test = stl_load_stl10(stl10_folder)


####################################### normalization #############################################
# X_train_n = (X_train/255)
# X_test_n = (X_test/255)
#
# del X_train
# del X_test
# gc.collect()
#
# mean_x_train = X_train_n.mean()
# std_x_train = X_train_n.std()
#
# X_train_c = (X_train_n - mean_x_train) / std_x_train
# X_test_c = (X_test_n - mean_x_train) / std_x_train
#
# del X_train_n
# del X_test_n
# gc.collect()
#
# mini = X_train_c.min()
# maxi = X_train_c.max() - mini
#
# X_train_c_r_n = (X_train_c - mini) / maxi
# X_test_c_r_n = (X_test_c - mini) / maxi
#
# del X_train_c
# del X_test_c
# gc.collect()
###################################################################################################

############################### Weight and biases loading #########################################
conv1_weights = np.load(os.path.join(root_folder, "BMDATA/TP2/model/conv1_weights.npy"))
conv1_bias = np.load(os.path.join(root_folder, "BMDATA/TP2/model/conv1_bias.npy"))

conv2_weights = np.load(os.path.join(root_folder, "BMDATA/TP2/model/conv2_weights.npy"))
conv2_bias = np.load(os.path.join(root_folder, "BMDATA/TP2/model/conv2_bias.npy"))

tr_conv1_weights = np.load(os.path.join(root_folder, "BMDATA/TP2/model/tr_conv1_weights.npy"))
tr_conv1_bias = np.load(os.path.join(root_folder, "BMDATA/TP2/model/tr_conv1_bias.npy"))

tr_conv2_weights = np.load(os.path.join(root_folder, "BMDATA/TP2/model/tr_conv2_weights.npy"))
tr_conv2_bias = np.load(os.path.join(root_folder, "BMDATA/TP2/model/tr_conv2_bias.npy"))
###################################################################################################

####################################### model building ############################################
conv1 = layers.Conv2D(filters=128, kernel_size=(5, 5), input_shape=(96, 96, 3),
                      activation='sigmoid', weights=[conv1_weights, conv1_bias])
maxpool1 = layers.MaxPooling2D(pool_size=(2, 2))

conv2 = layers.Conv2D(filters=128, kernel_size=(5, 5),
                      activation='sigmoid', weights=[conv2_weights, conv2_bias])
maxpool2 = layers.MaxPooling2D(pool_size=(2, 2))

tr_conv1 = layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                  weights=[tr_conv1_weights, tr_conv1_bias])
tr_conv2 = layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                  weights=[tr_conv2_weights, tr_conv2_bias])

model = models.Sequential([conv1, maxpool1, conv2, maxpool2, tr_conv1, tr_conv2])
model.summary()
###################################################################################################

###################################### filter visualization #######################################
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
###################################################################################################

######################### Visualizing a few images and their predictions ##########################
nb_image = 3
ix = 1
ax = plt.subplot(nb_image, 2, ix)
for i in range(nb_image):
    lst = [model.predict(np.expand_dims(X_train_c_r_n[i], axis=0))[0], X_train_c_r_n[i]]
    for j in range(2):
        ax = plt.subplot(nb_image, 2, ix)
        plt.imshow(lst[j])
        ix += 1

plt.show()
###################################################################################################

########################################### Sub model #############################################
model_2 = models.Sequential(model.layers[0:4])
model_2.summary()


local_features_x_train = model_2.predict(X_train_c_r_n)
local_features_x_test = model_2.predict(X_test_c_r_n)
###################################################################################################


######################################### Sum pooling #############################################
def vector_aggregation(descriptors):
    nb_descriptor = len(descriptors)
    sum_pooling = layers.AveragePooling2D(pool_size=(10, 10), input_shape=(21, 21, 128))
    model_3 = models.Sequential([sum_pooling])
    model_3.build()
    model_3.summary()
    res_sum = model_3.predict(descriptors)
    global_descriptor = np.reshape(res_sum, (nb_descriptor, 2 * 2 * 128))*10

    return global_descriptor


global_features_x_train = vector_aggregation(local_features_x_train)
global_features_x_test = vector_aggregation(local_features_x_test)
###################################################################################################

####################################### Classification ############################################
x = []
lst_train_score = []
lst_test_score = []

for max_iter in range(1000, 10000, 1000):
    model_4 = LogisticRegression(max_iter=max_iter)
    model_4.fit(global_features_x_train, y_train)

    train_score = model_4.score(global_features_x_train, y_train)
    test_score = model_4.score(global_features_x_test, y_test)

    print(f"training score after {max_iter} iter is {train_score}")
    print(f"test score after {max_iter} iter training is {test_score}")

    lst_train_score.append(train_score)
    lst_test_score.append(test_score)

score_df = pd.DataFrame.from_dict({'x': x,
                                   'train_score': lst_train_score,
                                   'test_score': lst_test_score}).set_index('x')
plt.plot(score_df)
plt.xlabel('iterations')
plt.ylabel('score')
plt.legend(['train_score', 'test_score'])
plt.show()
