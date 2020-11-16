from TP2.src.data_io import stl_load_stl10

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
X_train, y_train, X_test, y_test = stl_load_stl10(stl10_folder)


