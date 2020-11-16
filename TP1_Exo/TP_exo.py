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
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard

wdir = "C:/Users/Theo/PycharmProjects/BMDATA/TP1_Exo"
train_dir = os.path.join(wdir, "data/train")
valid_dir = os.path.join(wdir, "data/valid")
test_dir = os.path.join(wdir, "data/test")


def extract(directory_path):
    dir_dict = {}
    tmp_list = []
    for directory in os.listdir(directory_path):
        if not directory.endswith('.gitkeep'):
            sub_dir = os.path.join(directory_path, directory)
            for file in os.listdir(sub_dir):
                tmp_list.append(f"{sub_dir}/{file}")

            dir_dict[directory] = tmp_list.copy()
            tmp_list.clear()

    return dir_dict


train_dict = extract(train_dir)
valid_dict = extract(valid_dir)
test_dict = extract(test_dir)

classes = {}
for count, class_type in enumerate(train_dict.keys()):
    classes[class_type] = count


# A function to read and process the image to an acceptable format for our model:
def read_and_process_image(dict_of_dirs, unknown=False):
    """
    :return: two arrays :
        X an array of resized images
        y an array of labels
    """

    # Lets declare our image dimension
    # we are using coloured img
    n_rows = 200
    n_cols = 200
    channels = 3  # change to 1 if you want to use grayscale image

    X = []
    y = []

    for directory_name in dict_of_dirs:
        for image in dict_of_dirs[directory_name]:
            im = cv2.imread(image, cv2.IMREAD_COLOR)
            im2 = cv2.resize(im, (n_rows, n_cols), interpolation=cv2.INTER_NEAREST)
            X.append(im2)
            if not unknown:
                y.append(classes[directory_name])

    return np.array(X), np.array(y)


X, y = read_and_process_image(train_dict)

tmp = list(zip(X, y))
random.shuffle(tmp)
X, y = zip(*tmp)

X = np.array(X)
y = np.array(y)

# Visualizing some of the processed img
plt.figure(figsize=(20, 10))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i])
    plt.title(y[i])

plt.show()


### Split the data into 80/20 sets (train and test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)

# Clear memory
del X
del y
gc.collect()

#get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

# Setting batch size
batch_size = 32

# The following model is inspired from the vggnet architecture (details available at
# https://arxiv.org/pdf/1409.1556.pdf), in which we can see below that the filter size increases
# as we go down in layers (32 -> 64 -> 128 -> 512 and finally 1)

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(200,
                                                                                        200, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(len(classes)))

model.summary()

### compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
optimizer = optimizers.RMSprop(lr=1e-3)


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
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


### Making a TensorBoard callback object
NAME = "CNN_FromSratch2"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=50, validation_data=val_generator,
                    validation_steps=len(X_val) // batch_size,
                    callbacks=[tensorboard]
                    )


# To test the model on some images from the test set :
X_test, y_test = read_and_process_image(test_dict, unknown=True)
test_datagen = ImageDataGenerator(rescale=1./255)

text_labels = []
plt.figure(figsize=(30, 20))

for count, batch in enumerate(X_test):
    if count >= 10:
        break
    pred = model.predict(np.expand_dims(batch, axis=0))
    text_labels.append(np.argmax(pred))

    plt.subplot(2, 5, count + 1)
    plt.title(f"this is a {text_labels[count]}")
    imgplot = plt.imshow(batch)

plt.show()
