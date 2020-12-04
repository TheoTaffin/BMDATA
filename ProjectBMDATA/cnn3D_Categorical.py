import os
import patoolib


import os
import cv2
import gc
import random

# Data visualization library
import matplotlib.pyplot as plt


# ML/DL
from sklearn.model_selection import train_test_split

# Keras libraries
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras import models


# Data
import numpy as np
import pandas as pd
import re


data_path = "A:/BMW/DAD"
subject_list = ["Tester1", "Tester5", "Tester8", "Tester6"]
selected_mode = "front_depth"
rate = 2
segment_size = 12
n_rows = 125
n_cols = 125


def natural_keys(text):
    '''
    Thanks to Jeff Atwood
    Indoor enthusiast. Co-founder of Stack Overflow and Discourse.
    '''

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    text = sorted(text, key=alphanum_key)
    return text


def get_class(path):
    classes = {}
    i = 0
    for count, class_type in enumerate(os.listdir(path)):
        if "normal" not in class_type:
            classes[class_type] = i
            i += 1
    return classes


def extract(src_dir, subject_list, selected_mode, rate, class_names):

    dir_dict = {}

    for class_name in class_names.keys():
        dir_dict[class_name] = []

    for subject in subject_list:
        subject_path = os.path.join(src_dir, subject)

        for directory in os.listdir(subject_path):
            if "normal" not in directory:

                # sorting the images to ensure that they are correctly following one another
                # throughout selected time frame
                tmp = []
                for count, img in enumerate(natural_keys(os.listdir(os.path.join(subject_path,
                                                                                 directory,
                                                                                 selected_mode)))):

                    image_test_path = os.path.join(subject_path, directory, selected_mode, img)
                    if count % rate == 0:
                        tmp.append(image_test_path)
                dir_dict[directory].extend(tmp.copy())
                tmp.clear()

    return dir_dict


### A colored img is made up of 3 channels, i.e 3 arrays of red, green and blue pixel values
# A function to read and process the image to an acceptable format for our model:
def read_and_process_image(dict_of_dirs, n_rows, n_cols, segment_size, classes):
    """
    :return: two arrays :
        X an array of resized images
    """

    # Lets declare our image dimension
    # we are using coloured img

    X = []
    y = []
    tmp = []

    for count, directory_name in enumerate(dict_of_dirs):
        print(count)
        tmp.clear()
        i = 0
        for image in dict_of_dirs[directory_name]:  # Reading and resizing images

            im = cv2.imread(image, cv2.IMREAD_COLOR)
            im2 = cv2.resize(im, (n_rows, n_cols), interpolation=cv2.INTER_NEAREST)
            im3 = cv2.normalize(im2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_32F)
            im4 = sp_noise(im3, 0.005)
            tmp.append(np.asarray(im4))
            i += 1
            if i == segment_size:
                X.append(np.asarray(tmp.copy()))
                i = 0
                y.append(classes[directory_name])
                tmp.clear()

    return np.stack(X), y


def split_and_shuffle(segment_array, y):
    dataset = np.array(list(zip(segment_array, y)))
    np.random.shuffle(dataset)
    segment_array, y = zip(*dataset)
    segment_array = np.stack(segment_array)
    y = pd.get_dummies(y)
    X_train, X_val, y_train, y_val = train_test_split(segment_array, y, test_size=0.3, random_state=2)

    print("Shape of train images is:", X_train.shape)
    print("Shape of validation images is:", X_val.shape)
    print("Shape of labels is:", y_train.shape)
    print("Shape of labels is:", y_val.shape)

    return X_train, X_val, y_train, y_val


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([1, 1, 1], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(image.shape[:2])
    image[probs < (prob / 2)] = black
    image[probs > 1 - (prob / 2)] = white
    return image


#### Loading class name
class_names = get_class("A:/BMW/DAD/Tester1")

#### Loading train and validation images
train_img = extract(data_path, subject_list, selected_mode, rate, class_names)
segment_array, y = read_and_process_image(train_img, n_rows, n_cols, segment_size, class_names)
X_train, X_val, y_train, y_val = split_and_shuffle(segment_array, y)

#### Loading test images
test_img = extract(data_path, ["Tester18"], selected_mode, rate, class_names)
X_test, y_test = read_and_process_image(test_img, n_rows, n_cols, segment_size, class_names)

#### Shuffling or test images
dataset = np.array(list(zip(X_test, y_test)))
np.random.shuffle(dataset)
segment_array_test, y_test = zip(*dataset)
segment_array_test = np.stack(segment_array_test)
y_test = pd.get_dummies(y_test)


#### Setting batch size
batch_size = 16

#### Adding regulairzer to prevent overfitting
regularizer_l2 = l2(l=0.05)

model = models.Sequential()
model.add(layers.Conv3D(filters=16, kernel_size=(3, 16, 16), kernel_regularizer=regularizer_l2,
                        bias_regularizer=regularizer_l2,
                        activation='relu',
                        input_shape=(segment_size, n_rows, n_cols, 3)))
model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
model.add(layers.Dropout(0.4))
model.add(layers.Conv3D(filters=32, kernel_size=(3, 32, 32), kernel_regularizer=regularizer_l2,
                        bias_regularizer=regularizer_l2,
                        activation='relu'))
model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))

model.add(layers.Dense(units=32, activation='relu'))
model.add(layers.Dense(len(class_names), activation='softmax'))  # Softmax bcs its multiple


model.summary()

### compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=4)

model.evaluate(X_val, y_val, batch_size=3)


#### prompting a few predictions
predictions = model.predict(X_val[0:3])
predictions = list(map(np.argmax, predictions))
print([list(y_val[count]).index(1) for count in range(10)], [pred for pred in predictions])


for count, ima in enumerate(X_val[0:3]):
    print(y_val.iloc[count].idxmax(), predictions[count])
    ground_truth = [key for (key, value) in class_names.items() if value == y_val.iloc[count].idxmax()]
    prediction = [key for (key, value) in class_names.items() if value == predictions[count]]

    for img in ima[0:1]:
        plt.imshow(img)
        plt.title(f"g_t: {ground_truth}, pred: {prediction}")
        plt.show()
