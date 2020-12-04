import os
import patoolib


import os
import cv2
import gc
import random

# Data visualization library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ML/DL
from sklearn.model_selection import train_test_split

# Keras libraries
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import vgg16
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data
import numpy as np
import pandas as pd
import re


data_path = "A:/BMW/DAD"
subject_list = ["Tester1", "Tester2", "Tester3", "Tester4", "Tester5", "Tester6", "Tester8",
                "Tester11", "Tester15", "Tester21", "Tester22"]
selected_mode = "front_depth"
rate = 12
n_rows = 150
n_cols = 150


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
def read_and_process_image(dict_of_dirs, n_rows, n_cols, classes):
    """
    :return: two arrays :
        X an array of resized images
    """


    X = []
    y = []
    tmp = []

    for count, directory_name in enumerate(dict_of_dirs):
        print(count)
        tmp.clear()

        for image in dict_of_dirs[directory_name]:  # Reading and resizing images

            im = cv2.imread(image, cv2.IMREAD_COLOR)
            im2 = cv2.resize(im, (n_rows, n_cols), interpolation=cv2.INTER_NEAREST)

            # normalizing
            im3 = cv2.normalize(im2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
            # Adding salt and pepper noise
            im4 = sp_noise(im3, 0.005)

            X.append(np.asarray(im4))
            y.append(classes[directory_name])

    return np.stack(X), y


#### Function that splits the dataset into validation and training sets and shuffles it
def split_and_shuffle(X, y):

    dataset = np.array(list(zip(X, y)))
    np.random.shuffle(dataset)
    X, y = zip(*dataset)
    X = np.stack(X)
    y = pd.get_dummies(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2)

    print("Shape of train images is:", X_train.shape)
    print("Shape of validation images is:", X_val.shape)
    print("Shape of labels is:", y_train.shape)
    print("Shape of labels is:", y_val.shape)

    return X_train, X_val, y_train, y_val


#### function to add salt and pepper noise to an image, again another attempt to reduce overfitting
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
X, y = read_and_process_image(train_img, n_rows, n_cols, class_names)
X_train, X_val, y_train, y_val = split_and_shuffle(X, y)


#### Loading test images
test_img = extract(data_path,
                   ["Tester18", "Tester10", "Tester16", "Tester23"],
                   selected_mode,
                   rate,
                   class_names)
X_test, y_test = read_and_process_image(test_img, n_rows, n_cols, class_names)

#### Shuffling or test images
dataset = np.array(list(zip(X_test, y_test)))
np.random.shuffle(dataset)
X_test, y_test = zip(*dataset)
X_test = np.stack(X_test)
y_test = pd.get_dummies(y_test)


#### Visualizing a few images
for keys in class_names.keys():
    for ima in train_img[keys][0:1]:
        img = mpimg.imread(ima)
        plt.imshow(img)
        plt.title(ima)
        plt.show()

#################################################################################################
# # Comment previous code and use this to test the model locally
# dataset = np.load("A:/tmp_array/dataset_12_150_150.npz")
# X_train = dataset['arr_0']
# X_val = dataset['arr_1']
# y_train = dataset['arr_2']
# y_val = dataset['arr_3']
# X_test = dataset['arr_4']
# y_test = dataset['arr_5']
#################################################################################################

# Setting batch size
batch_size = 32

### Creating image generators
train_datagen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2)
val_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

regularizer_l2 = l2(l=0.03)

model = models.Sequential()
# model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding="valid",
#                         kernel_regularizer=regularizer_l2,
#                         bias_regularizer=regularizer_l2,
#                         activation='relu',
#                         input_shape=(n_rows, n_cols, 3)))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.3))
# model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding="valid",
#                         kernel_regularizer=regularizer_l2,
#                         bias_regularizer=regularizer_l2,
#                         activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.3))
# model.add(layers.Conv2D(filters=128, kernel_size=(5, 5),
#                         kernel_regularizer=regularizer_l2,
#                         bias_regularizer=regularizer_l2,
#                         activation='relu'))
conv_base = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(n_rows, n_cols, 3))
for layer in conv_base.layers[:-2]:
    layer.trainable = False
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(units=64, activation='relu', kernel_regularizer=regularizer_l2))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(len(class_names), activation='softmax'))  # Softmax bcs its multiple


model.summary()

#### compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


#### Fitting the model
history = model.fit(train_generator, validation_data=val_generator, batch_size=batch_size,
                    epochs=4, callbacks=[EarlyStopping(monitor="val_acc", patience=0)])


#### testing the model
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(X_test, y_test)
model.evaluate(test_generator, batch_size=batch_size)


#### prompting a few predictions
predictions = model.predict(X_test[0:10])
predictions = list(map(np.argmax, predictions))
print([y_test.iloc[count].idxmax() for count in range(10)], [pred for pred in predictions])


for count, ima in enumerate(X_test[0:10]):
    ground_truth = [key for (key, value) in class_names.items() if value == y_test.iloc[
        count].idxmax()]
    prediction = [key for (key, value) in class_names.items() if value == predictions[count]]
    plt.imshow(ima)
    plt.title(f"g_t: {ground_truth}, pred: {prediction}")
    plt.show()
