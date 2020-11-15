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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard

train_dir = "data/train"
test_dir = "data/validation"

### Fetching train and validation sets
train_dogs = [f"{train_dir}/{file_name}" for file_name in os.listdir(train_dir)
              if 'dog' in file_name]
train_cats = [f"{train_dir}/{file_name}" for file_name in os.listdir(train_dir)
              if 'cat' in file_name]

test_imgs = [f"{test_dir}/{file_name}" for file_name in os.listdir(test_dir)]


train_size = 2000
train_set = train_dogs[:train_size] + train_cats[:train_size]
random.shuffle(train_set)
random.shuffle(test_imgs)


### Plotting img
for ima in train_set[0:3]:
    img = mpimg.imread(ima)
    plt.imshow(img)
    plt.show()


### A colored img is made up of 3 channels, i.e 3 arrays of red, green and blue pixel values

# A function to read and process the image to an acceptable format for our model:
def read_and_process_image(list_of_images):
    """
    :return: two arrays :
        X an array of resized images
        y an array of labels
    """

    # Lets declare our image dimension
    # we are using coloured img
    n_rows = 175
    n_cols = 175
    channels = 3  # change to 1 if you want to use grayscale image

    X = []
    y = []

    for image in list_of_images:
        im = cv2.imread(image, cv2.IMREAD_COLOR)
        im2 = cv2.resize(im, (n_rows, n_cols), interpolation=cv2.INTER_NEAREST)
        X.append(im2)

        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)

    return np.array(X), np.array(y)


X, y = read_and_process_image(train_set)


# Visualizing some of the processed img
plt.figure(figsize=(20, 10))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i])
plt.show()

print(X.shape)
print(y.shape)
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

# Setting batch size
batch_size = 16


# The following model is inspired from the vggnet architecture (details available at
# https://arxiv.org/pdf/1409.1556.pdf), in which we can see below that the filter size increases
# as we go down in layers (32 -> 64 -> 128 -> 512 and finally 1)

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu', input_shape=(175,
                                                                                        175, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(4, 4), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid bcs its a two classes problem

model.summary()

### compiling the model
model.compile(loss='binary_crossentropy', metrics=['acc'])
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


# number of steps per epochs : tells our model how manny images we want to process before making
# a gradient update to our loss function.  atotaala a 3200 images divided by batch size of 32
# will give us 100 steps
### Making a TensorBoard callback object
NAME = "CNN_FromSratch"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=64,
                    validation_data=val_generator,
                    validation_steps=len(X_val) // batch_size,
                    callbacks=[tensorboard]
                    )


model.save_weights('model_weights.h5')
model.save('model_keras.h5')


### Plot some graphs of accuracy and loss in both the train and validation set to see if we can
# get some insight
# plotting train and val curve, we can get those value form the history object
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Train an validation accuracy
plt.plot(epochs, acc, "b", label='Training accuracy')
plt.plot(epochs, val_acc, "r", label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()

# Train an validation loss
plt.plot(epochs, loss, "b", label='Training loss')
plt.plot(epochs, val_loss, "r", label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.show()

### We're not overfitting since train and validation are pretty close. Accuracy keeps increasing
# as epochs are as well, giving us the intuition that increasing epochs will increase accuracy

# To test the model on some images from the test set :
X_test, y_test = read_and_process_image(test_imgs[:10])
test_datagen = ImageDataGenerator(rescale=1./255)

text_labels = []
plt.figure(figsize=(30, 20))

for count, batch in enumerate(test_datagen.flow(X_test, batch_size=1)):
    if count >= 10:
        break
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('dog')
    else:
        text_labels.append('cat')

    plt.subplot(2, 5, count + 1)
    plt.title(f"this is a {text_labels[count]}")
    imgplot = plt.imshow(batch[0])
    print(count)


plt.show()

