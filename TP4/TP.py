import cv2
from imutils import face_utils
import imutils
import dlib

import numpy as np
import matplotlib.pyplot as plt
import math

import os

import tensorflow as tf
from tensorflow.keras import layers

root_folder = "C:/Users/Theo/PycharmProjects/BMDATA"

# ### Data importation
# faceimages_data_train = np.load(os.path.join(root_folder, "TP4/faceimages/images_train.npy"))
# faceimages_data_test = np.load(os.path.join(root_folder, "TP4/faceimages/images_test.npy"))
# faceimages_labels_train = np.load(os.path.join(root_folder, "TP4/faceimages/labels_train.npy"))
# faceimages_labels_train_labels_test = np.load(os.path.join(root_folder,
#                                                            "TP4/faceimages/labels_test.npy"))
#
# landmarks_x_train = np.load(os.path.join(root_folder, "TP4/landmarks/x_train.npy"))
# landmarks_x_test = np.load(os.path.join(root_folder, "TP4/landmarks/x_test.npy"))
# landmarks_y_train = np.load(os.path.join(root_folder, "TP4/landmarks/y_train.npy"))
# landmarks_y_test = np.load(os.path.join(root_folder, "TP4/landmarks/y_test.npy"))
#
#
# ### Model implementation
# model = tf.keras.Sequential()
# model.add(layers.Dense(units=512, activation='relu', input_dim=68*68))
# model.add(layers.Dense(units=512, activation='relu'))
# model.add(layers.Dense(units=512, activation='relu'))
# model.add(layers.Dense(units=512, activation='relu'))
# model.add(layers.Dense(units=512, activation='relu'))
# model.add(layers.Dense(units=7, activation='softmax'))
# model.compile()
# model.summary()
#
# ### Model fit
# model.compile(optimizer="adam",
#               loss=tf.keras.losses.CategoricalCrossentropy(),
#               metrics=['accuracy'])
#
# history = model.fit(x=landmarks_x_train, y=landmarks_y_train, epochs=40,
#                     validation_data=(landmarks_x_test, landmarks_y_test))
#
#
# ### Plot some graphs of accuracy and loss in both the train and validation set to see if we can
# # get some insight
# # plotting train and val curve, we can get those value form the history object
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# # Train an validation accuracy
# plt.plot(epochs, acc, "b", label='Training accuracy')
# plt.plot(epochs, val_acc, "r", label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.show()
#
# # Train an validation loss
# plt.plot(epochs, loss, "b", label='Training loss')
# plt.plot(epochs, val_loss, "r", label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.figure()
#
# plt.show()

model = tf.keras.models.load_model(os.path.join(root_folder, 'TP4/model_keras.h5'))
model.load_weights(os.path.join(root_folder, 'TP4/model_weights.h5'))

# -----------------------------
# opencv initialization
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(filename="C:/Users\Theo\PycharmProjects\BMDATA\TP4\sentiment_analysis.mp4")
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')

# initialize dlib's face detector and create a predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
          (168, 100, 168), (158, 163, 32), (163, 38, 32), (180, 42, 220), (153, 20, 48)]


def euclidean(a, b):
    dist = math.sqrt(math.pow((b[0] - a[0]), 2) + math.pow((b[1] - a[1]), 2))
    return dist


# calculates distances between all 68 elements
def euclidean_all(a):
    distances = []
    for i in range(0, len(a)):
        for j in range(0, len(a)):
            distances.append(euclidean(a[i], a[j]))

    return distances


def detect_parts(image):
    # resize the image, and convert it to grayscale
    image = imutils.resize(image, width=200, height=200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image

    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        print(image.shape)
        plt.show()
        distances = euclidean_all(shape)

        # visualize all facial landmarks with a transparent overlay
        output = face_utils.visualize_facial_landmarks(image, shape, colors=colors)
        print(image.shape)
        plt.imshow(output)
        plt.show()

    return distances


while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle to main image
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
        test = detect_parts(detected_face)

        distances = np.expand_dims(detect_parts(detected_face), axis=0)

        predictions = model.predict(distances)  # store probabilities of 6 expressions
        # find max indexed array ( 'Angry' , 'Disgust' , 'Fear' , 'Happy' , 'Neutral' , 'Sad' ,
        # 'Surprise')
        # for example
        emotion = emotions[np.argmax(predictions)]

        # write emotion text above rectangle
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,
                                                                                  255), 2)
        print('yo')
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit
        break
    # kill open cv things

cap.release()
cv2.destroyAllWindows()

