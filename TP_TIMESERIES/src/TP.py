import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

root_dir = "C:/Users/Theo/PycharmProjects/BMDATA"


def import_data(file_path):
    column_names = ['user_id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv(file_path, names=column_names, na_values=';')
    return df


def normalize_features(df, cols):

    for col in cols:
        tmp = df[col].values
        min_max_scaler = preprocessing.MinMaxScaler()
        tmp_scaled = min_max_scaler.fit_transform(tmp.reshape(-1, 1))
        df[col] = pd.DataFrame(tmp_scaled)

    return df


### Visualization of change in activity over time
def plot_axis(axis, x, y, title):
    axis.plot(x, y)
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    axis.set_xlim([min(x), max(x)])
    axis.grid(True)


def plot_activity(activity,data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()


# In order to feed the network with such temporal dependencies a sliding time window is used to
# extract separate data segments. The window width and step size can be both adjusted and
# optimised for better accuracy. Each time step is associated with an activity label for each
# segment the most frequently appearing label is chosen. Here the time segment or window width
# is chosen to be 90.

def windows(df, size):
    count = 0
    progression = 0
    len_data = df.count()
    while count < len_data:
        tmp = (count * 100) // len_data
        yield int(count), int(count + size)
        count += size//2
        if tmp > progression:
            progression = tmp
            print(f"progression: {progression}")


def segment_signal(data, window_size=90):

    segments = np.empty((0, window_size, 3))
    labels = np.empty(0)

    for start, end in windows(data['timestamp'], window_size):
        x = data['x-axis'][start:end]
        y = data['y-axis'][start:end]
        z = data['z-axis'][start:end]

        if len(data['timestamp'][start:end]) == window_size:
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data['activity'][start:end])[0][0])

    return segments, labels


# data = import_data(os.path.join(root_dir, "TP_TIMESERIES/WISDM/WISDM_raw.txt"))
# data['z-axis'] = data['z-axis'].str.replace(";", "")
# data = normalize_features(data, ['x-axis', 'y-axis', 'z-axis'])
# data['z-axis'].fillna(0, inplace=True)
#
# for activity in np.unique(data['activity']):
#     subset = data[data['activity'] == activity][:180]
#     plot_activity(activity, subset)
#
# segments, labels = segment_signal(data)
# labels = pd.get_dummies(labels)
#
# np.save('segments', segments)
# np.save('label', labels)
# train_size = int((0.8*len(segments)))

# x_train = segments[0:train_size]
# y_train = labels[0:train_size]
# x_test = segments[train_size:len(segments)]
# y_test = labels[train_size:len(segments)]


# to avoid tf crashes
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def test_model(epochs, batch_size, title, lst_layers, x_train_t, x_val_t,  y_train_t, y_val_t):

    model_t = Sequential(lst_layers)
    model_t.compile(optimizer='adam',
                    loss="categorical_crossentropy",
                    metrics=['categorical_accuracy'])

    model_t.summary()
    w_save = model_t.get_weights()
    history = model_t.fit(x_train_t, y_train_t, batch_size=batch_size,
                          epochs=epochs, validation_data=(x_val_t, y_val_t))

    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Train an validation accuracy
    plt.plot(epochs, acc, "b", label='Training accuracy')
    plt.plot(epochs, val_acc, "r", label='Validation accuracy')
    plt.title(f'Training and validation accuracy {title}{batch_size}')
    plt.legend()
    plt.figure()
    plt.show()

    # Train an validation loss
    plt.plot(epochs, loss, "b", label='Training loss')
    plt.plot(epochs, val_loss, "r", label='Validation loss')
    plt.title(f'Training and validation loss {title}{batch_size}')
    plt.legend()
    plt.figure()

    plt.show()
    model_t.set_weights(w_save)
    return acc, val_acc, loss, val_loss


def grid_search(epochs_list, batch_size_list, cls, title, data=None):

    grid = {}

    for epoch in epochs_list:
        tmp_batch_size = {}
        for batch_s in batch_size_list:

            h1 = test_model(epoch, batch_s, title, cls, x_train_cn1, x_valn_cn1, y_train_cn1,
                            y_valn_cn1)
            tmp_batch_size[batch_s] = {'acc': h1[0], 'val_acc': h1[1], 'loss': h1[2], 'val_loss': h1[3]}

        grid[epoch] = tmp_batch_size

    return grid


epochs = 16
batch_size = 16


# # ### Model creation
m1_conv2d_1 = layers.Conv2D(filters=128, kernel_size=(2, 2), activation='relu',
                            input_shape=(90, 3, 1))
m1_max_pooling2d_1 = layers.MaxPool2D(pool_size=(2, 2))
m1_dropout_1 = layers.Dropout(0.5)
m1_flatten_1 = layers.Flatten()
m1_dense_1 = layers.Dense(units=128, activation='relu')
m1_dense_2 = layers.Dense(units=128, activation='relu')
m1_dense_3 = layers.Dense(units=6, activation='softmax')

m1_layers = [m1_conv2d_1, m1_max_pooling2d_1, m1_dropout_1, m1_flatten_1, m1_dense_1, m1_dense_2, m1_dense_3]


### Model creation 2
m2_lstm_1 = layers.LSTM(units=128, input_shape=(90, 3), return_sequences=True)
m2_dropout_1 = layers.Dropout(0.5)
m2_batch_normalization_1 = layers.BatchNormalization()
m2_lstm_2 = layers.LSTM(units=128)
m2_dropout_2 = layers.Dropout(0.5)
m2_batch_normalization_2 = layers.BatchNormalization()
m2_dense_1 = layers.Dense(units=32, activation='relu')
m2_dropout_3 = layers.Dropout(0.5)
m2_dense_2 = layers.Dense(units=6, activation='softmax')

m2_layers = [m2_lstm_1, m2_dropout_1, m2_batch_normalization_1, m2_lstm_2, m2_dropout_2,
                    m2_batch_normalization_2,  m2_dense_1, m2_dropout_3, m2_dense_2]


### Model creation 3
m3_conv1d_1 = layers.Conv1D(filters=100, kernel_size=9, input_shape=(90, 3))
m3_conv1d_2 = layers.Conv1D(filters=100, kernel_size=9)
m3_max_pooling1d_1 = layers.MaxPool1D(pool_size=3)
m3_conv1d_3 = layers.Conv1D(filters=160, kernel_size=9)
m3_conv1d_4 = layers.Conv1D(filters=160, kernel_size=9)
m3_global_average_pooling1d_1 = layers.GlobalAveragePooling1D()
m3_dropout_1 = layers.Dropout(0.5)
m3_dense_2 = layers.Dense(units=6, activation='softmax')

m3_layers = [m3_conv1d_1, m3_conv1d_2, m3_max_pooling1d_1, m3_conv1d_3, m3_conv1d_4,
            m3_global_average_pooling1d_1, m3_dropout_1, m3_dense_2]


segments, labels = np.load(os.path.join(root_dir, 'TP_TIMESERIES/src/segments.npy')), \
                   np.load(os.path.join(root_dir, 'TP_TIMESERIES/src/labels.npy'))


segments_cnn1 = segments.reshape((segments.shape[0], segments.shape[1], segments.shape[2], 1))

x_train_cn1, x_valn_cn1, y_train_cn1, y_valn_cn1 = train_test_split(segments_cnn1, labels, test_size=0.2,
                                                    random_state=2)
x_train, x_val, y_train, y_val = train_test_split(segments, labels, test_size=0.2, random_state=2)

# test_model(epochs, batch_size, "cnn1", m1_layers, x_train_cn1, x_valn_cn1, y_train_cn1, y_valn_cn1)
# test_model(epochs, batch_size, "rnn1", m2_layers, x_train, x_val, y_train, y_val)
# test_model(epochs, batch_size, "cnn2", m3_layers, x_train, x_val, y_train, y_val)

# test = grid_search([8, 16], [8, 16], m1_layers, "CNN2D")


