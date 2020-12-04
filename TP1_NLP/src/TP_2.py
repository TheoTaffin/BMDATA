import os
import json
import pandas as pd
import numpy as np
from TP1_NLP.src.TP_analysis import doc_to_line, load_clean_dataset, create_tokenizer

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant

root_dir = 'C:\\Users\\Theo\\PycharmProjects\\BMDATA'
pos_dir = 'TP1_NLP\\review_polarity\\txt_sentoken\\pos'
neg_dir = 'TP1_NLP\\review_polarity\\txt_sentoken\\neg'


def encode_doc(tokenizer, max_length, docs):

    docs_tk = tokenizer.texts_to_sequences(docs)
    docs_tk = pad_sequences(docs_tk, max_length, padding='post')

    return docs_tk


def define_model(n_words, weights, max_length, embedding_dim):

    embeddings1_1 = layers.Embedding(input_dim=n_words, input_length=max_length,
                                     output_dim=embedding_dim,
                                     embeddings_initializer=Constant(weights),
                                     trainable=False)
    # conv1_1 = layers.Conv1D(filters=32, kernel_size=8, activation='relu')
    # maxpool1_1 = layers.MaxPooling1D(pool_size=2)
    # flatten1_1 = layers.Flatten()
    # dense1_1 = layers.Dense(units=50, activation='relu')
    lstm1_1 = layers.LSTM(units=100, dropout=0.2, recurrent_dropout=0.2)
    dense1_2 = layers.Dense(units=1, activation='sigmoid')


    # model = Sequential([embeddings1_1, conv1_1, maxpool1_1, flatten1_1, dense1_1, dense1_2])
    model = Sequential([embeddings1_1, lstm1_1, dense1_2])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

    model.summary()

    return model


def build_embeddings(tokenizer):

    # load the whole embedding into memory
    embeddings_index = dict()
    with open(os.path.join(root_dir, 'TP1_NLP\\glove.6B.50d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 50))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


with open(os.path.join(root_dir, 'TP1_NLP/vocab.txt'), 'r') as f:
    vocab = json.loads(f.read())

docs_pos, lab_pos = load_clean_dataset(os.path.join(root_dir, pos_dir), vocab, label=1)
docs_neg, lab_neg = load_clean_dataset(os.path.join(root_dir, neg_dir), vocab, label=0)

docs_pos.extend(docs_neg)
lab_pos.extend(lab_neg)

dataset = np.array(list(zip(docs_pos, lab_pos)))
np.random.shuffle(dataset)
docs, labels = zip(*dataset)

tokenizer = create_tokenizer(docs)

max_length = max(map(len, docs))
docs = encode_doc(tokenizer, max_length, docs)

n_train = int(0.8 * len(dataset))

x_train, y_train = docs[:n_train], np.array(labels[:n_train])
x_test, y_test = docs[n_train:], np.array(labels[n_train:])

embedding_matrix = build_embeddings(tokenizer)


model1 = define_model(n_words=len(tokenizer.word_index)+1, weights=embedding_matrix,
                      max_length=max_length, embedding_dim=50)
model1.fit(x_train, y_train, epochs=10, batch_size=32)
model1.evaluate(x_test, y_test)
