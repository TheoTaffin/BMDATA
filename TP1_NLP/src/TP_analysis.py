import os
import json
import pandas as pd
import numpy as np


# import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer


root_dir = 'C:\\Users\\Theo\\PycharmProjects\\BMDATA'
pos_dir = 'TP1_NLP\\review_polarity\\txt_sentoken\\pos'
neg_dir = 'TP1_NLP\\review_polarity\\txt_sentoken\\neg'


def doc_to_line(file_name, vocab):

    vector = []
    for word in file_name.split():
        if word in vocab:
            vector.append(word)

    return vector


def load_clean_dataset(directory, vocab, label):

    docs = []
    labels = []

    for review in os.listdir(directory):
        with open(os.path.join(directory, review), "r") as f:
            doc = doc_to_line(f.read(), vocab)
            # fix unknown empty line
            if doc:
                docs.append(doc)
                labels.append(label)

    return docs, labels


def create_tokenizer(lines):

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)

    return tokenizer


def define_model(n_words):
    dense1_1 = layers.Dense(units=50, activation='relu', input_shape=(1, n_words))
    dense1_2 = layers.Dense(units=1, activation='sigmoid')

    model = Sequential([dense1_1, dense1_2])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

    return model


def predict_sentiment(review, model, tokenizer):
    review_tk = tokenizer.texts_to_matrix(np.array(review).reshape(1,), mode='count')
    pred = model.predict(review_tk)

    if pred < 0.5:
        sentiment = 'negative'
    else:
        sentiment = 'positive'
    return pred, sentiment


# if __name__ == '__main__':
#     with open(os.path.join(root_dir, 'TP1_NLP/vocab.txt'), 'r') as f:
#         vocab = json.loads(f.read())
#
#     docs_pos, lab_pos = load_clean_dataset(os.path.join(root_dir, pos_dir), vocab, label=1)
#     docs_neg, lab_neg = load_clean_dataset(os.path.join(root_dir, neg_dir), vocab, label=0)
#
#     docs_pos.extend(docs_neg)
#     lab_pos.extend(lab_neg)
#
#     dataset = np.array(list(zip(docs_pos, lab_pos)))
#     np.random.shuffle(dataset)
#     docs, labels = zip(*dataset)
#
#     tokenizer = create_tokenizer(docs)
#
#     n_train = int(0.8 * len(dataset))
#
#     x_train, y_train = docs[:n_train], np.array(labels[:n_train])
#     x_test, y_test = docs[n_train:], np.array(labels[n_train:])
#
#
#     x_train_tk = tokenizer.texts_to_matrix(x_train, mode='binary')
#     x_test_tk = tokenizer.texts_to_matrix(x_test, mode='binary')
#     model1 = define_model(x_train_tk.shape[1])
#     model1.fit(x_train_tk, y_train, epochs=10, verbose=0)
#     model1.evaluate(x_test_tk, y_test)
#
#
#     x_train_tk = tokenizer.texts_to_matrix(x_train, mode='count')
#     x_test_tk = tokenizer.texts_to_matrix(x_test, mode='count')
#     model2 = define_model(x_train_tk.shape[1])
#     model2.fit(x_train_tk, y_train, epochs=10, verbose=0)
#     model2.evaluate(x_test_tk, y_test)
#
#
#     x_train_tk = tokenizer.texts_to_matrix(x_train, mode='tfidf')
#     x_test_tk = tokenizer.texts_to_matrix(x_test, mode='tfidf')
#     model3 = define_model(x_train_tk.shape[1])
#     model3.fit(x_train_tk, y_train, epochs=10, verbose=0)
#     model3.evaluate(x_test_tk, y_test)
#
#
#     x_train_tk = tokenizer.texts_to_matrix(x_train, mode='freq')
#     x_test_tk = tokenizer.texts_to_matrix(x_test, mode='freq')
#     model4 = define_model(x_train_tk.shape[1])
#     model4.fit(x_train_tk, y_train, epochs=10, verbose=0)
#     model4.evaluate(x_test_tk, y_test)
#
#
#     # test positive text
#     text = 'Best movie ever! It was great, I recommend it.'
#     percent, sentiment = predict_sentiment(text, model2, tokenizer)
#     print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
#
#     # test negative text
#     text = 'This is a bad movie. it sucks, i will never watch it again, this is awful, no effort'
#     percent, sentiment = predict_sentiment(text, model2, tokenizer)
#     print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
