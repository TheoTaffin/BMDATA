import os
import string
import json
from collections import Counter

root_dir = 'C:\\Users\\Theo\\PycharmProjects\\BMDATA'
pos_dir = 'TP1_NLP\\review_polarity\\txt_sentoken\\pos'
neg_dir = 'TP1_NLP\\review_polarity\\txt_sentoken\\neg'


def clean_doc(doc):
    # remove punctuation characters
    for char in string.punctuation:
        doc = doc.replace(char, "")

    # remove digits
    for num in string.digits:
        doc = doc.replace(num, "")

    # remove escape characters
    tmp = [word for word in doc.split() if len(word) > 2]

    doc = " ".join(tmp)

    return doc


def add_doc_to_vocab(doc, vocab):
    doc_voc = Counter(doc.split())

    for word, count in doc_voc.items():
        if word in vocab.keys():
            vocab[word] += count
        else:
            vocab.update({word: count})

    return vocab


def process_doc(directory):

    dct = {}
    vocab = Counter()

    for review in os.listdir(directory):
        with open(os.path.join(directory, review), "r") as f:
            doc = clean_doc(f.read())
            vocab = add_doc_to_vocab(doc, vocab)
            dct[review.split("_")[0]] = doc

    return dct, vocab


def save_list(lines, file_names):
    with open(file_names, "w") as f:
        f.write(json.dumps(lines))


positive_review, pos_vocab = process_doc(os.path.join(root_dir, pos_dir))
negative_review, neg_vocab = process_doc(os.path.join(root_dir, neg_dir))


save_list(pos_vocab, "pos_vocab.txt")
save_list(neg_vocab, "neg_vocab.txt")
