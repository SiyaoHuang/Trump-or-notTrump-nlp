import pandas as pd
from keras.preprocessing import sequence
import nltk
import numpy as np
# import nltk.data
# nltk.download() 


def get_data1(name, vocabulary_size):
    train_data = pd.read_csv(name)
    raw_data = train_data.text.values
    X_label = np.array([0 if i < 0 else 1 for i in train_data.label.values])
    x = len(raw_data)
    X_train = np.zeros((x, 150))
    for i in range(x):
        words = nltk.word_tokenize(raw_data[i])
        y = len(words)
        for j in range(y):
            X_train[i][j] = hash(words[j]) % vocabulary_size
    return np.flip(X_train, 1), X_label


def get_test1(name, vocabulary_size):
    train_data = pd.read_csv(name)
    raw_data = train_data.text.values
    x = len(raw_data)
    X_train = np.zeros((x, 150))
    for i in range(x):
        words = nltk.word_tokenize(raw_data[i])
        y = len(words)
        for j in range(y):
            X_train[i][j] = hash(words[j]) % vocabulary_size
    return np.flip(X_train, 1)

def get_data(name, vocabulary_size):
    train_data = pd.read_csv(name)
    raw_data = train_data.text.values
    X_label = np.array([0 if i < 0 else 1 for i in train_data.label.values])
    x = len(raw_data)
    X_train = np.zeros((x, 150))
    for i in range(x):
        words = nltk.word_tokenize(raw_data[i])
        y = len(words)
        for j in range(y):
            X_train[i][150-y+j] = hash(words[j]) % vocabulary_size
    return X_train, X_label


def get_test(name, vocabulary_size):
    train_data = pd.read_csv(name)
    raw_data = train_data.text.values
    x = len(raw_data)
    X_train = np.zeros((x, 150))
    for i in range(x):
        words = nltk.word_tokenize(raw_data[i])
        y = len(words)
        for j in range(y):
            X_train[i][150-y+j] = hash(words[j]) % vocabulary_size
    return X_train
