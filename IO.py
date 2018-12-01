import pandas as pd
import nltk
import numpy as np
# import nltk.data
# nltk.download() 


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

def export(out, output):
    data = pd.read_csv('sample.csv')
    for i in range(len(out)):
        if(out[i] > 0.5):
            data.Label[i] = 1
        else:
            data.Label[i] = -1
    data.to_csv(output, index=False)

def get_data2(name, vocabulary_size):
    train_data = pd.read_csv(name)
    raw_data = train_data.text.values
    raw_like = train_data.favoriteCount.values
    raw_retw = train_data.retweetCount.values
    X_label = np.array([0 if i < 0 else 1 for i in train_data.label.values])
    x = len(raw_data)
    X_train = np.zeros((x, 150))
    X_train2 = np.zeros((x, 2))
    for i in range(x):
        words = nltk.word_tokenize(raw_data[i])
        X_train2[i][0] = raw_like[i]
        X_train2[i][1] = raw_retw[i]
        y = len(words)
        for j in range(y):
            X_train[i][150-y+j] = hash(words[j]) % vocabulary_size
    return X_train, X_train2 , X_label


def get_test2(name, vocabulary_size):
    train_data = pd.read_csv(name)
    raw_data = train_data.text.values
    raw_like = train_data.favoriteCount.values
    raw_retw = train_data.retweetCount.values
    x = len(raw_data)
    X_train = np.zeros((x, 150))
    X_train2 = np.zeros((x, 2))
    for i in range(x):
        words = nltk.word_tokenize(raw_data[i])
        X_train2[i][0] = raw_like[i]
        X_train2[i][1] = raw_retw[i]
        y = len(words)
        for j in range(y):
            X_train[i][150-y+j] = hash(words[j]) % vocabulary_size
    return X_train, X_train2
