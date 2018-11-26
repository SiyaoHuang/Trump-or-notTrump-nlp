from load import get_data, get_test
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

vocabulary_size = 5000
max_words = 150
embedding_size=32
batch_size = 64
num_epochs = 3
embedding_size=32

def run2(train, test):
    X_train, y_train = get_data(train, vocabulary_size)
    X_test = get_test(test, vocabulary_size)
    print(X_train.shape, y_train.shape)
    print(X_test.shape)
    model = Model()
    fit(X_train, y_train, model)
    out(X_test, model)

def run(train, test):
    X_train, y_train = get_data(train, vocabulary_size)
    X_test = get_test(test, vocabulary_size)
    print(X_train.shape, y_train.shape)
    print(X_test.shape)
    model = Model()
    fit(X_train, y_train, model)
    out(X_test, model)

def Model():
    embedding_size=32
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def fit(X_train, y_train, model):
    X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
    X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

def out(X_test, model):
    out = model.predict(X_test)
    for i in out:
        if(i > 0.7):
            print(1)
        else:
            print(-1)
