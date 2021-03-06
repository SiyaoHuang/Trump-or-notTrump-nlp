from IO import get_data, get_test, export, get_data2, get_test2
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model, Sequential
import keras
from sklearn.svm import SVC
import numpy as np

vocabulary_size = 5000
max_words = 150
embedding_size=32
batch_size = 100
num_epochs = 12

def run(train, test, output):
    X_train, y_train = get_data(train, vocabulary_size)
    X_test = get_test(test, vocabulary_size)
    print(X_train.shape, y_train.shape)
    print(X_test.shape)
    model = Model1()
    fit(X_train, y_train, model)
    out(X_train, model, output)
    print('---------')

def Model1():
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
    model.add(LSTM(200))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def fit(X_train, y_train, model):
    X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
    X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

def out(X_test, model, output):
    out = model.predict(X_test)
    export(out, output)

def run2(train, test, output):
    X_train, X_train2, y_train = get_data2(train, vocabulary_size)
    X_test, X_test2 = get_test2(test, vocabulary_size)
    print(X_train.shape, X_train2.shape, y_train.shape)
    print(X_test.shape, X_test2.shape)
    model = Model2()
    fit2(X_train, X_train2, y_train, model)
    out2(X_test, X_test2, model, output)

def Model2():
    input_text = Input(shape=(max_words,))
    embed_layer = Embedding(vocabulary_size, embedding_size, input_length=max_words)(input_text)
    LSTM_layer = LSTM(100)(embed_layer)
    mid_layer = Dense(100, activation='relu')(LSTM_layer)
    input_like = Input(shape=(2,))
    x = keras.layers.concatenate([mid_layer, input_like])
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_text, input_like], outputs=[x])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def fit2(X_train, X_train2, y_train, model):
    X_validt, X_validl, y_valid = X_train[:batch_size], X_train2[:batch_size], y_train[:batch_size]
    X_traint, X_trainl, y_train2 = X_train[batch_size:], X_train2[batch_size:], y_train[batch_size:]
    model.fit([X_traint, X_trainl], y_train2, validation_data=([X_validt, X_validl], y_valid), batch_size=batch_size, epochs=num_epochs)

def out2(X_test, X_test2, model, output):
    out = model.predict([X_test, X_test2])
    export(out, output)

def run3(train, test, output):
    X_train, X_train2, y_train = get_data2(train, vocabulary_size)
    X_test, X_test2 = get_test2(test, vocabulary_size)
    print(X_train.shape, X_train2.shape, y_train.shape)
    print(X_test.shape, X_test2.shape)
    model1 = Model1()
    fit(X_train, y_train, model1)
    X_train_m = model1.predict(X_train)
    # print(X_train_m.shape)
    for i in range(X_train_m.shape[0]):
        if(X_train_m[i] > 0.5):
            X_train_m[i] = 1
        else:
            X_train_m[i] = -1
    X_train_m = np.concatenate((X_train_m, X_train2), axis = 1)
    X_test_m = model1.predict(X_test)
    X_test_m = np.concatenate((X_test_m, X_test2), axis = 1)
    clf = SVC(gamma='auto')
    clf.fit(X_train_m, y_train)
    out = clf.predict(X_test_m) 
    export(out, output)
    # model3 = Model3()
    # print(X_train_m.shape)
    # print(X_train_m)
    # fit(X_train_m, y_train, model3)
    # out(X_test_m, model3, output)

# def Model3():
#     input_a = Input(shape=(3,))
#     x = Dense(10, activation='relu')(input_a)
#     x = Dense(1, activation='relu')(x)
#     model = Model(inputs=[input_a], outputs=[x])
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print(model.summary())
#     return model
