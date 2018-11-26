from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import pandas as pd
from keras.preprocessing import sequence
import pandas as pd
from keras.preprocessing import sequence
import nltk
import numpy as np
# import nltk.data
# nltk.download() 

# load data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

train_data = pd.read_csv('train.csv')
X_train = train_data.text.values
y_train = train_data.label.values

test = []
for sentence in X_train:
    words = nltk.word_tokenize(sentence)
    test.append(np.pad(words, (0, 150 - len(words)), 'constant'))

X_train = test

# creat model
vocabulary_size = 5000
max_words = 150
embedding_size=32
batch_size = 64
num_epochs = 3

model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

# scores = model.evaluate(X_test, y_test, verbose=0)
# print('Test accuracy:', scores[1])