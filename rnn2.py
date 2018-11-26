from load import get_data, get_test
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# set value
vocabulary_size = 5000
max_words = 150
embedding_size=32
batch_size = 64
num_epochs = 3

# load data
X_train, y_train = get_data('train.csv', vocabulary_size)
# X_test = get_test('test.csv', vocabulary_size)
X_test, y_test = X_train[:batch_size], y_train[:batch_size]
X_train, y_train = X_train[batch_size:], y_train[batch_size:]
print(X_train.shape, y_train.shape)
print(X_test.shape)

# creat model
embedding_size=32
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])

out = model.predict(X_test)
print(out)
