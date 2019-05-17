from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import  numpy as np

testData = pd.read_csv('data/test.csv').get_values()
trainData = pd.read_csv('data/train.csv').get_values()
x_train = trainData[:, 2:]
y_train = trainData[:, 1]

model = Sequential([
    Dense(64, input_shape=(300,)),
    Activation('sigmoid'),
    Dense(1),
    Activation('softmax'),
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

loss = model.fit(x_train, y_train, epochs=100, batch_size=32)
print(loss)