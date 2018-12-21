from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import TimeDistributed,Bidirectional
from keras.layers import Activation
from keras.layers import Flatten
import pandas as pd
import numpy as np
from config import Config
from dataprocess_old import Preprocess
import tensorflow as tf



def train(X,Y):

    model = Sequential()
    model.add(Embedding(input_dim=5154, output_dim=180,input_shape=(32,)))
    model.add(Bidirectional(LSTM(64,return_sequences=True)))
    # model.add(Dropout(0.3))
    # model.add(Flatten())
    model.add(TimeDistributed(Dense(1,activation='sigmoid')))
    # model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, Y, batch_size=5000, epochs=5, verbose=1, validation_split=0.3)

    return model

if __name__ == '__main__':
    X = pd.read_csv("train_X.csv")
    Y = pd.read_csv("train_Y.csv")
    X = X.astype(int)
    Y = Y.astype(int)
    X = X.values
    Y = Y.values
    # X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = train(X, Y)

    model.evaluate(X, Y)


