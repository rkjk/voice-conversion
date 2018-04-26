import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

from sys import exit

source_data = np.load('source_input_logfbank.npy')
target_data = np.load('target_input_logfbank.npy')

source_data = source_data[..., np.newaxis]
target_data = target_data[..., np.newaxis]

model = Sequential()
#model.add(Embedding(source_data.shape, output_dim=1))
model.add(LSTM(1, input_shape=(26,1), return_sequences=True ))

model.compile(loss='mean_squared_error',optimizer='sgd')

model.fit(source_data, target_data, batch_size=10, epochs=10)
