import os
import sys

import tensorflow as tf
from tensorflow import keras

layer = sys.argv[1]

def create_model(size: int):
    if layer == 'dense':
        model = tf.keras.Sequential([keras.layers.Dense(size, activation='linear', input_shape=(size,))])
    elif layer == 'conv1d':
        model = tf.keras.Sequential([keras.layers.Conv1D(size, kernel_size=size-1, activation='linear', input_shape=(size,1))])
    elif layer == 'gru':
        model = tf.keras.Sequential([keras.layers.GRU(size,  input_shape=(None,size))])
    elif layer == 'lstm':
        model = tf.keras.Sequential([keras.layers.LSTM(size,  input_shape=(None,size))])
    elif layer == 'tanh':
        model = tf.keras.Sequential([keras.layers.Activation(tf.nn.tanh, input_shape=(None,size))])
    elif layer == 'relu':
        model = tf.keras.Sequential([keras.layers.Activation(tf.nn.relu, input_shape=(None,size))])
    elif layer == 'sigmoid':
        model = tf.keras.Sequential([keras.layers.Activation(tf.nn.sigmoid, input_shape=(None,size))])

    return model

for size in [4, 8, 16, 32, 64]:
    model = create_model(size)
    model.save(f'{layer}_{size}.tf')
    os.system(f'python3.9 -m tf2onnx.convert --saved-model {layer}_{size}.tf --output {layer}_{size}.onnx')
