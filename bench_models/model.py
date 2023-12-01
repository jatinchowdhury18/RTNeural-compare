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
    # model = create_model(size)
    tf_model = f'{layer}_{size}.tf'
    # model.save(tf_model)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
    with open(f'{layer}_{size}.tflite', 'wb') as tflite_model_file:
        tflite_model_file.write(converter.convert())

    # os.system(f'python3.9 -m tf2onnx.convert --saved-model {tf_model} --output {layer}_{size}.onnx')
