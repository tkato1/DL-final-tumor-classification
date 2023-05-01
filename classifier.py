from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import load_data
from tensorflow.keras import Sequential, models
from tensorflow.keras.optimizers import SGD


import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    '''
    Read in MRI data of 3 classes, initialize model, and train and 
    test  model for a number of epochs.
    '''

    X, y = load_data("data/set1", downsampling_factor=4)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, 3, dtype=tf.float32)
    y = tf.reshape(y, (y.shape[0], y.shape[2]))

    train_inputs = np.array([np.array(val) for val in X])[:2500]
    # print("before reshaping 1")
    # print(train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape)
    train_inputs = train_inputs.reshape(-1, 1, 128, 128)
    train_inputs = train_inputs.transpose(0, 2, 3, 1)
    train_labels = y[:2500]

    test_inputs = np.array([np.array(val) for val in X])[2500:]
    test_inputs = test_inputs.reshape(-1, 1, 128, 128)
    # print("before reshaping 2")
    # print(train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape)
    test_inputs = test_inputs.transpose(0, 2, 3, 1)
    test_labels = y[2500:]

    # print("final")
    # print(train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape)

    model = Sequential()
    model.add(tf.keras.layers.Conv2D(32, 1, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(16, 1, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(8, 1, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(16, 1, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.add(tf.keras.layers.Dense(3))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]     

    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)
    
    model.fit(train_inputs, train_labels, epochs=20, batch_size=32, validation_data=(test_inputs, test_labels))

if __name__ == '__main__':
    main()