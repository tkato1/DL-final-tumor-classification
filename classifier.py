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
    
    :return: None
    '''

    inputs, labels = load_data("data/set1", downsampling_factor=4)
    inputs_1, labels_1 = load_data("data/set2", downsampling_factor=4)
    # print(np.shape(inputs_1))
    inputs = np.concatenate([inputs, inputs_1])
    labels = np.concatenate([labels, labels_1])
    train_inputs = np.array([np.array(val) for val in inputs])[:1200]
    train_inputs = train_inputs.reshape(-1, 1, 128, 128)
    train_inputs = train_inputs.transpose(0, 2, 3, 1)
    train_labels = labels[:1200]
    test_inputs = np.array([np.array(val) for val in inputs])[1200:]
    test_inputs = test_inputs.reshape(-1, 1, 128, 128)
    test_inputs = test_inputs.transpose(0, 2, 3, 1)
    test_labels = labels[1200:]

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
    opt = SGD(learning_rate=0.03)
    model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(train_inputs, train_labels, epochs=20, batch_size=64, validation_data=(test_inputs, test_labels))

if __name__ == '__main__':
    main()