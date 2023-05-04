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
    num_examples = np.arange(np.shape(X)[0])
    #shuffle them 
    num_examples = tf.random.shuffle(num_examples)
    
    X = tf.gather(X, num_examples)
    y = tf.gather(y, num_examples)
    y_not_one_hot = y
    y = tf.one_hot(y, 3, dtype=tf.float32)
    y = tf.reshape(y, (y.shape[0], y.shape[2]))

    train_inputs = np.array([np.array(val) for val in X])[:2100]
    train_inputs = train_inputs.reshape(-1, 1, 128, 128)
    train_inputs = train_inputs.transpose(0, 2, 3, 1)
    train_labels = y[:2100]

    validation_inputs = np.array([np.array(val) for val in X])[2100:2582]
    validation_inputs = validation_inputs.reshape(-1, 1, 128, 128)
    validation_inputs = validation_inputs.transpose(0, 2, 3, 1)
    validation_labels = y[2100:2582]

    test_inputs = np.array([np.array(val) for val in X])[2582:]
    test_inputs = test_inputs.reshape(-1, 1, 128, 128)
    test_inputs = test_inputs.transpose(0, 2, 3, 1)
    test_labels = y_not_one_hot[2582:]

    inputs = np.array([np.array(val) for val in X])
    inputs = inputs.reshape(-1, 1, 128, 128)
    inputs = inputs.transpose(0, 2, 3, 1)


    # print("final")
    # print(train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape)

    model = Sequential()
    model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(16, 2, activation="relu"))
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
    model.add(tf.keras.layers.Dense(5, activation="softmax"))
    model.add(tf.keras.layers.Dense(3))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    model.fit(train_inputs, train_labels, epochs=500, batch_size=64,
              validation_data=(validation_inputs, validation_labels))

    
    preds = np.argmax(model.predict(test_inputs), axis=1)
    print(tf.math.confusion_matrix(labels=test_labels, predictions=preds, num_classes=3))

    pred_2 = np.argmax(model.predict(inputs), axis=1)
    print(tf.math.confusion_matrix(labels=y_not_one_hot, predictions=pred_2, num_classes=3))

    # model.fit(train_inputs, train_labels, epochs=5, batch_size=64,
    #           validation_data=(validation_inputs, validation_labels))
    # y_prob = model.predict(test_inputs)
    # y_pred = np.argmax(y_prob, axis=1)
    # confusion = tf.math.confusion_matrix(
    #     labels=test_labels, predictions=y_pred).numpy()
    # print("confusion matrix:\n", confusion)
    # num_classes = 3
    # accuracy = np.zeros(num_classes)
    # precision = np.zeros(num_classes)
    # specificity = np.zeros(num_classes)
    # sensitivity = np.zeros(num_classes)
    # for i in range(num_classes):
    #     tp = confusion[i, i]
    #     fp = np.sum(confusion[:, i]) - tp
    #     fn = np.sum(confusion[i, :]) - tp
    #     tn = tp - tp
    #     for k in range(num_classes):
    #         for j in range(num_classes):
    #             if k != i and j != i:
    #                 tn += confusion[k, j]
    #     accuracy[i] = (tp + tn)/(tp + fp + tn + fn)
    #     sensitivity[i] = (tp)/(tp + fn)
    #     specificity[i] = (tn)/(tn + fp)
    #     precision[i] = (tp)/(tp + fp)


if __name__ == '__main__':
    main()
