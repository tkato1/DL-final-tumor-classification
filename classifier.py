from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import load_data
from tensorflow.keras import Sequential, models
from tensorflow.keras.optimizers import SGD
from confusion_matrix import make_confusion_matrix


import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def stats(confusion, num_classes=3):
    accuracy = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    sensitivity = np.zeros(num_classes)
    for i in range(num_classes):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp
        fn = np.sum(confusion[i, :]) - tp
        tn = tp - tp
        for k in range(num_classes):
            for j in range(num_classes):
                if k != i and j != i:
                    tn += confusion[k, j]
        accuracy[i] = (tp + tn)/(tp + fp + tn + fn +
                                 tf.keras.backend.epsilon())
        sensitivity[i] = (tp)/(tp + fn + tf.keras.backend.epsilon())
        specificity[i] = (tn)/(tn + fp + tf.keras.backend.epsilon())
        precision[i] = (tp)/(tp + fp + tf.keras.backend.epsilon())

    return accuracy, precision, specificity, sensitivity


def main():
    '''
    Read in MRI data of 3 classes, initialize model, and train and 
    test  model for a number of epochs.
    '''

    X, y = load_data("data/set1", downsampling_factor=4, process="uncrop")
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
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

    train_test_inputs = tf.convert_to_tensor(np.concatenate(
        [train_inputs, test_inputs], 0), dtype=tf.float32)
    train_test_labels = tf.convert_to_tensor(np.concatenate(
        [y_not_one_hot[:2100], y_not_one_hot[2582:]], 0), dtype=tf.int32)

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
    epochs = 10

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    history = model.fit(train_inputs, train_labels, epochs=epochs, batch_size=64,
                        validation_data=(validation_inputs, validation_labels))

    plt.plot(np.linspace(0, epochs, epochs, endpoint=True),
             history.history['accuracy'])
    plt.plot(np.linspace(0, epochs, epochs, endpoint=True),
             history.history['val_accuracy'])
    plt.title('Accuracy vs Epochs')
    plt.legend(['Training_Accuracy', 'Validation_Accuracy'])
    plt.xlabel('Epochs')
    plt.xticks(np.arange(0, epochs+1, 100))
    plt.yticks(np.arange(0, 1.01, .2))
    plt.ylabel('Accuracy')
    plt.savefig("visualizations/segmented_training_plot")
    plt.show()

    y_prob = model.predict(train_test_inputs)
    y_pred = np.argmax(y_prob, axis=1)

    confusion = tf.math.confusion_matrix(
        labels=train_test_labels, predictions=y_pred).numpy()
    print("confusion matrix:\n", confusion, confusion.shape)

    accuracy, precision, specificity, sensitivity = stats(confusion, 3)

    print(
        f"accuracy: {accuracy}, precision: {precision}, specificity: {specificity}, sensitivity: {sensitivity}")

    with open('stats/stat.txt', 'a') as f:
        f.write(
            f"accuracy: {accuracy}, precision: {precision}, specificity: {specificity}, sensitivity: {sensitivity}")

    make_confusion_matrix(confusion,
                          categories=["Glioma", "Meningioma",
                                      "Pituitary Tumor"],
                          output_file="visualizations/confusion_matrix_segmented")


if __name__ == '__main__':
    main()
