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
import argparse

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

size_to_df = {32:16, 64:8, 128:4}


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


def split(X, y, d):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    num_examples = np.arange(np.shape(X)[0])
    num_examples = tf.random.shuffle(num_examples)
    X = tf.gather(X, num_examples)
    y = tf.gather(y, num_examples)
    y_not_one_hot = y
    y = tf.one_hot(y, 3, dtype=tf.float32)
    y = tf.reshape(y, (y.shape[0], y.shape[2]))

    train_inputs = np.array([np.array(val) for val in X])[:2100]
    train_inputs = train_inputs.reshape(-1, 1, d, d)
    train_inputs = train_inputs.transpose(0, 2, 3, 1)
    train_labels = y[:2100]

    validation_inputs = np.array([np.array(val) for val in X])[2100:2582]
    validation_inputs = validation_inputs.reshape(-1, 1, d, d)
    validation_inputs = validation_inputs.transpose(0, 2, 3, 1)
    validation_labels = y[2100:2582]

    test_inputs = np.array([np.array(val) for val in X])[2582:]
    test_inputs = test_inputs.reshape(-1, 1, d, d)
    test_inputs = test_inputs.transpose(0, 2, 3, 1)

    train_test_inputs = tf.convert_to_tensor(np.concatenate(
        [train_inputs, test_inputs], 0), dtype=tf.float32)
    train_test_labels = tf.convert_to_tensor(np.concatenate(
        [y_not_one_hot[:2100], y_not_one_hot[2582:]], 0), dtype=tf.int32)
    
    return train_inputs, train_labels, validation_inputs, validation_labels, train_test_inputs, train_test_labels


def main():
    '''
    Read in MRI data of 3 classes, initialize model, and train and 
    test  model for a number of epochs.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=64, help='image size, 32, 64, or 128')
    parser.add_argument('--process', type=str, default="uncrop", help='process ie. uncrop, crop, segment')
    parser.add_argument('--input_dir', type=str, default="../data/raw", help="input data directory")
    parser.add_argument('--lr', type=int, default=0.001, help='learning_rate')
    parser.add_argument('--epochs', type=int, default=600, help='epochs')
    args = parser.parse_args()

    X, y = load_data(args.input_dir, downsampling_factor=size_to_df[args.image_size], process=args.process)
    train_inputs, train_labels, validation_inputs, validation_labels, train_test_inputs, train_test_labels = split(X, y, args.image_size)

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

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]
    epochs = args.epochs

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    history = model.fit(train_inputs, train_labels, epochs=epochs, batch_size=64,
                        validation_data=(validation_inputs, validation_labels))

    plt.plot(np.linspace(0, epochs, epochs, endpoint=True),
             history.history['accuracy'])
    plt.plot(np.linspace(0, epochs, epochs, endpoint=True),
             history.history['val_accuracy'])
    plt.title(f"Accuracy vs Epochs for {args.process} images")
    plt.legend(['Training_Accuracy', 'Validation_Accuracy'])
    plt.xlabel('Epochs')
    plt.xticks(np.arange(0, epochs+1, 100))
    plt.yticks(np.arange(0, 1.01, .2))
    plt.ylabel('Accuracy')
    plt.savefig(f"../visualizations/{args.process}/training_accuracy{args.image_size}1")
    plt.show()

    y_prob = model.predict(train_test_inputs)
    y_pred = np.argmax(y_prob, axis=1)

    confusion = tf.math.confusion_matrix(labels=train_test_labels, predictions=y_pred).numpy()
    print("confusion matrix:\n", confusion)

    accuracy, precision, specificity, sensitivity = stats(confusion, 3)
    print(f"accuracy: {accuracy}\nprecision: {precision}\nspecificity: {specificity}\nsensitivity: {sensitivity}\n")

    with open('../stats/stat.txt', 'a') as f:
        f.write(f"{args.process} || accuracy: {accuracy}, precision: {precision}, specificity: {specificity}, sensitivity: {sensitivity}")

    make_confusion_matrix(confusion,
                          categories=["Glioma", "Meningioma",
                                      "Pituitary Tumor"],
                          title=f"Confusion Matrix",
                          output_file=f"../visualizations/{args.process}/confusion_matrix{args.image_size}1")

if __name__ == '__main__':
    main()