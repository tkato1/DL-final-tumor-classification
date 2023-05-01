from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import load_data
from tensorflow.keras import Sequential

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class ClassifierModel(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(ClassifierModel, self).__init__()

        self.batch_size = 32
        self.num_classes = 3
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        # TODO: Initialize all trainable parameters
        #convolutional layer 1
        # 5 x 5 filters that operate on 3 channels and and output 16 filters
        self.classifier_layers = Sequential([
            tf.keras.layers.Conv2D(32, 1, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 1, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 1, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 1, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation="softmax"),
            tf.keras.layers.Dense(3)
        ])
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.03)

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        return self.classifier_layers(inputs)


    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        :return: the accuracy of the model as a Tensor

        the paper defined the accuracy as the folllowing with true positive (TP), false positive (FP), false negative (FN), and true negative
                  TP + TN
            -------------------
             TP + FP + TN + FN
(TN) 
        """
        # tp = correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        # fp = correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 0))
        # fn = correct_predictions = tf.equal(tf.argmax(logits, 0), tf.argmax(labels, 1))
        # tn = correct_predictions = tf.equal(tf.argmax(logits, 0), tf.argmax(labels, 0))
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    # have an array of indices of num_examples
    num_examples = np.arange(np.shape(train_inputs)[0])
    #shuffle them 
    num_examples = tf.random.shuffle(num_examples)
    
    training_inputs = tf.gather(train_inputs, num_examples)
    training_labels = tf.gather(train_labels, num_examples)

    for batch in range(0, len(num_examples), model.batch_size):
        batch_end = batch + model.batch_size
        inputs = training_inputs[batch:batch_end]
        labels = training_labels[batch:batch_end]
        with tf.GradientTape() as tape:
            output = model.call(tf.image.random_flip_left_right(inputs))
            loss = model.loss(output, labels)
            print("loss is", loss)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """

    result = model.call(test_inputs)
    return model.accuracy(result, test_labels)


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''

    inputs, labels = load_data("data/set1", downsampling_factor=4)
    train_inputs = np.array([np.array(val) for val in inputs])[:600]
    train_inputs = train_inputs.reshape(-1, 1, 128, 128)
    train_inputs = train_inputs.transpose(0, 2, 3, 1)
    train_labels = labels[:600]
    print(np.shape(train_inputs))
    print(np.shape(train_labels))
    test_inputs = np.array([np.array(val) for val in inputs])[600:]
    test_labels = labels[600:]


    model = ClassifierModel()
    for i in range(10):
        train(model, train_inputs, train_labels)
        print("epoch", i + 1, "testing result:", test(model, test_inputs, test_labels).numpy())


if __name__ == '__main__':
    main()