from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import load_data

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 3
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        # TODO: Initialize all trainable parameters
        #convolutional layer 1
        # 5 x 5 filters that operate on 3 channels and and output 16 filters
        self.conv_filter_1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=0.1), dtype=tf.float32)
        self.conv_bias_1 = tf.Variable(tf.zeros([16]))
        # convolutional layer 2
        # no more rbg, now channels is for each filter /feature
        self.conv_filter_2 = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=0.1), dtype=tf.float32) 
        self.conv_bias_2 = tf.Variable(tf.zeros([20]))
        # convolutional layer 3
        self.conv_filter_3 = tf.Variable(tf.random.truncated_normal([5,5,20,20], stddev=0.1), dtype=tf.float32)
        self.conv_bias_3 = tf.Variable(tf.zeros([20]))
        # dense layer 1
        self.dl1_weights = tf.Variable(tf.random.truncated_normal([80, 60], stddev=0.1))
        self.dl1_bias = tf.Variable(tf.zeros([60]))
        # dense layer 2
        self.dl2_weights = tf.Variable(tf.random.truncated_normal([60, 30], stddev=0.1))
        self.dl2_bias = tf.Variable(tf.zeros([30]))
        # dense layer 3
        self.dl3_weights = tf.Variable(tf.random.truncated_normal([30, 3], stddev=0.1))
        self.dl3_bias = tf.Variable(tf.zeros([3]))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        inputs = tf.cast(inputs, tf.float32)
        
        # Convolution Layer 1
        conv_layer_1 = tf.nn.conv2d(inputs, filters=self.conv_filter_1, strides=2, padding="SAME") # should stride be (batch_stride, height_stride, width_stride, channels_stride)?
        conv_layer_1 = tf.nn.bias_add(conv_layer_1, self.conv_bias_1)
        # Batch Normalization 1
        mean, variance = tf.nn.moments(conv_layer_1, axes=[0, 1, 2])
        batch_norm_1 = tf.nn.batch_normalization(conv_layer_1, mean, variance, None, None, 0.00001)
        # ReLU 1
        relu1 = tf.nn.relu(batch_norm_1)
        # Max Pooling 1
        max_pool_1 = tf.nn.max_pool(relu1, ksize=3, strides=2, padding="SAME")

        # Convolution Layer 2
        conv_layer_2 = tf.nn.conv2d(max_pool_1, filters=self.conv_filter_2, strides=2, padding="SAME")
        conv_layer_2 = tf.nn.bias_add(conv_layer_2, self.conv_bias_2)
        # Batch Normalization 2
        mean_2, variance_2 = tf.nn.moments(conv_layer_2, axes=[0, 1, 2])
        batch_norm_2 = tf.nn.batch_normalization(conv_layer_2, mean_2, variance_2, None, None, 0.00001)
        # ReLU 2
        relu2 = tf.nn.relu(batch_norm_2)
        # Max Pooling 3
        max_pool_2 = tf.nn.max_pool(relu2, ksize=2, strides=2, padding="SAME")

        #Convolution Layer 3
        if is_testing:
            conv_layer_3 = conv2d(inputs=max_pool_2, filters=self.conv_filter_3, strides=[1,1,1,1], padding="SAME")
        else:
            conv_layer_3 = tf.nn.conv2d(max_pool_2, filters=self.conv_filter_3, strides=1, padding="SAME") 
        conv_layer_3 = tf.nn.bias_add(conv_layer_3, self.conv_bias_3)
        # Batch Normalization 3
        mean_3, variance_3 = tf.nn.moments(conv_layer_3, axes=[0, 1, 2])
        batch_norm_3 = tf.nn.batch_normalization(conv_layer_3, mean_3, variance_3, None, None, 0.00001)
        # ReLU 3
        relu3 = tf.nn.relu(batch_norm_3)

        # Flatten
        flattened = self.flatten(relu3)

        # Dense Layer 1 + Dropout Layer
        dense_layer_1 = tf.matmul(flattened, self.dl1_weights) + self.dl1_bias
        dropout_layer_1 = tf.nn.dropout(dense_layer_1, 0.3)
        # Dense Layer 2 + Dropout Layer
        dense_layer_2 = tf.matmul(dropout_layer_1, self.dl2_weights) + self.dl2_bias
        dropout_layer_2 = tf.nn.dropout(dense_layer_2, 0.3)
        # Dense Layer 3
        dense_layer_3 = tf.matmul(dropout_layer_2, self.dl3_weights) + self.dl3_bias
        return dense_layer_3


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

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
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
            output = model.call(tf.image.random_flip_left_right (inputs), is_testing=False)
            loss = model.loss(output, labels)
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

    result = model.call(test_inputs, is_testing=True)
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

    train_inputs, train_labels = load_data("data/set1", downsampling_factor=4) 
    test_inputs, test_labels = load_data("data/set2", downsampling_factor=4) 
    model = Model()
    for i in range(10):
        train(model, train_inputs, train_labels)
        print("epoch", i + 1, "testing result:", test(model, test_inputs, test_labels).numpy())


if __name__ == '__main__':
    main()