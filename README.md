# DL-final-tumor-classification

## Installing Dependencies**
conda env create -f environment.yml

## preprocess.py
Processes the data in /data into np.ndarray X of shape (number_of_images, image_width, image_height), and np.ndarray y of shape (number_of_images, 1).
Contains additional utils to prepare data for CNN.

## classifier.py
Contains neural network implementaition and analysis