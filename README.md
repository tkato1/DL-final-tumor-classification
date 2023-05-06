# DL-final-tumor-classification

This repository contains an exploration and reimplementation of this paper: https://doi.org/10.30534/ijatcse/2019/155862019.

## Installing Dependencies**
conda env create -f environment.yml

## preprocess.py
Processes the data in /data into np.ndarray X of shape (number_of_images, image_width, image_height), and np.ndarray y of shape (number_of_images, 1).
Contains additional utils to prepare data for CNN.

## classifier.py
Implementation of the CNN architecture described in the paper. Contains code for additional visualization and analysis of model performance
