# DL-final-tumor-classification

This repository contains an exploration and reimplementation of this paper: https://doi.org/10.30534/ijatcse/2019/155862019.

## Installing Dependencies
conda env create -f environment.yml

## preprocess.py
Processes the data in /data into np.ndarray X of shape (number_of_images, image_width, image_height), and np.ndarray y of shape (number_of_images, 1).
Contains additional utils to prepare data for CNN.

## classifier.py
Implementation of the CNN architecture described in the paper. Contains code for additional visualization and analysis of model performance

## Contributors: 
Toshiki Kato, Raphael Li, Alexander Zheng

## References:
1. Alqudah, A. M., Alquraan, H., Qasmieh, I. A., Alqudah, A., & Al-Sharu, W. (2019). Brain tumor classification using deep learning technique - A comparison between cropped, uncropped, and segmented lesion images with different sizes. International Journal of Advanced Trends in Computer Science and Engineering, 8(6), 3684–3691. https://doi.org/10.30534/ijatcse/2019/155862019 
2. Cheng, Jun, et al. Enhanced performance of brain tumor classification via tumor region augmentation and partition. PloS one, Vol 10, 2015. DOI: 10.1371/ journal.pone.0140381. 
3. Confusion Matrix Visualizer: https://github.com/DTrimarchi10/confusion_matrix
