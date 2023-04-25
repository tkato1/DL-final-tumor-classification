import os
import scipy.io
import imageio
import mat73
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import block_reduce


# specify the input directory containing .mat files
input_dir = "data/test"

def load_data(input_dir, process="uncrop", downsampling_factor=1):
    """
    input_dir: directory of raw .mat files
    process: the modification to the image(uncrop, crop, segment)
    """
    raw_images = os.listdir(input_dir)
    number_images = len(raw_images)

    #pre-initialization
    X = np.empty(number_images, dtype=object)
    y = np.empty(number_images, dtype='uint8')

    for i, filename in enumerate(raw_images):
        if filename.endswith('.mat'): # load the .mat file using scipy.io.loadmat()
            data = mat73.loadmat(os.path.join(input_dir, filename))['cjdata'] #dict_keys(['PID', 'image', 'label', 'tumorBorder', 'tumorMask'])
            image_data = np.asarray(data['image'].astype('uint8'))
            downsampled_image = downsample(image_data, factor=downsampling_factor) #downsampling image
            X[i] = downsampled_image
            y[i] = data['label']

    return X, y

def downsample(image, factor=1):
    """
    image: a 2d array where each entry represents pixel of an image
    factor: a integer factor by which to shrink the image by
    """
    return block_reduce(image, block_size=(factor, factor), func=np.mean)

def visualize(image):
    """
    image: a 2d array where each entry represents pixel of an image
    """
    plt.imshow(image, cmap="gray")
    plt.show()

if __name__ == "__main__":
    X, y = load_data(input_dir, process='uncrop', downsampling_factor=1)
    print(f"X shape: {X[0].shape}, Y shape: {y.shape}")
    visualize(X[0])
    print(f"X: {X}") #[[image1], [image2], [image3]] where images are 2d arrays with entries representing pixels of the image
    print(f"Y: {y}") #[label1, label2, label3] where labels in {1., 2., 3.} representing class of tumor