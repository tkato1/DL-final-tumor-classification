import os
import scipy.io
import imageio
import mat73
import numpy as np
import matplotlib.pyplot as plt

# specify the input directory containing .mat files
input_dir = "raw_data/test"

def load_data(input_dir, process="uncropped"):
    """
    input_dir: directory of raw .mat files
    process: the modification to the image(uncropped, cropped, segmented)
    """
    raw_images = os.listdir(input_dir)
    number_images = len(raw_images)

    #preinitialization
    X = np.empty(number_images, dtype=object)
    y = np.empty(number_images, dtype='uint8')

    for i, filename in enumerate(raw_images):
        if filename.endswith('.mat'):
            # load the .mat file using scipy.io.loadmat()
            data = mat73.loadmat(os.path.join(input_dir, filename))['cjdata']
            # print(data.keys()) ##the keys of the data
            image_data = np.asarray(data['image'].astype('uint8'))
            image_data = image_data.reshape((-1, 1))
            X[i] = image_data
            y[i] = data['label']
    return X, y

def visualize(image):
    """
    image: a 2d array where each entry represents pixel of an image
    """
    plt.imshow(image, cmap="gray")
    plt.show()

if __name__ == "__main__":
    X, y = load_data(input_dir)
    print(f"X shape: {X.shape}, Y shape: {y.shape}")
    print(f"X: {X}") #[[image1], [image2], [image3]] where images are 2d arrays with entries representing pixels of the image
    print(f"Y: {y}") #[label1, label2, label3] where labels in {1., 2., 3.} representing class of tumor