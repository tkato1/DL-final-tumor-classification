import os
import scipy.io
import imageio
import mat73
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import block_reduce


# specify the input directory containing .mat files
input_dir = "data/set1"

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
            if process == "segment":
                image_data = np.where(data['tumorMask'], image_data, 0) #masking image with tumorMask
                image_data=  crop_nonzero(image_data)
                image_data = padding(image_data, 128, 128)
            if process == "crop":
                image_data=  crop_nonzero(image_data)
                image_data = padding(image_data, 128, 128)
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

def crop_nonzero(arr):
    # find indices of all non-zero elements in the array
    non_zero_indices = np.argwhere(arr != 0)

    # compute minimum and maximum indices in each dimension
    min_indices = np.min(non_zero_indices, axis=0)
    max_indices = np.max(non_zero_indices, axis=0)

    # compute center of bounding box
    center = (min_indices + max_indices) / 2

    # compute size of bounding box based on original aspect ratio
    height = max_indices[0] - min_indices[0]
    width = max_indices[1] - min_indices[1]
    aspect_ratio = width / height
    new_width = int(np.ceil(aspect_ratio * height))

    # crop array to bounding box size centered around center
    half_height = height // 2
    half_width = new_width // 2
    center = center.astype(int)
    cropped_arr = arr[center[0]-half_height:center[0]+half_height,
                      center[1]-half_width:center[1]+half_width]

    return cropped_arr

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


if __name__ == "__main__":
    #[[image1], [image2], [image3]] where images are 2d arrays with entries representing pixels of the image
    #[label1, label2, label3] where labels in {1., 2., 3.} representing class of tumor
    X, y = load_data(input_dir, process='uncrop', downsampling_factor=4)
    print(f"X shape: {X[2].shape}, Y shape: {y.shape}")

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(15):        
        axes[i].imshow(X[i], cmap="gray")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.show()