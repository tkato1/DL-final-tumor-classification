import os
import scipy.io
import imageio
import mat73
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse


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
                masked_image = np.where(data['tumorMask'], image_data, 0) #masking image with tumorMask
                cropped_image = crop_nonzero(masked_image) #cropped it
                image_data = cv2.resize(cropped_image, (512,512), interpolation=cv2.INTER_AREA) #scaled to 512x512
            if process == "crop":
                masked_image = np.where(data['tumorMask'], image_data, 0) #this is only to retrieve the crop indices
                cropped_masked_image, indices = crop_nonzero(masked_image, include_index=True) #cropped it
                a, b, c, d = indices
                # visualize(cropped_masked_image) #masked
                # visualize(image_data[a:b, c:d]) #unmasked
                image_data = cv2.resize(image_data[a:b, c:d], (512,512), interpolation=cv2.INTER_AREA) #scaled to 512x512

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


def crop_nonzero(arr, include_index=False):
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
    if include_index:
        return cropped_arr, (center[0]-half_height, center[0]+half_height, center[1]-half_width, center[1]+half_width)
    else:
        return cropped_arr


if __name__ == "__main__":
    TESTING = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=str, default='uncrop',
                        help='specify the process to be done. i.e, uncrop, crop, segment')
    args = parser.parse_args()
    process = args.process

    X, y = load_data(input_dir, process=process, downsampling_factor=4) 
    print(f"X shape: {X[0].shape}, Y shape: {y.shape}")
    #[[image1], [image2], [image3]] where images are 2d arrays with entries representing pixels of the image
    #[label1, label2, label3] where labels in {1., 2., 3.} representing class of tumor

    if TESTING:
        visualize(X[0])
    
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