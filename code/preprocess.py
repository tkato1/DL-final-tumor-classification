import argparse
import os

import cv2
import mat73
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from skimage.measure import block_reduce


def load_data(input_dir, process="uncrop", downsampling_factor=1, jpegs=False, output_dir="", save_labels=False):
    """
    input_dir: directory of raw .mat files
    process: the modification to the image(uncrop, crop, segment)
    jpegs: if you want to save the data as a jpeg in the output_dir
    save_labels: if you want to save the labels of each image in the labels.txt file
    """
    # raw_images = os.listdir(input_dir)
    raw_images = [f for f in os.listdir(input_dir) if not f.startswith('.')]
    number_images = len(raw_images)

    # pre-initialization
    X = np.empty(number_images, dtype=object)
    y = np.empty(number_images, dtype='uint8')

    for i, filename in enumerate(raw_images):
        if filename.endswith('.mat'):  # load the .mat file using scipy.io.loadmat()
            # dict_keys(['PID', 'image', 'label', 'tumorBorder', 'tumorMask'])
            data = mat73.loadmat(os.path.join(input_dir, filename))['cjdata']
            image_data = np.asarray(data['image'].astype('uint8'))
            mask = np.asarray(data['tumorMask'])

            if process == "uncrop":
                image_data = cv2.resize(image_data, (512,512), interpolation=cv2.INTER_AREA) #scaled to 512x512

            if process == "segment":
                # masking image with tumorMask
                masked_image = np.where(mask, image_data, 0)
                cropped_image = crop_nonzero(masked_image)  # cropped it
                image_data = cv2.resize(
                    cropped_image, (512, 512), interpolation=cv2.INTER_AREA)  # scaled to 512x512

            if process == "crop":
                # this is only to retrieve the crop indices
                masked_image = np.where(mask, image_data, 0)
                cropped_masked_image, indices = crop_nonzero(
                    masked_image, include_index=True)  # cropped it
                a, b, c, d = indices
                image_data = cv2.resize(
                    image_data[a:b, c:d], (512, 512), interpolation=cv2.INTER_AREA)  # scaled to 512x512

            downsampled_image = downsample(
                image_data, factor=downsampling_factor)  # downsampling image

            if jpegs:
                plt.imshow(downsampled_image, cmap="gray")
                plt.savefig(output_dir + "image" + str(i) + ".jpg")
            else:
                X[i] = downsampled_image / 255
                y[i] = data['label'] - 1

            if save_labels:
                with open('labels/labels.txt', 'a') as f:
                    f.write(str(data['label']))
        else:
            print(filename, i, f"{filename} not .mat")

    X = np.stack(X, axis=0)
    y = np.reshape(y, (-1, 1))

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=str, default='uncrop',
                        help='specify the process to be done. i.e, uncrop, crop, segment')
    parser.add_argument('--input_path', type=str,
                        default="../data/test", help='directory of input data')
    parser.add_argument('--output_path', type=str,
                        default="jpegs/uncropped32/", help='directory of output data')
    args = parser.parse_args()

    process = args.process
    input_dir = args.input_path
    output_dir = args.output_path

    X, y = load_data(input_dir, process=process,
                     downsampling_factor=4, output_dir=output_dir)
    print(f"X shape: {X.shape}, Y shape: {y.shape}")

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(10):
        axes[i].imshow(X[i], cmap="gray")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.show()
