import os
import numpy as np
from skimage import io, exposure
import csv

def make_lungs(input_path, output_path):
    """
    Preprocess JSRT raw image data

    :param input_path: path to the JSRT image folder
    :param output_path: path to save the preprocessed JSRT image
    """

    for i, filename in enumerate(os.listdir(input_path)):
        try:
            # read .IMG files (binary files), then reshape
            # max value in each binary file is 4095, min value is 0 so we normalize the values to [0,1]
            img = 1.0 - np.fromfile(input_path + '/' + filename, dtype='>u2').reshape((2048, 2048)) * 1. / 4096
        except:
            pass

        # do histogram equalization to improve contrast in images
        img = exposure.equalize_hist(img)

        # save image, here filename[:-4] means get the real name without extension (.IMG)
        io.imsave(output_path + '/' + filename[:-4] + '.png', img)
        print('Lung', i, filename)

def make_masks(raw_image_path, left_lung_path, right_lung_path, output_path):
    """
    Preprocess JSRT label image data

    :param raw_image_path: path to the JSRT image folder
    :param left_lung_path: path to the JSRT left lung label image folder
    :param right_lung_path: path to the JSRT right lung label image folder
    :param output_path: path to save the preprocessed JSRT label image
    """

    for i, filename in enumerate(os.listdir(raw_image_path)):
        # read label images
        left = io.imread(left_lung_path + '/' + filename[:-4] + '.gif')
        right = io.imread(right_lung_path + '/' + filename[:-4] + '.gif')

        # combine left and right lung to one image and save
        io.imsave(output_path + '/' + filename[:-4] + 'msk.png', np.clip(left + right, 0, 255))
        print('Mask', i, filename)

def create_csv_file(image_and_mask_path, save_file_path, filename):
    """
    Create csv file containing raw image filenames and mask filenames

    :param image_and_mask_path: Path to folder containing raw image files and masks
    :param save_file_path: Path to save csv file
    :param filename: Name of csv file
    """

    list_lung_name, list_mask_name = [], []

    for root, dirs, files in os.walk(image_and_mask_path):
        files.sort()
        for f in files:
            if "msk" not in f:
                list_lung_name.append(f)
            else:
                list_mask_name.append(f)

    with open(save_file_path + '/' + filename + '.csv', 'w', newline='') as csvfile:
        fieldnames = ['Image', 'Mask']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, j in zip(list_lung_name, list_mask_name):
            writer.writerow({'Image': i, 'Mask': j})