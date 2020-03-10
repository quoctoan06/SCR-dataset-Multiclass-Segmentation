import os
import shutil
import cv2
import numpy as np
from skimage import io, exposure

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

        # resize image to 256 x 256
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

        # save image, here filename[:-4] means get the real name without extension (.IMG)
        io.imsave(output_path + '/' + filename[:-4] + '_image.png', img)
        print('Lung', i, filename)

def make_masks(raw_image_path, mask_parent_path, save_path):
    """
    Preprocess mask

    :param raw_image_path: path to the JSRT image folder
    :param mask_parent_path: path to parent folder containing mask folders of classes
    :param save_path: path to save the masks
    """

    seg_mask_folders = ["left_lung", 'right_lung', 'left_clavicle', 'right_clavicle', 'heart']
    for i, filename in enumerate(os.listdir(raw_image_path)):
        # read masks in one mask folder
        mask_path = mask_parent_path + '/' + seg_mask_folders[0] + '/' + filename[:-4] + '.gif'
        mask = np.array(io.imread(mask_path))
        mask[mask > 0] = 1

        # combine masks
        for j in range(len(seg_mask_folders[1:])):
            temp = np.array(io.imread(mask_parent_path + '/' + seg_mask_folders[j+1] + '/' + filename[:-4] + '.gif'))
            mask[temp > 0] = (j+2)

        # resize to 256 x 256
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_CUBIC)

        # save mask
        io.imsave(save_path + '/' + filename[:-4] + '_mask.png', mask)
        print('Mask', i, filename)

if __name__ == '__main__':
    all_247_images_path = '/home/quoctoan/PycharmProjects/Lung_Segmentation/data/All247images/'
    mask_parent_folder_path = '/home/quoctoan/PycharmProjects/Lung_Segmentation/data/Masks/'
    train_image_path = '/home/quoctoan/PycharmProjects/Lung_Segmentation/data/train_image_and_label/image/'
    train_mask_path = '/home/quoctoan/PycharmProjects/Lung_Segmentation/data/train_image_and_label/label/'

    val_image_path = '/home/quoctoan/PycharmProjects/Lung_Segmentation/data/val_image_and_label/image/'
    val_mask_path = '/home/quoctoan/PycharmProjects/Lung_Segmentation/data/val_image_and_label/label/'
    test_image_path = '/home/quoctoan/PycharmProjects/Lung_Segmentation/data/test_image_and_label/image/'
    test_mask_path = '/home/quoctoan/PycharmProjects/Lung_Segmentation/data/test_image_and_label/label/'

    # make_lungs(all_247_images_path, train_image_path)
    # make_masks(all_247_images_path, mask_parent_folder_path, train_mask_path)

    # Move the first 24 pairs of image and mask from train folder to val folder (10% of total data)
    images = os.listdir(train_image_path)
    masks = os.listdir(train_mask_path)
    images.sort()
    masks.sort()
    count_val = 1
    for image, mask in zip(images, masks):
        if image.endswith('.png') and mask.endswith('.png') and count_val <= 24:
            shutil.move(train_image_path + '/' + image, val_image_path + '/' + image)
            shutil.move(train_mask_path + '/' + mask, val_mask_path + '/' + mask)
            count_val += 1
    print("Completed creating validation dataset")

    # Move the first 24 pairs of image and mask from train folder to test folder (10% of total data)
    images = os.listdir(train_image_path)
    masks = os.listdir(train_mask_path)
    images.sort()
    masks.sort()
    count_test = 1
    for image, mask in zip(images, masks):
        if image.endswith('.png') and mask.endswith('.png') and count_test <= 24:
            shutil.move(train_image_path + '/' + image, test_image_path + '/' + image)
            shutil.move(train_mask_path + '/' + mask, test_mask_path + '/' + mask)
            count_test += 1
    print("Completed creating test dataset")