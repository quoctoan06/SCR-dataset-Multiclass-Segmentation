"""
Lung Segmentation
    # Test and evaluate trained model
    Usage:
    python3 evaluate.py --test_output_path=/home/quoctoan/PycharmProjects/Lung_Segmentation/data/test_image_and_label \
    --save_csv_path=/home/quoctoan/PycharmProjects/Lung_Segmentation \
    --checkpoint_path=/home/quoctoan/PycharmProjects/Lung_Segmentation/model.100.hdf5
"""

import numpy as np
import pandas as pd
import argparse
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, color, exposure, img_as_float, transform
from matplotlib import pyplot as plt

from src.load_data import loadDataJSRT
from src.preprocess_data import create_csv_file

# define command line arguments
argparser = argparse.ArgumentParser(description='Evaluate Lung Segmentation Model')

argparser.add_argument(
    '-teop',
    '--test_output_path',
    help='test_output_path - path to save the preprocessed JSRT test label image'
)

argparser.add_argument(
    '-scp',
    '--save_csv_path',
    help='save_csv_path - path to save csv file (contain image filenames and mask filenames)'
)

argparser.add_argument(
    '-cp',
    '--checkpoint_path',
    help='checkpoint_path - path to the saved checkpoint model'
)


def IoU(y_true, y_pred):
    """
    Returns Intersection over Union score for ground truth and predicted masks.

    :param y_true: numpy array, ground truth mask (GT)
    :param y_pred: numpy array, predicted mask
    """

    assert y_true.dtype == bool and y_pred.dtype == bool
    # flatten() returns a copy of the array collapsed into one dimension.
    # Ex: ([[1,2], [3,4]]) --> ([1, 2, 3, 4])
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """
    Returns Dice Similarity Coefficient for ground truth and predicted masks.

    :param y_true: numpy array, ground truth mask (GT)
    :param y_pred: numpy array, predicted mask
    """

    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """
    Returns image with GT lung field outlined with red border,
    and predicted lung field filled with blue.

    :param img: raw image
    :param gt: ground truth image
    :param mask: predicted mask
    """

    # prepare background
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))

    # get border from ground truth image
    boundary = morphology.dilation(gt, morphology.disk(3)) ^ gt

    # draw predicted mask onto the background with blue color [0, 0, 1]
    color_mask[mask == 1] = [0, 0, 1]

    # draw ground truth border onto the background with red color [1, 0, 0]
    color_mask[boundary == 1] = [1, 0, 0]

    # here, raw image (height, width, channel) has channel = 1
    # dstack returns an image (height, width, channel) with channel = 3
    img_color = np.dstack((img, img, img))

    # convert RGB image to HSV image
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # get the Hue (color) channel from color_mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]

    # get the Saturation channel from color_mask and multiply with a factor
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    # convert HSV to RGB
    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """
    Morphologically removes small (less than size) connected regions of 0s or 1s.
    """

    # remove objects smaller than the specified size
    img = morphology.remove_small_objects(img, size)

    # remove contiguous holes smaller than the specified size
    img = morphology.remove_small_holes(img, size)

    return img

def _main_(args):
    """
    :param args: command line arguments
    """

    # Save test image filenames and respective mask filenames to CSV file
    create_csv_file(args.test_output_path, args.save_csv_path, 'idx_test')

    # Path to csv-file. File should contain X-ray raw image filenames as first column,
    # and mask filenames as second column.
    csv_path = args.save_csv_path + '/' + 'idx_test.csv'

    # Path to the folder with images. Images will be read from path + path_from_csv
    path = args.test_output_path + '/'

    df = pd.read_csv(csv_path)

    # Load test data
    im_shape = (256, 256)
    X, y = loadDataJSRT(df, path, im_shape)

    # Number of test samples
    n_test = X.shape[0]

    # Input shape
    inp_shape = X[0].shape

    # Load model
    model_path = args.checkpoint_path
    UNet = load_model(model_path)

    # For inference standard Keras ImageGenerator can be used
    test_gen = ImageDataGenerator(rescale=1.)

    # Initialize arrays to contain IoU and Dice value
    ious = np.zeros(n_test)
    dices = np.zeros(n_test)

    ################################### Predict and Plot ############################################

    gts, prs = [], []   # ground truths, predicts
    i = 0
    plt.figure(figsize=(10, 10))

    # xx is numpy array of image data, yy is numpy array of corresponding label
    for xx, yy in test_gen.flow(X, y, batch_size=1):
        # shrink image's intensity level
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))

        # predict and reshape width and height of predicted mask
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])

        # reshape width and height of ground truth mask
        mask = yy[..., 0].reshape(inp_shape[:2])

        # convert to array of True/False
        gt = mask > 0.5
        pr = pred > 0.5

        # remove small regions from predicted mask
        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

        # io.imsave('{}'.format(df.iloc[i].path), masked(img, gt, pr, 1))

        gts.append(gt)
        prs.append(pr)

        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        print (df.iloc[i][0], ious[i], dices[i])

        if i < 12:
            plt.subplot(4, 3, 3 * i + 1)
            plt.title('Raw ' + df.iloc[i][0])
            plt.axis('off')
            plt.imshow(img, cmap='gray')

            plt.subplot(4, 3, 3 * i + 2)
            plt.title('IoU = {:.4f}'.format(ious[i]))
            plt.axis('off')
            plt.imshow(masked(img, gt, pr, 1))

            plt.subplot(4, 3, 3 * i + 3)
            plt.title('Prediction')
            plt.axis('off')
            plt.imshow(pred, cmap='jet')

            # plt.subplot(12, 4, 4 * i + 4)
            # plt.title('Difference')
            # plt.axis('off')
            # plt.imshow(np.dstack((pr.astype(np.int8), gt.astype(np.int8), pr.astype(np.int8))))

        i += 1
        if i == n_test:
            break

    ##########################################################################################
    print('Mean IoU: ', ious.mean())
    print('Mean Dice: ', dices.mean())
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.show()


if __name__ == '__main__':
    # parse the arguments
    args = argparser.parse_args()
    _main_(args)