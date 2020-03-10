from model.unet_model import *
from data_preprocess import *
from train import mean_iou

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras import backend as K
import numpy as np
import cv2
import os

def image_normalized(file_path):
    img = cv2.imread(file_path, 0)
    img_shape = img.shape
    image_size = (img_shape[1], img_shape[0])
    img_standard = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_new = img_standard
    img_new = np.asarray([img_new / 255.])
    return img_new, image_size

def iou_one_label(y_true, y_pred, label: int):
    sess = tf.Session()
    y_true = K.equal(K.argmax(y_true), label)
    y_pred = K.equal(K.argmax(y_pred), label)

    # convert tensor to numpy array
    y_true = y_true.eval(session=sess)
    y_pred = y_pred.eval(session=sess)

    # flatten the array
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    # calculate the |intersection| (AND) of the labels
    intersection = np.logical_and(y_true_f, y_pred_f).sum()

    # calculate the |union| (OR) of the labels
    union = np.logical_or(y_true_f, y_pred_f).sum()

    # plus 1 to avoid dividing by 0
    return (intersection + 1) * 1. / (union + 1)

def mean_iou_one_label(y_true_array, y_pred_array, label: int):
    """
    Return the Intersection over Union (IoU) score over a specific number of images
    for a specific label

    :param y_true_array: array of the expected y values as a one-hot
    :param y_pred_array: array of the predicted y values as a one-hot or softmax output
    :return: the scalar IoU value (mean over a number of images)
    """
    # initialize a variable to store total IoU in
    total_iou = 0.
    # iterate over images to calculate IoU
    for i in range(len(y_true_array)):
        total_iou = total_iou + iou_one_label(y_true_array[i], y_pred_array[i], label)
    # divide total IoU by number of images to get mean IoU
    return total_iou / len(y_true_array)

if __name__ == '__main__':

    # path to test images
    test_image_path = "/home/quoctoan/PycharmProjects/Lung_Segmentation/data/test_image_and_label/image/"
    # path to test labels
    test_label_path = "/home/quoctoan/PycharmProjects/Lung_Segmentation/data/test_image_and_label/label/"
    # path to save the predictions
    save_path = "/home/quoctoan/PycharmProjects/Lung_Segmentation/data/test_image_and_label/result/"

    dp = DataPreprocess(test_path=test_image_path, save_path=save_path, flag_multi_class=True, num_classes=6)

    # load model
    model = load_model('SCR_model_unet_150_epochs.hdf5', custom_objects={'mean_iou': mean_iou})

    # predict and save prediction
    y_pred_array = []
    for name in os.listdir(test_image_path):
        image_path = os.path.join(test_image_path, name)
        img, img_size = image_normalized(image_path)
        img = np.expand_dims(img, axis=-1)
        prediction = model.predict(img)
        for _, item in enumerate([prediction[0]]):
            y_pred_array.append(item)
        dp.saveTestResult([prediction[0]], img_size, name.split('.')[0])
        print("Save prediction for image %s" % name)

    # preprocess label
    y_true_array = []
    for label_name in os.listdir(test_label_path):
        label_path = os.path.join(test_label_path, label_name)
        label = cv2.imread(label_path, 0)
        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_CUBIC)
        new_label = np.zeros(label.shape + (6,))    # num_class = 6
        for i in range(6):
            new_label[label == i, i] = 1
        label = new_label
        y_true_array.append(label)

    # calculate Mean IoU for each label over test set
    label_dict = {"left_lung": 1, 'right_lung': 2, 'left_clavicle': 3, 'right_clavicle': 4, 'heart': 5}
    for key, value in label_dict.items():
        meanIoU = mean_iou_one_label(y_true_array, y_pred_array, value)
        print("Mean IoU of %s is %.3f" % (key, meanIoU))
