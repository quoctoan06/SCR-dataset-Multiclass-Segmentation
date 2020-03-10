from model.unet_model import  *
from data_preprocess import *

import keras
from keras.callbacks import TensorBoard
from keras.optimizers import *
from keras import backend as K
import matplotlib.pyplot as plt

def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.

    :param y_true: the expected y values as a one-hot
    :param y_pred: the predicted y values as a one-hot or softmax output
    :param label: the label to return the IoU for

    :return: the IoU for the given label
    """

    # Extract the label values using the argmax operator then calculate equality of the
    # predictions and truths to the label. Remember, here argmax returns the index ([0; num_class-1])
    # of the maximum value along the depth axis. So for each label, y_true and y_pred will be a
    # (H x W x 1) tensor of 0 and 1.
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())

    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)

    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection

    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)

def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.

    :param y_true: the expected y values as a one-hot
    :param y_pred: the predicted y values as a one-hot or softmax output
    :return: the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        total_iou = total_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels

if __name__ == '__main__':

    # path to images and labels which are prepared to train
    train_path = "/home/quoctoan/PycharmProjects/Lung_Segmentation/data/train_image_and_label/"
    image_folder = "image"
    label_folder = "label"
    valid_path =  "/home/quoctoan/PycharmProjects/Lung_Segmentation/data/test_image_and_label/"
    valid_image_folder ="image"
    valid_label_folder = "label"
    log_filepath = './log'
    flag_multi_class = True
    num_classes = 6
    dp = DataPreprocess(train_path=train_path, image_folder=image_folder, label_folder=label_folder,
                        valid_path=valid_path, valid_image_folder=valid_image_folder, valid_label_folder=valid_label_folder,
                        flag_multi_class=flag_multi_class,
                        num_classes=num_classes)

    # train model
    train_data = dp.trainGenerator(batch_size=8)
    valid_data = dp.validGenerator(batch_size=8)

    model = unet(num_class=num_classes)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=["accuracy", mean_iou])
    # model.summary()

    tb_cb = TensorBoard(log_dir=log_filepath)
    model_checkpoint = keras.callbacks.ModelCheckpoint('SCR_model_unet_150_epochs.hdf5', monitor='val_acc', verbose=1, save_best_only=True)
    history = model.fit_generator(train_data,
                                  steps_per_epoch=28,
                                  epochs=150,
                                  validation_steps=10,
                                  validation_data=valid_data,
                                  callbacks=[model_checkpoint, tb_cb])

    # draw the loss and accuracy curve
    plt.figure(12, figsize=(12, 12), dpi=60)
    plt.subplot(311)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.title('Loss')
    plt.legend()

    plt.subplot(312)
    plt.plot(history.history['acc'], label='train_acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(313)
    plt.plot(history.history['mean_iou'], label='train_mean_iou')
    plt.plot(history.history['val_mean_iou'], label='val_mean_iou')
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.title('Mean IoU')
    plt.legend()

    plt.show()
