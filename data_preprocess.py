from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import cv2
import warnings

warnings.filterwarnings("ignore")

# COLOR MAP
one = [128, 128, 128]
two = [128, 0, 0]
three = [0, 128, 192]
four = [102, 204, 0]
five = [204, 0, 204]
six = [0, 255, 255]
COLOR_DICT = np.array([one, two, three, four, five, six])

class DataPreprocess:
    def __init__(self, train_path=None, image_folder=None, label_folder=None,
                 valid_path=None, valid_image_folder =None, valid_label_folder = None,
                 test_path=None, save_path=None,
                 img_rows=1024, img_cols=1024,
                 flag_multi_class=True,
                 num_classes = 6):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.train_path = train_path
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.valid_path = valid_path
        self.valid_image_folder = valid_image_folder
        self.valid_label_folder = valid_label_folder
        self.test_path = test_path
        self.save_path = save_path
        # parameter for Keras class ImageDataGenerator
        self.data_gen_args = dict(rotation_range=0.2,
                                  width_shift_range=0.05,
                                  height_shift_range=0.05,
                                  shear_range=0.05,
                                  zoom_range=0.05,
                                  vertical_flip=True,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
        self.image_color_mode = "grayscale"
        self.label_color_mode = "grayscale"
        self.flag_multi_class = flag_multi_class
        self.num_class = num_classes
        self.target_size = (256, 256)
        self.img_type = 'png'

    def adjustData(self, img, label):
        # multi-class segmentation
        if (self.flag_multi_class):
            img = img / 255.
            # label in label_generator will have shape like (8, 256, 256, 1) if grayscale
            # label[:, :, :, 0] will discard the last axis, meaning (8, 256, 256)
            label = label[:, :, :, 0] if (len(label.shape) == 4) else label[:, :, 0]
            new_label = np.zeros(label.shape + (self.num_class,))   # (batch_size, H, W, num_class)
            # by default, pixel value in GT label must be between [0; num_class-1]
            for i in range(self.num_class):
                new_label[label == i, i] = 1
            label = new_label   # each pixel in new label is a vector of 0 and 1, ex: [0, 0, 0, 0, 0, 1]
        # binary segmentation
        elif (np.max(img) > 1):
            img = img / 255.
            label = label / 255.
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
        return (img, label)

    def trainGenerator(self, batch_size,
                       image_save_prefix="image",
                       label_save_prefix="label",
                       save_to_dir=None,
                       seed=7):
        """
        Generate image and label at the same time
        Use the same seed for image_datagen and label_datagen to ensure the transformation for image and label are the same
        If you want to visualize the results of generator, set save_to_dir = "your path"
        """

        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.train_path,
            classes=[self.image_folder],
            class_mode=None,
            color_mode=self.image_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            shuffle=True,
            seed=seed)
        label_generator = label_datagen.flow_from_directory(
            self.train_path,
            classes=[self.label_folder],
            class_mode=None,
            color_mode=self.label_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=label_save_prefix,
            shuffle=True,
            seed=seed)

        # for i in image_generator:
        #     idx = (image_generator.batch_index - 1) * image_generator.batch_size
        #     print(image_generator.filenames[idx: idx + image_generator.batch_size])
        #     print(label_generator.filenames[idx: idx + label_generator.batch_size])

        train_generator = zip(image_generator, label_generator)
        for (img, label) in train_generator:
            img, label = self.adjustData(img, label)
            yield (img, label)

    def validGenerator(self, batch_size, seed=7):
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.valid_path,
            classes=[self.valid_image_folder],
            class_mode=None,
            color_mode=self.image_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed)
        label_generator = label_datagen.flow_from_directory(
            self.valid_path,
            classes=[self.valid_label_folder],
            class_mode=None,
            color_mode=self.label_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed)

        valid_generator = zip(image_generator, label_generator)
        for (img, label) in valid_generator:
            img, label = self.adjustData(img, label)
            yield (img, label)

    def testGenerator(self):
        filenames = os.listdir(self.test_path)
        for filename in filenames:
            img = io.imread(os.path.join(self.test_path, filename), as_gray=True)
            img = img / 255.
            img = trans.resize(img, self.target_size, mode='constant')
            img = np.reshape(img, img.shape + (1,)) if (not self.flag_multi_class) else img  # binary or multi-class
            img = np.reshape(img, (1,) + img.shape)
            yield img

    def saveTestResult(self, npyfile, size, name, threshold=127):
        """
        Save prediction result

        :param npyfile: prediction result (numpy arrays of predictions)
        :param size: size to save
        :param name: name to save
        :param threshold: value to distinguish background pixel and object pixel in binary segmentation case
        """
        for _, item in enumerate(npyfile):  # index, image
            img = item  # (H, W, num_class)
            img_std = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            if self.flag_multi_class:   # multi-class segmentation
                for row in range(len(img)):
                    for col in range(len(img[row])):
                        # get a class (prediction) for each pixel
                        # here argmax returns the index ([0; num_class-1]) of the maximum value along the depth axis
                        num = np.argmax(img[row][col])

                        # paint the pixel
                        img_std[row][col] = COLOR_DICT[num]

            # binary segmentation
            else:
                for row in range(len(img)):
                    for col in range(len(img[row])):
                        num = img[row][col]
                        if num < (threshold/255.0):     # (0.5)
                            img_std[row][col] = one
                        else:
                            img_std[row][col] = two
            img_std = cv2.resize(img_std, size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.save_path, ("%s_predict." + self.img_type) % (name)), img_std)