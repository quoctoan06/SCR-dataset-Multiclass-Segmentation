"""
Lung Segmentation
    # Train a new model using JSRT data
    Usage:
    python3 train.py --raw_image_path=/home/quoctoan/PycharmProjects/Lung_Segmentation/data/All247images \
    --left_lung_path=/home/quoctoan/PycharmProjects/Lung_Segmentation/data/Masks/left_lung \
    --right_lung_path=/home/quoctoan/PycharmProjects/Lung_Segmentation/data/Masks/right_lung \
    --train_output_path=/home/quoctoan/PycharmProjects/Lung_Segmentation/data/train_image_and_label \
    --test_output_path=/home/quoctoan/PycharmProjects/Lung_Segmentation/data/test_image_and_label \
    --save_csv_path=/home/quoctoan/PycharmProjects/Lung_Segmentation
"""

from src.image_gen import ImageDataGenerator
from src.load_data import loadDataJSRT
from src.build_model import build_UNet2D
from src.preprocess_data import make_lungs, make_masks, create_csv_file

import argparse
import pandas as pd
import os, shutil
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint

# define command line arguments
argparser = argparse.ArgumentParser(description='Train Lung Segmentation Model')

argparser.add_argument(
    '-rip',
    '--raw_image_path',
    help='raw_image_path - path to the JSRT image folder'
)

argparser.add_argument(
    '-llp',
    '--left_lung_path',
    help='left_lung_path - path to the JSRT left lung label image folder'
)

argparser.add_argument(
    '-rlp',
    '--right_lung_path',
    help='right_lung_path - path to the JSRT right lung label image folder'
)

argparser.add_argument(
    '-trop',
    '--train_output_path',
    help='train_output_path - path to save the preprocessed JSRT training label image'
)

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

def _main_(args):
    """
    :param args: command line arguments
    """

    # Preprocess raw image and mask
    # make_lungs(args.raw_image_path, args.train_output_path)
    # make_masks(args.raw_image_path, args.left_lung_path, args.right_lung_path, args.train_output_path)

    # Move the first of 12 pairs of image and mask from train folder to test folder (5% of total data)
    # files = os.listdir(args.train_output_path)
    # files.sort()
    # count = 1
    # for f in files:
    #     if f.endswith('.png') and count <= 24:
    #         src = args.train_output_path + '/' + f
    #         dst = args.test_output_path + '/' + f
    #         shutil.move(src, dst)
    #         count += 1

    # Save training image filenames and respective mask filenames to CSV file
    # create_csv_file(args.train_output_path, args.save_csv_path, 'idx_train')

    ##########################################################################################
    # Path to csv-file. File should contain X-ray raw image filenames as first column,
    # and mask filenames as second column.
    csv_path = args.save_csv_path + '/' + 'idx_train.csv'

    # Path to the folder with images. Images will be read from path + path_from_csv
    path = args.train_output_path + '/'

    df = pd.read_csv(csv_path)

    # Shuffle rows in data frame. Random state is set for reproducibility
    # here frac=1 means get 100% of rows
    df = df.sample(frac=1, random_state=23)

    # Number of training samples
    n_train = int(len(df))

    # get all samples in training folder
    df_train = df[:n_train]

    # Load and split data
    im_shape = (256, 256)
    X, y = loadDataJSRT(df_train, path, im_shape)

    # Training set: 90%; Validation set: 5%
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=1)

    # Build model
    inp_shape = X_train[0].shape    # (256, 256, 1)
    UNet = build_UNet2D(inp_shape)
    UNet.summary()
    UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Visualize model
    plot_model(UNet, 'model.png', show_shapes=True)

    ##########################################################################################
    model_file_format = 'model.{epoch:03d}.hdf5'
    print(model_file_format)
    checkpointer = ModelCheckpoint(model_file_format, period=10)

    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)

    test_gen = ImageDataGenerator(rescale=1.)

    batch_size = 8

    # X_train.shape[0] is number of training samples
    UNet.fit_generator(train_gen.flow(X_train, y_train, batch_size),
                       steps_per_epoch=(X_train.shape[0] + batch_size - 1) // batch_size,   # floor division
                       epochs=100,
                       callbacks=[checkpointer],
                       validation_data=test_gen.flow(X_val, y_val),
                       validation_steps=(X_val.shape[0] + batch_size - 1) // batch_size
                       )

if __name__ == '__main__':
    # parse the arguments
    args = argparser.parse_args()
    _main_(args)
