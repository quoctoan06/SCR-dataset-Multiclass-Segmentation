import numpy as np
from skimage import transform, io, img_as_float, exposure

def loadDataJSRT(df, path, im_shape):
    """
    Load data after preprocessed in preprocess_data.py.

    Here, data is preprocessed in the following ways:
        - Resize to im_shape (after implement equalize histogram)
        - Normalize by data set mean and std

    Resulting shape should be (n_samples, img_width, img_height, 1).
    Data frame should contain paths to images and masks as two columns (relative to 'path').

    :param df: data frame containing raw image names and label names
    :param path: path to folder containing preprocessed raw image and label
    :param im_shape: image shape
    """

    X, y = [], []
    for i, item in df.iterrows():
        # raw image
        img = io.imread(path + item[0])
        img = transform.resize(img, im_shape)
        img = np.expand_dims(img, -1)   # expand 1 dimension at the end. Ex: (2,2) --> (2,2,1)

        # label (ground truth - GT)
        mask = io.imread(path + item[1])
        mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)

        # save to list
        X.append(img)
        y.append(mask)

    # convert to numpy array
    X = np.array(X)
    y = np.array(y)

    # normalize raw image
    X -= X.mean()
    X /= X.std()

    print ('### Data Loaded ###')
    print ('\t{}'.format(path))
    print ('\t{}\t{}'.format(X.shape, y.shape))
    print ('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print ('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))

    return X, y