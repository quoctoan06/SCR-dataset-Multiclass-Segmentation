import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap

def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label!')
    colormap = create_pascal_label_colormap()
    if np.max(label) > len(colormap):
        raise ValueError('Label value is too large!')
    return colormap[label]

def visualize_segmentation(image, seg_map):
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.figure()
    plt.imshow(seg_image)
    plt.imshow(image, alpha=0.5)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':

    test_path = "/home/quoctoan/PycharmProjects/Lung_Segmentation/data/test_image_and_label/image/"
    predict_path = "/home/quoctoan/PycharmProjects/Lung_Segmentation/data/test_image_and_label/result/"

    for filename in os.listdir(test_path):
        img_path = os.path.join(test_path, filename)
        prediction_path = os.path.join(predict_path, filename.split('.')[0] + "_predict.png")
        img = cv2.imread(img_path, 1)

        # image read by OpenCV has colors in BGR order
        # so we must reverse the color channel to RGB
        img = img[:, :, ::-1]

        seg_map = cv2.imread(prediction_path, 0)
        visualize_segmentation(img, seg_map)
