from skimage import data, filters, morphology, feature
from matplotlib import pyplot as plt
import numpy as np
import os

INPUT_DIR_PATH = 'input'

THRESHOLD_VALUE = 4/5
CANNY_SIGMA_VALUE = 10

PLOT_SHAPE = (2,2)


def list_file_paths(dir_path):
    return [os.path.join(dir_path, file) for file in os.listdir(dir_path)]

def get_input_data(file_paths):
    return [data.imread(file) for file in file_paths], [data.imread(file, as_grey=True) for file in file_paths]

def perform_image_computations(input, input_grey):
    out = []
    for input_item, input_grey_item in zip(input, input_grey):
        zeros = np.zeros_like(input_grey_item)
        r, g, b = zeros, zeros, zeros
        g = np.zeros_like(input_grey_item)
        b = np.zeros_like(input_grey_item)
        threshb = np.zeros_like(input_grey_item)

        for j, input_item_row in enumerate(input_item):
            for k, input_item_cell in enumerate(input_item_row):
                r[j][k], g[j][k], b[j][k] = input_item_cell
                threshb[j][k] = b[j][k] > r[j][k]*THRESHOLD_VALUE
        sobel = filters.sobel(threshb)
        canny = feature.canny(threshb, sigma=CANNY_SIGMA_VALUE)
        out.append([threshb, sobel, canny])

    return out


def plot_images(input, out):
    for input_item,out_item in zip(input, out):
        x_size, y_size = PLOT_SHAPE
        plt.subplot(x_size, y_size, 1)
        plt.imshow(input_item)

        for index, img in enumerate(out_item):
            plt.subplot(x_size, y_size, index+2)
            plt.imshow(img)
            plt.gray()

        plt.show()

file_paths = list_file_paths(INPUT_DIR_PATH)
input, input_grey = get_input_data(file_paths)

out = perform_image_computations(input, input_grey)

plot_images(input, out)



