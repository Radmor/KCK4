from skimage import data, filters, morphology, feature, draw
from matplotlib import pyplot as plt
import numpy as np
import os
import copy

INPUT_DIR_PATH = 'input'


THRESHOLD_VALUE = 4/5
CANNY_SIGMA_VALUE = 10
RADIUS_MAGNIFIER = 0.85

vlist = [[(510, 905), (698, 209), (1390, 0), (1950, 509), (998, 1444), (1742, 1271)],
         [(832, 160), (1520, 105), (516, 762), (1918, 697), (852, 1360), (1567, 1357)],
         [(1272, 59), (696, 386), (1824, 1181), (1878, 438), (651, 1047), (1180, 1451)]]

PLOT_SHAPE = (2, 2)


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
        # out.append([threshb, sobel, canny])
        out.append([threshb, sobel])

    return out


# expects list of 6 tuples with vertex coords
# warning: clears input list! use copy.copy to ensure safety
def get_hex_coords(vexlist):
    if len(vexlist) != 6:
        return -1

    # sort vexlist
    vexx = [coord[0] for coord in vexlist]
    sortx = sorted(vexx)

    nodups = len(vexx) == len(set(vexx))
    sortvex = []
    if nodups:
        for x in sortx:
            el = [item for item in vexlist if item[0] == x]
            vexlist.remove(el[0])
            sortvex.append(el[0])

    else:
        vexy = [coord[1] for coord in vexlist]
        sorty = sorted(vexy)
        for x in sortx:
            ellist = []
            for y in sorty:
                el = [item for item in vexlist if item[0] == x and item[1] == y]
                if len(el) > 0:
                    ellist.append(el[0])
            vexlist.remove(ellist[0])
            sortvex.append(ellist[0])

    botleft = sortvex[0]
    botright = sortvex[1]
    left = sortvex[2]
    right = sortvex[3]
    topleft = sortvex[4]
    topright = sortvex[5]

    left_to_right = (right[0] - left[0], right[1] - left[1])
    botleft_to_topright = (topright[0] - botleft[0], topright[1] - botleft[1])
    topleft_to_botright = (botright[0] - topleft[0], botright[1] - topleft[1])

    ltrhexes = [(left[0] + (0.2 + i*0.15) * left_to_right[0], left[1] + (0.2 + i*0.15) * left_to_right[1]) for i in range(0, 5)]
    blttrhexes = [(botleft[0] + (0.2 + i*0.15) * botleft_to_topright[0], botleft[1] + (0.2 + i*0.15) * botleft_to_topright[1]) for i in range(0, 5)]
    tltbrhexes = [(topleft[0] + (0.2 + i*0.15) * topleft_to_botright[0], topleft[1] + (0.2 + i*0.15) * topleft_to_botright[1]) for i in range(0, 5)]

    bothex = ((blttrhexes[0][0] + tltbrhexes[4][0])/2, (blttrhexes[0][1] + tltbrhexes[4][1])/2)
    botlefthex = ((blttrhexes[0][0] + ltrhexes[0][0])/2, (blttrhexes[0][1] + ltrhexes[0][1])/2)
    botrighthex = ((ltrhexes[4][0] + tltbrhexes[4][0])/2, (ltrhexes[4][1] + tltbrhexes[4][1])/2)
    tophex = ((blttrhexes[4][0] + tltbrhexes[0][0])/2, (blttrhexes[4][1] + tltbrhexes[0][1])/2)
    toplefthex = ((ltrhexes[0][0] + tltbrhexes[0][0])/2, (ltrhexes[0][1] + tltbrhexes[0][1])/2)
    toprighthex = ((blttrhexes[4][0] + ltrhexes[4][0])/2, (blttrhexes[4][1] + ltrhexes[4][1])/2)
    centralhex = ((ltrhexes[2][0] + blttrhexes[2][0] + tltbrhexes[2][0])/3, (ltrhexes[2][1] + blttrhexes[2][1] + tltbrhexes[2][1])/3)

    hexes = [blttrhexes[0], bothex, tltbrhexes[4], botlefthex, blttrhexes[1], tltbrhexes[3], botrighthex,
             ltrhexes[0], ltrhexes[1], centralhex, ltrhexes[3], ltrhexes[4],
             toplefthex, tltbrhexes[1], blttrhexes[3], toprighthex, tltbrhexes[0], tophex, blttrhexes[4]]

    # compute the estimated radius
    diag1 = ((right[0] - left[0]) ** 2 + (right[1] - left[1])) ** (1 / 2)
    diag2 = ((topright[0] - botleft[0]) ** 2 + (topright[1] - botleft[1])) ** (1 / 2)
    diag3 = ((botright[0] - topleft[0]) ** 2 + (botright[1] - topleft[1])) ** (1 / 2)

    diag = (diag1 + diag2 + diag3) / 3

    # return a tuple of a list of 19 tuples of hex coords and circle radius
    return hexes, (diag / 2) * 0.15 * RADIUS_MAGNIFIER


# expects input from the get_hex_coords function
def get_hex_colors(hexinfo, image, image_grey):
    colors = []

    for hex in hexinfo[0]:
        circle = np.zeros_like(image_grey)
        rr, cc = draw.circle(hex[1], hex[0], hexinfo[1])
        circle[rr, cc] = 1
        n_of_pixels = 0
        colsum = [0, 0, 0]

        for j, image_row in enumerate(image):
            for k, image_cell in enumerate(image_row):
                if circle[j][k] == 1:
                    n_of_pixels += 1
                    colsum = [colsum[i] + image_cell[i] for i in range(0, 3)]

        cols = [col/n_of_pixels for col in colsum]
        normcols = [col/255 for col in cols]
        colors.append(normcols)

    return colors


def compute_vertices(vexlist):
    # tuples of list of hex coords and the computed radius
    return [get_hex_coords(copy.copy(vertices)) for vertices in vexlist]


def compute_hex_colors(hexinfo, input, input_grey):
    return [get_hex_colors(hex, image, image_grey) for hex, image, image_grey in zip(hexinfo, input, input_grey)]


def plot_images(input, out, hexinfo, colors):
    for input_item, out_item, hexinfo_item, colors_item in zip(input, out, hexinfo, colors):
        x_size, y_size = PLOT_SHAPE
        ax = plt.subplot(x_size, y_size, 1)
        plt.imshow(input_item)
        for el in hexinfo_item[0]:
            circle = plt.Circle((el[0], el[1]), radius=hexinfo_item[1], alpha=0.4, color='white')
            ax.add_artist(circle)

        ax = plt.subplot(x_size, y_size, 2)
        plt.imshow(out_item[0])
        for el, col in zip(hexinfo_item[0], colors_item):
            circle = plt.Circle((el[0], el[1]), radius=hexinfo_item[1], color=col)
            ax.add_artist(circle)

        for index, img in enumerate(out_item):
            plt.subplot(x_size, y_size, index+3)
            plt.imshow(img)
            plt.gray()

        plt.show()

file_paths = list_file_paths(INPUT_DIR_PATH)
input, input_grey = get_input_data(file_paths)
hexinfo = compute_vertices(vlist)
colors = compute_hex_colors(hexinfo, input, input_grey)


out = perform_image_computations(input, input_grey)

plot_images(input, out, hexinfo, colors)



