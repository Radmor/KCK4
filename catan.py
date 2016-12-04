from skimage import data, filters, morphology, feature, draw, measure, color
from matplotlib import pyplot as plt
import numpy as np
import os
import copy
from scipy.ndimage.morphology import binary_fill_holes
import cv2
from sklearn.cluster import KMeans

INPUT_DIR_PATH = 'input'


THRESHOLD_VALUE = 4/5
CANNY_SIGMA_VALUE = 10
RADIUS_MAGNIFIER = 0.95

KMEANS_CENTROIDS = np.array([[0.2, 0.2, 0.2], [0.2, 0.5, 0.5], [0.2, 0.4, 0.8], [0.4, 0.5, 0.7], [0.3, 0.6, 0.9]])

# PLOT_SHAPE = (2, 2)


def list_file_paths(dir_path):
    return [os.path.join(dir_path, file) for file in os.listdir(dir_path)]


def get_input_data(file_paths):
    # return [cv2.imread(file, 1) for file in file_paths], [cv2.imread(file, 0) for file in file_paths]


    imgs = [cv2.imread(file, 1) for file in file_paths]
    rs = [500.0 / img.shape[1] for img in imgs]
    dims = [(500, int(img.shape[0] * r)) for img, r in zip(imgs, rs)]
    return [cv2.resize(image, dim, interpolation=cv2.INTER_AREA) for image, dim in zip(imgs, dims)], [cv2.resize(
        cv2.imread(file, 0), dim, interpolation=cv2.INTER_AREA) for file, dim in zip(file_paths, dims)]


def perform_image_computations(input, input_grey):
    out = []
    for input_item, input_grey_item in zip(input, input_grey):
        r = np.zeros_like(input_grey_item)
        g = np.zeros_like(input_grey_item)
        b = np.zeros_like(input_grey_item)
        threshb = np.zeros_like(input_grey_item)
        threshb2 = np.zeros_like(input_grey_item)
        hsv = color.rgb2hsv(input_item)
        bmax = 0

        # for j, input_item_row in enumerate(input_item):
        #     for k, input_item_cell in enumerate(input_item_row):
                # b[j][k], g[j][k], r[j][k] = input_item_cell
                # threshb[j][k] = b[j][k] > r[j][k]*THRESHOLD_VALUE

                # r[j][k], g[j][k], b[j][k] = input_item_cell
                # if b[j][k] > bmax:
                #     bmax = b[j][k]

                # if 120/360 <= input_item_cell[0] <= 340/360 and input_item_cell[1] > 0.1:
                #     threshb[j][k] = 1.0
                #     # print(input_item_cell)
                # else:
                #     threshb[j][k] = 0.0
        # print(bmax)
        for j, input_item_row in enumerate(input_item):
            for k, input_item_cell in enumerate(input_item_row):
                # print(input_grey_item[j][k])
                b[j][k], g[j][k], r[j][k] = input_item_cell
                threshb[j][k] = b[j][k] > r[j][k]*THRESHOLD_VALUE and 200 > input_grey_item[j][k] > 50
        # filled = binary_fill_holes(threshb)

        # filling holes
        # Copy the thresholded image.
        im_floodfill = threshb.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = threshb.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255);

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        hex = threshb | im_floodfill_inv



        # for i in range(0, 20):
        #     filled = morphology.binary_closing(filled)
        #     filled = morphology.binary_opening(filled)

        # blurred = cv2.GaussianBlur(filled, (5, 5), 0)

        # c = cv2.findContours(filled.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # peri = cv2.arcLength(c, True)
        # approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # print(approx)
        # print(threshb)
        # print(input_grey_item)

        #
        # cv2.imshow('img', input_item)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow('img', hexclr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # # plt.imshow(filled)
        # plt.gray()
        # plt.show()
        sobel = filters.sobel(threshb)
        canny = feature.canny(threshb, sigma=CANNY_SIGMA_VALUE)
        # out.append([threshb, sobel, canny])
        out.append(hex)

    return out


# expects list of 6 tuples with vertex coords
# warning: clears input list! use copy.copy to ensure safety
def get_hex_coords(vexlist):
    if len(vexlist) != 6:
        return -1

    vexlist = [el[0] for el in vexlist]

    # sort vexlist
    # vexx = [coord[0] for coord in vexlist]
    # sortx = sorted(vexx)
    #
    # nodups = len(vexx) == len(set(vexx))
    # sortvex = []
    # if nodups:
    #     for x in sortx:
    #         el = [item for item in vexlist if item[0] == x]
    #         vexlist.remove(el[0])
    #         sortvex.append(el[0])
    #
    # else:
    #     vexy = [coord[1] for coord in vexlist]
    #     sorty = sorted(vexy)
    #     for x in sortx:
    #         ellist = []
    #         for y in sorty:
    #             el = [item for item in vexlist if item[0] == x and item[1] == y]
    #             if len(el) > 0:
    #                 ellist.append(el[0])
    #         vexlist.remove(ellist[0])
    #         sortvex.append(ellist[0])
    #
    # botleft = sortvex[0]
    # botright = sortvex[1]
    # left = sortvex[2]
    # right = sortvex[3]
    # topleft = sortvex[4]
    # topright = sortvex[5]

    # no sorting, since opencv function gives vertices in order around the contour

    left = vexlist[0]
    topleft = vexlist[1]
    topright = vexlist[2]
    right = vexlist[3]
    botright = vexlist[4]
    botleft = vexlist[5]

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
    if hexinfo == -1:
        return [(0, 0, 0) for i in range(0, 19)]

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


def get_corners_contours_opencv(input):
    vertices = []
    contours = []
    for input_item in input:
        # getting contours
        cnts = cv2.findContours(input_item.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts[1]:
            # cv2.drawContours(input_item, [c], -1, (0, 255, 0), 2)
            # hexclr = cv2.cvtColor(hex, cv2.COLOR_GRAY2RGB)
            # cv2.drawContours(hexclr, [c], -1, (0, 255, 0), 2)
            peri = cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, 0.04 * peri, True)

        vertices.append(corners.tolist())
        contours.append(cnts)
    return vertices, contours


def colors_classification(colors):
    classified = []
    for n, col in enumerate(colors):
        add1 = False
        for cl in classified:
            add2 = True
            for c in cl:
                if abs(c[1][0] - col[0]) < 0.2 and abs(c[1][1] - col[1]) < 0.2 and abs(c[1][2] - col[2]) < 0.2:
                    continue
                else:
                    add2 = False
                    break
            if add2:
                add1 = True
                cl.append((n, col))
                break
        if not add1:
            classified.append([(n, col)])
    return classified


def colors_classification_kmeans(colors):
    # est = KMeans(n_clusters=5)
    est = KMeans(n_clusters=5, n_init=1, init=KMEANS_CENTROIDS)
    est.fit(colors)
    # print(est.cluster_centers_)
    return est.labels_


def compute_colors_classification(colors):
    return [colors_classification_kmeans(colors_item) for colors_item in colors]


def get_gameboard_corners(input):
    for input_item, second in input:
        # coords = feature.corner_peaks(feature.corner_harris(input_item,k=0.24), min_distance=50,)
        # coords = feature.corner_peaks(feature.corner_kitchen_rosenfeld(input_item,cval=5), min_distance=50,)
        coords = feature.corner_peaks(feature.corner_moravec(input_item,), min_distance=50,)


        # coords_subpix = feature.corner_subpix(input_item, coords, window_size=20)
        # print(coords)
        # print(coords_subpix)

        #contours = measure.find_contours(input_item, 0.8)

        # Display the image and plot all contours found
        # fig, ax = plt.subplots()
        # ax.imshow(input_item, interpolation='nearest', cmap=plt.cm.gray)
        #
        # for n, contour in enumerate(contours):
        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


        fig, ax = plt.subplots()
        ax.imshow(input_item, interpolation='nearest', cmap=plt.cm.gray)
        # ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
        # ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
        ax.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)
        plt.show()


def get_gameboard_corners(input):
    for input_item, second in input:
        # coords = feature.corner_peaks(feature.corner_harris(input_item,k=0.24), min_distance=50,)
        # coords = feature.corner_peaks(feature.corner_kitchen_rosenfeld(input_item,cval=5), min_distance=50,)
        # coords = feature.corner_peaks(feature.corner_moravec(input_item,), min_distance=50,)


        # coords_subpix = feature.corner_subpix(input_item, coords, window_size=20)
        # print(coords)
        # print(coords_subpix)

        #contours = measure.find_contours(input_item, 0.8)

        # Display the image and plot all contours found
        # fig, ax = plt.subplots()
        # ax.imshow(input_item, interpolation='nearest', cmap=plt.cm.gray)
        #
        # for n, contour in enumerate(contours):
        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


        # fig, ax = plt.subplots()
        # ax.imshow(input_item, interpolation='nearest', cmap=plt.cm.gray)
        # # ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
        # # ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
        # ax.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)
        # plt.show()

        return feature.corner_peaks(feature.corner_moravec(input_item,), min_distance=50,)


def plot_images(input, filled, hexinfo, vertices, contours, colors, classification):
    for input_item, filled_item, hexinfo_item, vertices_item, contour_item, color_item, classification_item in \
            zip(input, filled, hexinfo, vertices, contours, colors, classification):
        image = input_item.copy()
        heximage = cv2.cvtColor(filled_item, cv2.COLOR_GRAY2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for c in contour_item[1]:
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        if hexinfo_item != -1:
            for v, col, cl in zip(hexinfo_item[0], color_item, classification_item):
                vmod = tuple([int(v[0]), int(v[1])])
                cv2.circle(image, vmod, int(hexinfo_item[1]), (255, 255, 255))
                colmod = tuple([255*val for val in col])
                cv2.circle(heximage, vmod, 10, colmod, 15)
                cv2.putText(image, str(cl), (int(v[0]) - 7, int(v[1]) + 7), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # for n, cl in enumerate(classification_item):
            #     for p in cl:
            #         cv2.putText(image, str(n), (int(hexinfo_item[0][p[0]][0]) - 7, int(hexinfo_item[0][p[0]][1]) + 7), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        for i, v in enumerate(vertices_item):
            cv2.circle(image, tuple(v[0]), 10, (0, 0, 100 + 30*i), 1)
        cv2.imshow('img', image)
        cv2.imshow('img2', heximage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # x_size, y_size = PLOT_SHAPE
        # ax = plt.subplot(x_size, y_size, 1)
        # plt.imshow(input_item)

        # for el in hexinfo_item[0]:
        #     circle = plt.Circle((el[0], el[1]), radius=hexinfo_item[1], alpha=0.4, color='white')
        #     ax.add_artist(circle)

        # ax = plt.subplot(x_size, y_size, 2)
        # plt.imshow(out_item[0])
        # for el, col in zip(hexinfo_item[0], colors_item):
        #     circle = plt.Circle((el[0], el[1]), radius=hexinfo_item[1], color=col)
        #     ax.add_artist(circle)

        # ax.plot(corners[:, 1], corners[:, 0], '+r', markersize=15)

        # for index, img in enumerate(out_item):
        #
        #     ax = plt.subplot(x_size, y_size, index+3)
        #     # ax.plot(corners[:, 1], corners[:, 0], '+r', markersize=15)
        #     plt.imshow(img)
        #     plt.gray()
        #
        # plt.show()

file_paths = list_file_paths(INPUT_DIR_PATH)
input, input_grey = get_input_data(file_paths)
filled = perform_image_computations(input, input_grey)
vertices, contours = get_corners_contours_opencv(filled)
hexinfo = compute_vertices(vertices)
colors = compute_hex_colors(hexinfo, input, input_grey)
classification  = compute_colors_classification(colors)




# corners = get_gameboard_corners(out)

plot_images(input, filled, hexinfo, vertices, contours, colors, classification)



