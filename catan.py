from skimage import draw
import numpy as np
import os
import copy
import cv2
from sklearn.cluster import KMeans

INPUT_DIR_PATH = 'input'


THRESHOLD_VALUE = 4/5
CANNY_SIGMA_VALUE = 10
RADIUS_MAGNIFIER = 0.95

KMEANS_CENTROIDS = np.array([[0.2, 0.2, 0.2], [0.2, 0.5, 0.5], [0.2, 0.4, 0.8], [0.4, 0.5, 0.7], [0.3, 0.6, 0.9]])
UNTRANSFORMED_HEX_VERTICES = np.array([[122, 50], [266, 50], [338, 175], [266, 300], [122, 300], [50, 175]])

# PLOT_SHAPE = (2, 2)


def list_file_paths(dir_path):
    return [os.path.join(dir_path, file) for file in os.listdir(dir_path)]


def get_input_data(file_paths):
    imgs = [cv2.imread(file, 1) for file in file_paths]
    rs = [500.0 / img.shape[1] for img in imgs]
    dims = [(500, int(img.shape[0] * r)) for img, r in zip(imgs, rs)]
    return [cv2.resize(image, dim, interpolation=cv2.INTER_AREA) for image, dim in zip(imgs, dims)], [cv2.resize(
        cv2.imread(file, 0), dim, interpolation=cv2.INTER_AREA) for file, dim in zip(file_paths, dims)]


def check_for_hexagon(thresholded):
    # Copy the thresholded image.
    im_floodfill = thresholded.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = thresholded.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    hex = thresholded | im_floodfill_inv

    hex = cv2.GaussianBlur(hex, (7, 7), 0)
    cnts = cv2.findContours(hex.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_peri = 0
    max_peri_index = 0

    for i, c in enumerate(cnts[1]):
        peri = cv2.arcLength(c, True)
        if peri > max_peri:
            max_peri = peri
            max_peri_index = i

    if len(cnts[1]) > 0:
        corners = cv2.approxPolyDP(cnts[1][max_peri_index], 0.01 * max_peri, True)
        corners_slim = [v[0] for v in corners.tolist()]
        if len(corners_slim) == 6 and cv2.isContourConvex(corners):
            return corners_slim, cnts[1][max_peri_index], hex, True
        else:
            return [], [], [], False
    else:
        return [], [], [], False


def perform_image_computations(input, input_grey):
    out = []
    corners_out = []
    contours_out = []
    for input_item, input_grey_item in zip(input, input_grey):
        r = np.zeros_like(input_grey_item)
        g = np.zeros_like(input_grey_item)
        b = np.zeros_like(input_grey_item)
        threshb = np.zeros_like(input_grey_item)
        hsv = cv2.cvtColor(input_item, cv2.COLOR_BGR2HSV)

        found = False

        for j, input_item_row in enumerate(input_item):
            for k, input_item_cell in enumerate(input_item_row):
                b[j][k], g[j][k], r[j][k] = input_item_cell
                threshb[j][k] = b[j][k] > r[j][k] * THRESHOLD_VALUE and 200 > input_grey_item[j][k] > 50

        corners, contours, hex, found = check_for_hexagon(threshb)

        for dist in range(30, 90, 10):
            if found:
                break

            for angle in range(180, 300, 10):
                if found:
                    break
                for j, input_item_row in enumerate(input_item):
                    for k, input_item_cell in enumerate(input_item_row):
                        threshb[j][k] = (angle - dist / 2) / 2 < hsv[j][k][0] < (angle + dist / 2) / 2# and hsv[j][k][1] > 100
                corners, contours, hex, found = check_for_hexagon(threshb)

        if found:
            corners_out.append(corners)
            contours_out.append(contours)
            out.append(hex)
        else:
            corners_out.append([])
            contours_out.append([])
            out.append([])

    return corners_out, contours_out, out


# expects list of 6 tuples with vertex coords
# warning: clears input list! use copy.copy to ensure safety
def get_hex_coords(vexlist):
    vexlist = vexlist[0]

    if len(vexlist) != 6:
        return -1

    # vexlist = [el[0] for el in vexlist]

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
    return [get_hex_coords(copy.copy(vertices)) if len(vertices) != 0 else [] for vertices in vexlist]


def compute_hex_colors(hexinfo, input, input_grey):
    return [get_hex_colors(hex, image, image_grey) if len(hex) != 0 else [] for hex, image, image_grey in zip(hexinfo, input, input_grey)]


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
    return [colors_classification_kmeans(colors_item) if len(colors_item) != 0 else [] for colors_item in colors]


def straighten_image(input, input_grey, vertices):
    h, _ = cv2.findHomography(np.array(vertices), UNTRANSFORMED_HEX_VERTICES)
    output = cv2.warpPerspective(input, h, (400, 400))
    output_grey = cv2.warpPerspective(input_grey, h, (400, 400))
    return output, output_grey, h


def straighten_images(input, input_grey, vertices):
    straight = []
    straight_grey = []
    perspective = []
    for input_item, input_grey_item, vertices_item in zip(input, input_grey, vertices):
        if len(vertices_item) == 0:
            straight.append([])
            straight_grey.append([])
            perspective.append([])
        else:
            ret = straighten_image(input_item, input_grey_item, vertices_item)
            straight.append(ret[0])
            straight_grey.append(ret[1])
            perspective.append(ret[2])
    return straight, straight_grey, perspective


def transform_points(points, perspective):
    return [cv2.perspectiveTransform(np.float32([points_item]), perspective_item) if len(points_item) != 0 else [] for points_item, perspective_item
            in zip(points, perspective)]


def untransform_points(points, perspective):
    return [cv2.perspectiveTransform(np.float32([points_item[0]]), np.linalg.inv(perspective_item)) if len(points_item) != 0 else [] for points_item, perspective_item
            in zip(points, perspective)]


def plot_images(input, filled, mod_hexinfo, hexinfo, vertices, contours, colors, classification):
    for i, (input_item, filled_item, mod_hexinfo_item, hexinfo_item, vertices_item, contour_item, color_item,
            classification_item) in enumerate(zip(input, filled, mod_hexinfo, hexinfo, vertices, contours, colors, classification)):
        image = input_item.copy()
        if len(hexinfo_item) != 0:
            heximage = cv2.cvtColor(filled_item, cv2.COLOR_GRAY2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.drawContours(image, contour_item, -1, (0, 255, 0), 2)
            for v, col, cl in zip(mod_hexinfo_item[0], color_item, classification_item):
                # print(v)
                vmod = tuple([int(v[0]), int(v[1])])
                cv2.circle(image, vmod, int(hexinfo_item[1]), (255, 255, 255))
                colmod = tuple([255*val for val in col])
                cv2.circle(heximage, vmod, 10, colmod, 15)
                cv2.putText(image, str(cl), (int(v[0]) - 7, int(v[1]) + 7), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # for n, cl in enumerate(classification_item):
            #     for p in cl:
            #         cv2.putText(image, str(n), (int(hexinfo_item[0][p[0]][0]) - 7, int(hexinfo_item[0][p[0]][1]) + 7), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # for i, v in enumerate(vertices_item):
            #     cv2.circle(image, tuple(v), 10, (0, 0, 100 + 30*i), 1)
        #     cv2.imshow('img2', heximage)
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('out/' + str(i) + '.jpg', image)


file_paths = list_file_paths(INPUT_DIR_PATH)
input, input_grey = get_input_data(file_paths)
vertices, contours, filled = perform_image_computations(input, input_grey)

straight, straight_grey, perspective = straighten_images(input, input_grey, vertices)
persp_vertices = transform_points(vertices, perspective)
hexinfo = compute_vertices(persp_vertices)
colors = compute_hex_colors(hexinfo, straight, straight_grey)
classification = compute_colors_classification(colors)

plot_images(input, filled, untransform_points(hexinfo, perspective), hexinfo, vertices, contours, colors, classification)



