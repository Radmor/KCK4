from skimage import data, filters, morphology, feature
from matplotlib import pyplot as plt
import numpy as np
import os
import copy

input = []
input_grey = []

vlist = [[(510, 905), (698, 209), (1390, 0), (1950, 509), (998, 1444), (1742, 1271)],
         [(832, 160), (1520, 105), (516, 762), (1918, 697), (852, 1360), (1567, 1357)],
         [(1272, 59), (696, 386), (1824, 1181), (1878, 438), (651, 1047), (1180, 1451)]]


# expects list of 6 tuples with vertex coords
# warning: clears input list!
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

    # return list of 19 tuples of hex coords
    return hexes

# expects list of 6 tuples with vertex coords
def est_radius(vexlist):
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

    diag1 = ((right[0] - left[0])**2 + (right[1] - left[1]))**(1/2)
    diag2 = ((topright[0] - botleft[0])**2 + (topright[1] - botleft[1]))**(1/2)
    diag3 = ((botright[0] - topleft[0])**2 + (botright[1] - topleft[1]))**(1/2)

    diag = (diag1 + diag2 + diag3)/3

    return diag * 0.15


for filename in os.listdir("input"):
    input.append(data.imread("input/" + filename))
    input_grey.append(data.imread("input/" + filename, as_grey=True))

out = []
hexes = []
radii = []
for i in range(0, len(input)):
    r = np.zeros_like(input_grey[i])
    g = np.zeros((len(input[i]), len(input[i][0])))
    b = np.zeros((len(input[i]), len(input[i][0])))
    threshb = np.zeros((len(input[i]), len(input[i][0])))
    for j in range(0, len(input[i])):
        for k in range(0, len(input[i][j])):
            r[j][k] = input[i][j][k][0]
            g[j][k] = input[i][j][k][1]
            b[j][k] = input[i][j][k][2]
            if b[j][k] > r[j][k]*4/5:
                threshb[j][k] = 1
            else:
                threshb[j][k] = 0
    sobel = filters.sobel(threshb)
    canny = feature.canny(threshb, sigma=10)
    out.append([threshb, sobel, canny])
    hexes.append(get_hex_coords(copy.copy(vlist[i])))
    radii.append(est_radius(copy.copy(vlist[i])))


for i in range(0, len(input)):
    ax = plt.subplot(2, 2, 1)
    plt.imshow(input[i])
    for el in hexes[i]:
        circle = plt.Circle((el[0], el[1]), radius=radii[i]*0.5, alpha=0.4, color='white')
        ax.add_artist(circle)

    for nn, img in enumerate(out[i]):
        plt.subplot(2, 2, nn+2)
        plt.imshow(img)
        plt.gray()

    plt.show()
