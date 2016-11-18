from skimage import data, filters, morphology, feature
from matplotlib import pyplot as plt
import numpy as np
import os

input = []
input_grey = []

for filename in os.listdir("input"):
    input.append(data.imread("input/" + filename))
    input_grey.append(data.imread("input/" + filename, as_grey=True))

out = []
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


for i in range(0, len(input)):
    plt.subplot(2, 2, 1)
    plt.imshow(input[i])

    for nn, img in enumerate(out[i]):
        plt.subplot(2, 2, nn+2)
        plt.imshow(img)
        plt.gray()

    plt.show()
