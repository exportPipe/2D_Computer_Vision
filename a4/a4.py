from math import floor
import numpy as np
import skimage.io as skm
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def derivativeHorizontal(imageIn, offset):
    sobelFilterHorizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    imageOut = np.empty((len(imageIn), len(imageIn[0])))

    for row in range(0, len(imageIn) - 1):
        for pixel in range(0, len(imageIn[0]) - 1):
            pixelDerivative = (float(imageIn[row][pixel - 1]) - float(imageIn[row][pixel + 1])) / 2
            imageOut[row][pixel] = pixelDerivative
            if imageIn[row][pixel] < 0:
                imageIn[row][pixel] = 0
            elif imageIn[row][pixel] > 255:
                imageIn[row][pixel] = 255

    return filter(imageOut, sobelFilterHorizontal, offset, 'min')


def derivativeVertical(imageIn, offset):
    sobelFilterVertical = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    imageOut = np.empty((len(imageIn), len(imageIn[0])))

    for row in range(0, len(imageIn) - 1):
        for pixel in range(0, len(imageIn[0]) - 1):
            pixelDerivative = (float(imageIn[row - 1][pixel]) - float(imageIn[row + 1][pixel])) / 2
            imageOut[row][pixel] = pixelDerivative
            if imageIn[row][pixel] < 0:
                imageIn[row][pixel] = 0
            elif imageIn[row][pixel] > 255:
                imageIn[row][pixel] = 255

    return filter(imageOut, sobelFilterVertical, offset, 'min')


def getEdgeThickness(imageHorizontal, imageVertical):
    gradientAbsolute = np.sqrt(pow(imageHorizontal, 2) + pow(imageVertical, 2))

    return gradientAbsolute


def padding(image, offset, edge):
    height, width = image.shape
    out_image = np.zeros((height + offset * 2, width + offset * 2))
    if edge == 'min':
        out_image[offset: -offset, offset: -offset] = image
    if edge == 'max':
        out_image = (out_image + 1) * 255
        out_image[offset: -offset, offset: -offset] = image

    if edge == 'continue':
        tmp = image[0]
        for top in range(0, offset):
            out_image[top, offset: offset + width] = tmp
        tmp = image[top]
        for bot in range(height + offset, height + 2 * offset):
             out_image[bot, offset: offset + height] = tmp
        out_image[offset: -offset, offset: -offset] = image
        tmp = out_image[:, offset]
        for left in range(0, offset):
            out_image[:, left] = tmp
        tmp = out_image[:, width - offset]
        for right in range(width + offset, width + 2 * offset):
            out_image[:, right] = tmp

    return out_image


def filter(image, filter_mask, off, edge):
    if off == 0:
        off = 1

    out_image = np.copy(image)

    m, n = image.shape

    out_image = np.resize(out_image, (round(m / off), round(n / off)))

    s = sum(sum(filter_mask))
    if s == 0:
        s = 1
    else:
        s = 1 / s

    k = floor(len(filter_mask) / 2)

    for v in range(k, n - k, off):
        for u in range(k, m - k, off):
            total = 0
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    p = image[u + i][v + j]
                    c = filter_mask[j + k][i + k]
                    total = total + c * p
            q = int(round(s * total))
            if q <= 0 or q >= 255:
                if edge == "min":
                    q = 0
                elif edge == "max":
                    q = 255
                elif edge == "continue":
                    q = image[u + 1][v]
            out_image[floor(u / off)][floor(v / off)] = q
    return out_image


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


if __name__ == '__main__':
    # imageRGB = skm.imread('images/dot01.png')
    # image = rgb2gray(imageRGB)

    offset = 2

    # imageRGB = skm.imread('images/dot02.png')
    # image = rgb2gray(imageRGB)

    image = skm.imread('images/fhorn.jpg')

    image = padding(image, offset, 'continue')
    imageHorizontal = derivativeHorizontal(image, offset)
    imageVertical = derivativeVertical(image, offset)
    edgeThickness = getEdgeThickness(imageHorizontal, imageVertical)

    print(edgeThickness)

    # plot original
    plt.figure(1, dpi=300)
    plt.subplot(311)
    plt.imshow(image, cmap=cm.Greys_r)
    # plot horizontal
    plt.figure(1, dpi=300)
    plt.subplot(312)
    plt.imshow(imageHorizontal, cmap=cm.Greys_r)
    # plot vertical
    plt.figure(1, dpi=300)
    plt.subplot(313)
    plt.imshow(imageVertical, cmap=cm.Greys_r)

    plt.show()
