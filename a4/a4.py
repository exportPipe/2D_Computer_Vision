from math import floor
import numpy as np
import skimage.io as skm
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# wieso so komisch bei dot01 und dot02
# warum ist output abhängig von scaling

def derivativeHorizontal(imageIn):

    imageOut = np.empty((len(imageIn), len(imageIn[0])))

    for row in range(0, len(imageIn) - 1):
        for pixel in range(0, len(imageIn[0]) - 1):
            pixelDerivative = (float(imageIn[row][pixel - 1]) - float(imageIn[row][pixel + 1])) / 2
            imageOut[row][pixel] = pixelDerivative

            if imageIn[row][pixel] < 0:
                imageIn[row][pixel] = 0
            elif imageIn[row][pixel] > 255:
                imageIn[row][pixel] = 255

    # return filter(imageOut, sobelFilterHorizontal, scaling, 'min')
    return imageOut


def derivativeVertical(imageIn):


    imageOut = np.empty((len(imageIn), len(imageIn[0])))

    for row in range(0, len(imageIn) - 1):
        for pixel in range(0, len(imageIn[0]) - 1):
            pixelDerivative = (float(imageIn[row - 1][pixel]) - float(imageIn[row + 1][pixel])) / 2
            imageOut[row][pixel] = pixelDerivative
            if imageIn[row][pixel] < 0:
                imageIn[row][pixel] = 0
            elif imageIn[row][pixel] > 255:
                imageIn[row][pixel] = 255

    # return filter(imageOut, sobelFilterVertical, scaling, 'min')
    return imageOut


def getEdgeThickness(imageHorizontal, imageVertical):
    gradientAbsolute = np.sqrt(pow(imageHorizontal, 2) + pow(imageVertical, 2))

    return gradientAbsolute


# a3
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


# a3
def filter1(image, filter_mask, off):
    if off == 0:
        off = 1

    out_image = np.copy(image)

    # IMAGE SIZE / should change width height?
    m, n = image.shape
    if n > m:
        m, n = n, m

    # OUTPUT IMAGE / reshape to offset size
    out_image = np.resize(out_image, (round(m / off), round(n / off)))

    # SCALE s / one divided by sum of all coefficients
    s = sum(sum(filter_mask))
    if s == 0:
        s = 1
    else:
        s = 1 / s

    # k is filter distance from current pixel
    k = floor(len(filter_mask) / 2)

    # moving mask by offset
    for v in range(k, n - k, off):
        for u in range(k, m - k, off):
            # collecting new pixel value q in mask area
            total = 0
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    p = image[u + i][v + j]
                    c = filter_mask[j + k][i + k]
                    total = total + c * p
            q = int(round(s * total))
            if q < 0:
                q = 0
            if q > 255:
                q = 255
            # setting q in img out
            out_image[floor(u / off)][floor(v / off)] = q

    return out_image


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def execute_sobel(derivate, hor_or_ver, scale):
    if hor_or_ver == 'hor':
        sobelFilterHorizontal = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        return filter1(derivate, sobelFilterHorizontal, scale)

    if hor_or_ver == 'ver':
        sobelFilterVertical = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        return filter1(derivate, sobelFilterVertical, scale)



if __name__ == '__main__':
    imageRGB = skm.imread('images/dot01.png')
    imageOrg = rgb2gray(imageRGB)

    # imageRGB = skm.imread('images/dot02.png')
    # imageOrg = rgb2gray(imageRGB)

    # imageOrg = skm.imread('images/fhorn.jpg')

    offset = 20
    scaling = 1

    imageOrg = padding(imageOrg, offset, 'max')

    imageHorizontal = derivativeHorizontal(imageOrg)
    imageVertical = derivativeVertical(imageOrg)

    imageHorizontalSobel = execute_sobel(imageHorizontal, 'hor', scaling)
    imageVerticalSobel = execute_sobel(imageVertical, 'ver', scaling)

    edgeThickness = getEdgeThickness(imageHorizontalSobel, imageVerticalSobel)

    imageHorizontal = padding(imageHorizontal, 1, 'max')
    imageVertical = padding(imageVertical, 1, 'max')
    edgeThickness = padding(edgeThickness, 1, 'max')

    # print(edgeThickness)

    # plot original
    plt.figure(1, dpi=300)
    plt.subplot(411)
    plt.imshow(imageOrg, cmap=cm.Greys_r)
    # plot horizontal
    plt.figure(1, dpi=300)
    plt.subplot(412)
    plt.imshow(imageHorizontal, cmap=cm.Greys_r)
    # plot vertical
    plt.figure(1, dpi=300)
    plt.subplot(413)
    plt.imshow(imageVertical, cmap=cm.Greys_r)
    # plot edgeThickness
    plt.figure(1, dpi=300)
    plt.subplot(414)
    plt.imshow(edgeThickness, cmap=cm.Greys_r)

    plt.show()
    exit(0)
