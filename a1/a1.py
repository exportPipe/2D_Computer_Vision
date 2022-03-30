from math import floor

import numpy
import skimage.io as skm
import matplotlib.pyplot as plt


def reduce_rgb(rgb):
    image = skm.imread(fname='images/bild01.jpg')
    for pixel_line in image:
        for pixel in pixel_line:
            if rgb == 'r':
                pixel[1] = 0
                pixel[2] = 0

            if rgb == 'g':
                pixel[0] = 0
                pixel[2] = 0

            if rgb == 'b':
                pixel[0] = 0
                pixel[1] = 0

    plt.imshow(image)
    skm.show()


def mirror_image(ver_or_hor):
    image = skm.imread(fname='images/bild01.jpg')
    height = len(image)
    width = len(image[0])

    if ver_or_hor == 'ver':
        column_r = 0
        for column_l in range(width - 1, floor(width/2), -1):
            print(f"{column_l} - {column_r}")
            for line in image:
                tmp = line[column_r]
                line[column_r] = line[column_l]
                line[column_l] = tmp
            column_r += 1

    if ver_or_hor == 'hor':
        low_line = height - 1
        for high_line in range(0, height):
            if low_line <= floor(height / 2):
                break
            print(f"{high_line} - {low_line}")
            for pixel in range(0, width - 1):
                tmp = image[low_line][pixel]
                image[low_line][pixel] = image[high_line][pixel]
                image[high_line][pixel] = tmp
            low_line -= 1

    plt.imshow(image)
    skm.show()


if __name__ == '__main__':
    reduce_rgb('r')
    # reduce_rgb('g')
    # reduce_rgb('b')
    mirror_image('hor')
