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
        columnR = 0
        for columnL in range(width - 1, floor(width/2), -1):
            print(f"{columnL} - {columnR}")
            for line in image:
                tmp = line[columnR]
                line[columnR] = line[columnL]
                line[columnL] = tmp
            columnR += 1

    if ver_or_hor == 'hor':
        current_low_line = 0
        current_high_line = height - 1
        for line in image:
            if current_low_line >= floor(height/2):
                break
            for idx in range(0, width - 1):
                tmp = image[current_low_line][idx]
                image[current_low_line][idx] = image[current_high_line][idx]
                image[current_high_line] = tmp
            current_low_line += 1
            current_high_line -= 1
    plt.imshow(image)
    skm.show()


if __name__ == '__main__':
    # reduce_rgb('r')
    # reduce_rgb('g')
    # reduce_rgb('b')
    mirror_image('hor')
