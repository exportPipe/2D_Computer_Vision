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


def flip_image(ver_or_hor):
    image = skm.imread(fname='images/bild01.jpg')
    height = len(image)
    width = len(image[0])

    # flip vertically
    if ver_or_hor == 'ver':
        current_left_column = 0
        for current_right_column in range(width - 1, floor(width/2), -1):
            print(f"{current_left_column} - {current_right_column}")
            for line in image:
                tmp = line[current_left_column]
                line[current_left_column] = line[current_right_column]
                line[current_right_column] = tmp
            current_left_column += 1

    # flip horizontally
    if ver_or_hor == 'hor':
        curr_bottom_line = height - 1
        for curr_high_line in range(0, floor(height/2)):
            print(f"{curr_high_line} - {curr_bottom_line}")
            for pixel in range(0, width - 1):
                tmp = image[curr_bottom_line][pixel]
                image[curr_bottom_line][pixel] = image[curr_high_line][pixel]
                image[curr_high_line][pixel] = tmp
            curr_bottom_line -= 1

    plt.imshow(image)
    skm.show()


if __name__ == '__main__':
    reduce_rgb('r')
    reduce_rgb('g')
    reduce_rgb('b')
    flip_image('ver')
    flip_image('hor')
