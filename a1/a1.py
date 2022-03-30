from math import floor

import numpy
import skimage.io as skm
import matplotlib.pyplot as plt


def reduce_rgb(rgb):
    image = skm.imread(fname='images/bild01.jpg')
    for point_line in image:
        for point in point_line:
            if rgb == 'r':
                point[1] = 0
                point[2] = 0

            if rgb == 'g':
                point[0] = 0
                point[2] = 0

            if rgb == 'b':
                point[0] = 0
                point[1] = 0

    plt.imshow(image)
    skm.show()


def flip_image(ver_or_hor):
    image = skm.imread(fname='images/bild01.jpg')
    height = len(image)
    width = len(image[0])

    # flip vertically
    if ver_or_hor == 'ver':
        current_left_point = 0
        for current_right_point in range(width - 1, floor(width / 2), -1):
            # print(f"{current_left_point} - {current_right_point}")
            for line in image:
                tmp = line[current_left_point]
                line[current_left_point] = line[current_right_point]
                line[current_right_point] = tmp
            current_left_point += 1

    # flip horizontally
    if ver_or_hor == 'hor':
        curr_bottom_line = height - 1
        for curr_high_line in range(0, floor(height / 2)):
            # print(f"{curr_high_line} - {curr_bottom_line}")
            for point in range(0, width):
                tmp = image[curr_bottom_line][point]
                image[curr_bottom_line][point] = image[curr_high_line][point]
                image[curr_high_line][point] = tmp
            curr_bottom_line -= 1

    plt.imshow(image)
    skm.show()


def compute_histogram(img_file):
    image = skm.imread(fname=img_file)
    histogram = numpy.zeros(256)

    def compute(img):
        for line in range(0, len(image)):
            for point in range(0, len(image[0])):
                img[line][point] = 0.3 * img[line][point][0] + 0.59 * img[line][point][1] + 0.11 * img[line][point][2]
                histogram[img[line][point][0]] += 1

    compute(image)
    plt.plot(histogram)
    plt.show()
    return histogram


if __name__ == '__main__':
    reduce_rgb('r')
    reduce_rgb('g')
    reduce_rgb('b')
    flip_image('ver')
    flip_image('hor')
    compute_histogram('images/bild01.jpg')
    compute_histogram('images/bild02.jpg')
    compute_histogram('images/bild03.jpg')
    compute_histogram('images/bild04.jpg')
    compute_histogram('images/bild05.jpg')
