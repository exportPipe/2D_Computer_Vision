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


def flip_image(ver_or_hor: str):
    image = skm.imread(fname='images/bild01.jpg')
    height = len(image)
    width = len(image[0])

    # flip vertically
    # TODO: only half of the image is mirrored
    if ver_or_hor == 'ver':
        current_left_point = 0
        for current_right_point in range(width - 1, 0, -1):
            # print(f"{current_left_point} - {current_right_point}")
            for row in image:
                row[current_left_point], row[current_right_point] = row[current_right_point], row[current_left_point]
            current_left_point += 1

    # flip horizontally
    # TODO: only half of the image is mirrored
    if ver_or_hor == 'hor':
        curr_bottom_row = height - 1
        # for curr_high_row in range(0, floor(height / 2)):
        for curr_high_row in range(0, height):
            # print(f"{curr_high_row} - {curr_bottom_row}")
            for point in range(0, width):
                image[curr_bottom_row][point], image[curr_high_row][point] \
                    = image[curr_high_row][point], image[curr_bottom_row][point]
            curr_bottom_row -= 1

    plt.imshow(image)
    skm.show()


def compute_histogram(img_file: str, show_gray: bool, show_histogram: bool):
    image = skm.imread(fname=img_file)
    histogram = numpy.zeros(256, dtype=int)

    def compute():
        for row in range(0, len(image)):
            for point in range(0, len(image[0])):
                image[row][point] = 0.3 * image[row][point][0] + 0.59 * image[row][point][1] \
                                     + 0.11 * image[row][point][2]
                histogram[image[row][point][0]] += 1

    compute()
    if show_histogram:
        plt.plot(histogram)
        plt.title(img_file)
        plt.show()
    if show_gray:
        plt.imshow(image)
        skm.show()
    return histogram


if __name__ == '__main__':
    # reduce_rgb('r')
    # reduce_rgb('g')
    # reduce_rgb('b')
    flip_image('ver')
    flip_image('hor')
    # compute_histogram('images/bild01.jpg', False, True)
    # compute_histogram('images/bild02.jpg', False, True)
    # compute_histogram('images/bild03.jpg', False, True)
    # compute_histogram('images/bild04.jpg', False, True)
    # compute_histogram('images/bild05.jpg', False, True)
    exit(0)
