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


if __name__ == '__main__':
    reduce_rgb('r')
    reduce_rgb('g')
    reduce_rgb('b')
