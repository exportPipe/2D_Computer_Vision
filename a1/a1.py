import numpy
import skimage.io as skm
import matplotlib.pyplot as plt


def show_image():
    image = skm.imread(fname='images/bild01.jpg')
    plt.imshow(image)
    skm.show()
    for pixel_line in image:
        for pixel in pixel_line:
            print(pixel)


if __name__ == '__main__':
    show_image()
