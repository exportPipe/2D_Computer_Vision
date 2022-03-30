import numpy
import skimage.io as skm
import matplotlib.pyplot as plt


def show_image():
    image = skm.imread(fname='images/bild01.jpg')
    print(image)
    plt.imshow(image)
    skm.show()
    print("END")


if __name__ == '__main__':
    show_image()
