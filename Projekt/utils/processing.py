from .settings import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_binary(in_image: np.ndarray, threshold) -> np.ndarray:
    return 1 * (in_image > threshold)


def get_text(grid):
    image = np.array(grid)
    image = rgb2gray(image)
    image = get_binary(image, 128)
    images = list()

    images.append(image[0:28, 0:28])
    images.append(image[0:28, 28:28*2])
    images.append(image[0:28, 0:28])

    for i in range(0, 4):
        for j in range(0, 4):
            images.append(image[i:i+28])

    for img in images:
        plt.figure(1, dpi=300)
        plt.imshow(img, cmap=cm.Greys_r)
        plt.show()

    return "Dieser Satz"

    # TODO; Niklas CNN

    # plt.figure(1, dpi=300)
    # plt.subplot(211)
    # plt.imshow(image, cmap=cm.Greys_r)
    # plt.figure(1, dpi=300)
    # plt.subplot(212)
    # plt.imshow(image, cmap=cm.Greys_r)
    # plt.show()

