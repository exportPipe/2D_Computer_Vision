from math import floor

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def filter1(image, filter_mask, off):
    s = 1 / (len(filter_mask) ** 2)

    k = floor(len(filter_mask) / 2)

    copy = np.copy(image)

    for v in range(k, len(image) - k - 1):
        for u in range(k, len(image[0]) - k - 1):
            total = 0
            for j in range(-k, k):
                for i in range(-k, k):
                    p = image[u+i][v+j]
                    c = filter_mask[j+k][i+k]
                    total = total + c * p
            print(total)
            q = floor(s * total)
            if q < 0:
                q = 0
            if q > 255:
                q = 255
            copy[u][v] = q

    return copy


def filter2(image, filter_mask, off, edge):
    return image


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


if __name__ == "__main__":
    # read img
    img = io.imread("images/lena.jpg")

    # convert to numpy array
    img = np.array(img)

    # convert to grayscale
    img = rgb2gray(img)

    fm = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    fm_smooth = np.array([
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
    ])

    imgOut = filter1(img, fm, 0)
    # imgOut = filter2(img, fm, 0, 'min')

    # plot img
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    # plot imgOut
    plt.figure(1)
    plt.subplot(212)
    plt.imshow(imgOut, cmap=cm.Greys_r)

    plt.show()
