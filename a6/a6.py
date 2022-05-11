from math import floor

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def dilate(in_image, filter_h, iter_num):
    width, height = in_image.shape
    copy = in_image.copy()

    k = floor(len(filter_h) / 2)
    tmp = []
    for iteration in range(iter_num):
        for v in range(k, height - k):
            for u in range(k, width - k):
                for j in range(-k, k + 1):
                    for i in range(-k, k + 1):
                        tmp.append(in_image[u + i][v + j] + filter_h[i + k][j + k])
                copy[u][v] = max(tmp)
                if copy[u][v] > 255:
                    copy[u][v] = 255
                if copy[u][v] < 0:
                    copy[u][v] = 0
                tmp.clear()
    return copy


def erode(in_image, filter_h, iter_num):
    in_image = in_image.T
    in_image = dilate(in_image, filter_h, iter_num)
    return in_image.T


if __name__ == "__main__":
    # read img
    img = io.imread("images/fhorn.jpg")
    # convert to numpy array
    img = np.array(img)

    H = np.array([[0, 0, 0],
                  [0, 1, 1],
                  [0, 0, 0]])
    eroded_img = erode(img, H, 2)
    # dilated_image = dilate(img, H, 2)
    # dilated_image_iter = dilate(img, H, 4)

    # plot img
    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    # plot hough array
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(eroded_img, cmap=cm.Greys_r)
    plt.show()

    # # plot hough
    # plt.figure(1, dpi=300)
    # plt.subplot(211)
    # plt.imshow(img, cmap=cm.Greys_r)
    # # plot hough max
    # plt.figure(1, dpi=300)
    # plt.subplot(212)
    # plt.imshow(dilated_image_iter, cmap=cm.Greys)
    # plt.show()

    exit(0)
