from math import floor

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def dilate(in_image, filter_h):
    width, height = in_image.shape
    copy = in_image.copy()

    k = floor(len(filter_h) / 2)
    tmp = []
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


def erode(in_image, filter_h):
    width, height = in_image.shape
    copy = in_image.copy()

    k = floor(len(filter_h) / 2)
    tmp = []
    for v in range(k, height - k):
        for u in range(k, width - k):
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    tmp.append(in_image[u + i][v + j] - filter_h[i + k][j + k])
            copy[u][v] = min(tmp)
            if copy[u][v] > 255:
                copy[u][v] = 255
            if copy[u][v] < 0:
                copy[u][v] = 0
            tmp.clear()
    return copy


if __name__ == "__main__":
    # read img
    img = io.imread("images/fhorn.jpg")
    # convert to numpy array
    img = np.array(img).astype(np.int16)

    h1 = np.array([[0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]])
    h2 = np.array([[0, 0, 0],
                   [0, 1, 1],
                   [0, 0, 0]])

    dilated_image = dilate(img, h1)
    # tmp = img.copy()
    # for i in range(5):
    #     tmp = erode(tmp, h1)



    # dilated_image = dilate(img, h2)
    # eroded_img = erode(img, h2)

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(dilated_image, cmap=cm.Greys_r)
    plt.show()

    # plt.figure(1, dpi=300)
    # plt.subplot(211)
    # plt.imshow(img, cmap=cm.Greys_r)
    # plt.figure(1, dpi=300)
    # plt.subplot(212)
    # plt.imshow(tmp, cmap=cm.Greys_r)
    # plt.show()

    exit(0)
