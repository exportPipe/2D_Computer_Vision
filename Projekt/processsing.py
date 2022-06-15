from math import floor

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

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

def laplace(in_image, filter_h):
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

if __name__ == "__main__":

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    # read img
    #img = io.imread("images/buchstaben.jpg")
    #img = io.imread("images/buchstaben1.jpg")
    img = io.imread("images/Buchstabenklein.jpg")
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
    la = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]])

    imggray = rgb2gray(img)

    #plt.figure(1, dpi=300)
    #plt.imshow(imggray, cmap=cm.Greys_r)
    #plt.show()

    dilated_image = dilate(imggray, h1)

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(dilated_image, cmap=cm.Greys_r)
    plt.show()

    laplace_image = laplace(dilated_image, la)

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(laplace_image, cmap=cm.Greys_r)
    plt.show()

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    exit(0)