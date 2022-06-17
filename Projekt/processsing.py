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

#threshold to set everything gray to black
def threshold_operation(hough_array, threshold):
    width, height = hough_array.shape
    copy = hough_array.copy()
    for v in range(height):
        for u in range(width):
            if hough_array[u][v] < threshold:
                copy[u][v] = 0
    return copy


#def median_filter(in_image, filter_size, offset):
#    if offset == 0:
#        offset = 1
#    copy = np.copy(in_image)
#    m, n = copy.shape
#    if n > m:
#        m, n = n, m
#    copy = np.resize(copy, (round(m / offset), round(n / offset)))
#
#    p = np.ndarray(filter_size ** 2, dtype=int)
#
#    for v in range(1, m - filter_size - 1, offset):
#        for u in range(1, n - filter_size - 1, offset):
#            k = 0
#            for j in range(-floor(filter_size / 2), floor(filter_size / 2) + 1):
#                for i in range(-floor(filter_size / 2), floor(filter_size / 2) + 1):
#                    p[k] = in_image[u + i][v + j]
#                    k += 1
#            p = np.sort(p, kind='heapsort')
#
#            copy[floor(u / offset)][floor(v / offset)] = p[floor(len(p) / 2)]
#    return copy

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

#    plt.figure(1, dpi=300)
#    plt.subplot(211)
#    plt.imshow(img, cmap=cm.Greys_r)
#    plt.figure(1, dpi=300)
#    plt.subplot(212)
#    plt.imshow(dilated_image, cmap=cm.Greys_r)
#    plt.show()

    laplace_image = laplace(dilated_image, la)

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(laplace_image, cmap=cm.Greys_r)
    plt.show()

    t = round(np.amax(laplace_image) / 2)
    thresholde_image = threshold_operation(laplace_image, t)

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(thresholde_image, cmap=cm.Greys_r)
    plt.show()

#    imgOut = median_filter(laplace_image, 3, 1)
#
#    plt.figure(1, dpi=300)
#    plt.subplot(211)
#    plt.imshow(img, cmap=cm.Greys_r)
#    plt.figure(1, dpi=300)
#    plt.subplot(212)
#    plt.imshow(imgOut, cmap=cm.Greys_r)
#    plt.show()

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    exit(0)