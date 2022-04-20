from math import floor

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def filter1(image, filter_mask, off):
    if off == 0:
        off = 1

    out_image = np.copy(image)

    # IMAGE SIZE / should change width height?
    m, n = image.shape
    if n > m:
        m, n = n, m

    # OUTPUT IMAGE / reshape to offset size
    out_image = np.resize(out_image, (round(m / off), round(n / off)))

    # SCALE s / one divided by sum of all coefficients
    s = sum(sum(filter_mask))
    if s == 0:
        s = 1
    else:
        s = 1 / s

    # k is filter distance from current pixel
    k = floor(len(filter_mask) / 2)

    # moving mask by offset
    for v in range(k, n - k, off):
        for u in range(k, m - k, off):
            # collecting new pixel value q in mask area
            total = 0
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    p = image[u + i][v + j]
                    c = filter_mask[j + k][i + k]
                    total = total + c * p
            q = int(round(s * total))
            if q < 0:
                q = 0
            if q > 255:
                q = 255
            # setting q in img out
            out_image[floor(u / off)][floor(v / off)] = q

    return out_image


def filter2(image, filter_mask, off, edge):
    if off == 0:
        off = 1

    out_image = np.copy(image)

    m, n = image.shape
    if n > m:
        m, n = n, m

    out_image = np.resize(out_image, (round(m / off), round(n / off)))

    s = sum(sum(filter_mask))
    if s == 0:
        s = 1
    else:
        s = 1 / s

    k = floor(len(filter_mask) / 2)

    for v in range(k, n - k, off):
        for u in range(k, m - k, off):
            total = 0
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    p = image[u + i][v + j]
                    c = filter_mask[j + k][i + k]
                    total = total + c * p
            q = int(round(s * total))
            if q <= 0 or q >= 255:
                if edge == "min":
                    q = 0
                elif edge == "max":
                    q = 255
                elif edge == "continue":
                    q = image[u + 1][v]
            out_image[floor(u / off)][floor(v / off)] = q

    return out_image


def median_filter(in_image, filter_size, offset):
    if offset == 0:
        offset = 1
    copy = np.copy(in_image)
    m, n = copy.shape
    if n > m:
        m, n = n, m
    copy = np.resize(copy, (round(m / offset), round(n / offset)))

    p = np.ndarray(filter_size ** 2, dtype=int)

    for v in range(1, m - filter_size - 1, offset):
        for u in range(1, n - filter_size - 1, offset):
            k = 0
            for j in range(-floor(filter_size / 2), floor(filter_size / 2) + 1):
                for i in range(-floor(filter_size / 2), floor(filter_size / 2) + 1):
                    p[k] = in_image[u + i][v + j]
                    k += 1
            p = np.sort(p, kind='heapsort')

            copy[floor(u / offset)][floor(v / offset)] = p[floor(len(p) / 2)]
            print(copy[floor(u / offset)][floor(v / offset)])
    return copy


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


if __name__ == "__main__":
    # read img
    img = io.imread("images/lena.jpg")
    img2 = io.imread("images/pepper.jpg")
    img3 = io.imread("images/tree.png")

    # convert to numpy array
    img = np.array(img)
    img2 = np.array(img2)
    img3 = np.array(img3)

    # convert to grayscale
    img = rgb2gray(img)

    fm = np.array([
        [3, 5, 3],
        [5, 8, 5],
        [3, 5, 3]
    ])

    ö = 1
    fm_smooth = np.array([
        [ö, ö, ö],
        [ö, ö, ö],
        [ö, ö, ö]
    ])

    fmbig = np.array([
        [3, 3, 5, 3, 3],
        [3, 3, 5, 3, 3],
        [5, 5, 8, 5, 5],
        [3, 3, 5, 3, 3],
        [3, 3, 5, 3, 3]
    ])

    # FILTER
    # imgOut = filter1(img, fmbig, 2)
    # origImage = img

    # FILTER 2
    imgOut = filter2(img3, fm, 0, 'min')
    origImage = img3

    # MEDIAN
    # imgOut = median_filter(img2, 3, 1)
    # origImage = img2

    # plot img
    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(origImage, cmap=cm.Greys_r)
    # plot imgOut
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(imgOut, cmap=cm.Greys_r)

    plt.show()
    exit(0)
