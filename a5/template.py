import math
from math import floor

import skimage.util
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def filter2(image, filter_mask, scale, edge):
    if scale == 0:
        scale = 1

    out_image = np.copy(image)

    m, n = image.shape
    if n > m:
        m, n = n, m

    out_image = np.resize(out_image, (round(m / scale), round(n / scale)))

    s = sum(sum(filter_mask))
    if s == 0:
        s = 1
    else:
        s = 1 / s

    k = floor(len(filter_mask) / 2)

    for v in range(k, n - k, scale):
        for u in range(k, m - k, scale):
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
            out_image[floor(u / scale)][floor(v / scale)] = q
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
    return copy


def padding(image, offset, edge):
    height, width = image.shape
    out_image = np.zeros((height + offset * 2, width + offset * 2))
    if edge == 'min':
        out_image[offset: -offset, offset: -offset] = image
    if edge == 'max':
        out_image = (out_image + 1) * 255
        out_image[offset: -offset, offset: -offset] = image

    if edge == 'continue':
        tmp = image[0]
        for top in range(0, offset):
            out_image[top, offset: offset + width] = tmp
        tmp = image[top]
        for bot in range(height + offset, height + 2 * offset):
            out_image[bot, offset: offset + height] = tmp
        out_image[offset: -offset, offset: -offset] = image
        tmp = out_image[:, offset]
        for left in range(0, offset):
            out_image[:, left] = tmp
        tmp = out_image[:, width - offset]
        for right in range(width + offset, width + 2 * offset):
            out_image[:, right] = tmp

    return out_image


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def invert_gray(image):
    w, h = image.shape
    copy = image.copy()
    for v in range(0, w):
        for u in range(0, h):
            copy[u][v] = 255 - image[u][v]
    return copy


def linear_ht(im_edge, angle_steps: int, radius_steps: int):
    width, height = im_edge.shape
    x_ctr = round(width / 2)
    y_ctr = round(height / 2)
    n_ang = angle_steps
    d_ang = math.pi / angle_steps
    n_rad = radius_steps
    r_max = math.sqrt(x_ctr * x_ctr + y_ctr * y_ctr)
    d_rad = (2 * r_max) / n_rad
    hough_array = np.zeros((n_ang, n_rad))

    def fill_hough_accumulutor():
        for v in range(0, height):
            for u in range(0, width):
                if im_edge[u][v] > 0:
                    do_pixel(u, v)

    def do_pixel(u, v):
        x = u - x_ctr
        y = v - y_ctr
        for a in range(0, n_ang):
            theta = d_ang * a
            r = round(
                (x * math.cos(theta) + y * math.sin(theta)) / d_rad) + n_rad / 2
            r = round(r)
            if 0 <= r < n_rad:
                hough_array[a][r] += 1

    fill_hough_accumulutor()
    return hough_array


def threshold_operation(hough_array, threshold):
    width, height = hough_array.shape
    for v in range(0, height):
        for u in range(0, width):
            if hough_array[u][v] < threshold:
                hough_array[u][v] = 0
    return hough_array


if __name__ == "__main__":
    # read img
    img = io.imread("images/noisy-lines.tif")
    # convert to numpy array
    img = np.array(img)

    hough_arr = linear_ht(img, 255, 255)
    hough_arr_inv = invert_gray(hough_arr)
    # hough_arr = threshold_operation(hough_arr_inv, 126)

    # plot img
    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    # plot imgOut
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(hough_arr, cmap=cm.Greys_r)

    # plt.figure(1, dpi=300)
    # plt.subplot(313)
    # plt.imshow(hough_arr_inv, cmap=cm.Greys_r)
    plt.show()

    exit(0)
