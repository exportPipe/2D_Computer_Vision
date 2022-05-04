import math
from math import floor

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
    copy = hough_array.copy()
    for v in range(0, height):
        for u in range(0, width):
            if hough_array[u][v] > threshold:
                copy[u][v] = 0
    return copy


if __name__ == "__main__":
    # read img
    img = io.imread("images/noisy-lines.tif")
    # convert to numpy array
    img = np.array(img)

    hough_arr = linear_ht(img, 100, 100)

    hough_arr_inv = invert_gray(hough_arr)

    t = round(np.amax(hough_arr) / 2)
    hough_arr_threshold = threshold_operation(hough_arr, t)

    # plot img
    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    # plot hough array
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(hough_arr, cmap=cm.Greys_r)
    plt.show()

    # plot hough
    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(hough_arr, cmap=cm.Greys_r)
    # plot hough max
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(hough_arr_threshold, cmap=cm.Greys_r)
    plt.show()

    exit(0)
