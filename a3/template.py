from math import floor

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def filter1(image, filter_mask, off):
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
    m_o, n_o = out_image.shape
    if n > m:
        m, n = n, m
    v_o, u_o = k, k

    for v in range(k, n - k, off):
        v_o += 1
        if v_o >= n_o:
            break
        for u in range(k, m - k, off):
            u_o += 1
            if u_o >= m_o:
                u_o = k
            total = off
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
            out_image[u_o][v_o] = q

    return out_image


def filter2(image, filter_mask, off, edge):
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
    m_o, n_o = out_image.shape
    if n > m:
        m, n = n, m
    v_o, u_o = k, k

    for v in range(k, n - k, off):
        v_o += 1
        if v_o >= n_o:
            break
        for u in range(k, m - k, off):
            u_o += 1
            if u_o >= m_o:
                u_o = k
            total = off
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    p = image[u + i][v + j]
                    c = filter_mask[j + k][i + k]
                    total = total + c * p
            q = int(round(s * total))
            if q < 0 or q > 255:
                if edge == "min":
                    q = 0
                elif edge == "max":
                    q = 255
                elif edge == "continue":
                    q = image[u + 1][v]
            out_image[u_o][v_o] = q

    return out_image


def median_filter(in_image, filter_size, offset):
    copy = np.copy(in_image)
    p = np.ndarray(filter_size ** 2, dtype=int)

    for v in range(1, len(in_image) - filter_size):
        for u in range(1, len(in_image[0]) - filter_size):
            k = 0
            for j in range(-floor(filter_size / 2), floor(filter_size / 2) + 1):
                for i in range(-floor(filter_size / 2), floor(filter_size / 2) + 1):
                    p[k] = in_image[u + i][v + j]
                    k += 1
            p = np.sort(p, kind='heapsort')
            copy[u][v] = p[floor(len(p) / 2)]
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

    ö = (1 / 9)
    fm_smooth = np.array([
        [ö, ö, ö],
        [ö, ö, ö],
        [ö, ö, ö]
    ])

    # FILTER
    # imgOut = filter1(img, fm, 3)
    # origImage = img

    # FILTER 2
    imgOut = filter2(img2, fm, 2, 'continue')
    origImage = img2

    # MEDIAN
    # imgOut = median_filter(img2, 3, 1)
    # origImage = img2

    # plot img
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(origImage, cmap=cm.Greys_r)
    # plot imgOut
    plt.figure(1)
    plt.subplot(212)
    plt.imshow(imgOut, cmap=cm.Greys_r)

    plt.show()
    exit(0)
