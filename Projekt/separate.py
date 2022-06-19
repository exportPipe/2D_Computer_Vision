from math import floor


import skimage.transform
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_binary(in_image: np.ndarray, threshold) -> np.ndarray:
    return 1 * (in_image < threshold)


def dilate(in_image: np.ndarray, filter_h) -> np.ndarray:
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


def erode(in_image: np.ndarray, filter_h) -> np.ndarray:
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


def laplace(in_image: np.ndarray, filter_h) -> np.ndarray:
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


def median_filter(in_image: np.ndarray, filter_size, offset) -> np.ndarray:
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


def get_regions(in_image: np.ndarray, region_size) -> np.ndarray:
    # PASS 1
    label = 2
    collisions = list()
    neighbors = list()

    width, height = in_image.shape
    copy = in_image.copy()

    # current pixel
    for v in range(1, height - 1):
        for u in range(1, width - 1):
            if copy[u][v] == 1:
                # current neighbors
                if region_size == 4:
                    neighbors.append(copy[u][v - 1])
                    neighbors.append(copy[u - 1][v])
                elif region_size == 8:
                    neighbors.append(copy[u][v - 1])
                    neighbors.append(copy[u - 1][v])
                    neighbors.append(copy[u - 1][v - 1])
                    neighbors.append(copy[u + 1][v - 1])

                # all neighbors are background ( = 0)
                if sum(neighbors) == 0:
                    copy[u][v] = label
                    label += 1
                    neighbors.clear()
                    continue

                # sort neighbors descending
                neighbors.sort(reverse=True)

                one_labeled = False
                several_labeled = False
                count = 0
                for i in neighbors:
                    # mindestens ein Nachbar mit Label
                    if count == 0 and i > 1:
                        one_labeled = True
                    # mindestens zwei Nachbarn mit Label
                    if count > 0 and i > 1:
                        several_labeled = True
                        one_labeled = False
                    count += 1

                if one_labeled:
                    copy[u][v] = neighbors[0]
                    neighbors.clear()
                    continue
                elif several_labeled:
                    copy[u][v] = neighbors[0]
                    for neighbor in neighbors:
                        if neighbor != neighbors[0] and neighbor > 1:
                            collisions.append((neighbor, neighbors[0]))
                neighbors.clear()

    # PASS 2 -----------------------
    # list of all labels > 1
    labels = list()
    for i in range(2, label):
        labels.append(i)

    # list of lists of all labels > 1
    labels_r = list()
    for i in labels:
        labels_r.append({i})

    r_a = 0
    r_b = 0
    for collision in collisions:
        for idx, entry in enumerate(labels_r):
            if collision[0] in entry:
                r_a = idx
            elif collision[1] in entry:
                r_b = idx
        if r_a != r_b:
            labels_r[r_a] = labels_r[r_a].union(labels_r[r_b])
            labels_r[r_b].clear()

    # PASS 3 ----------------------
    for v in range(height):
        for u in range(width):
            if copy[u][v] > 1:
                for entry in labels_r:
                    if copy[u][v] in entry:
                        copy[u][v] = min(entry)

    return copy


def separate_region(in_image: np.ndarray, region: int) -> np.ndarray:
    width, height = in_image.shape
    copy = in_image.copy()
    for v in range(0, width):
        for u in range(0, height):
            if copy[u][v] != region:
                copy[u][v] = 0
            else:
                copy[u][v] = 1
    return copy


if __name__ == "__main__":
    # read img
    img = io.imread("images/bxfz.png")
    # img_resized = skimage.transform.resize(img, (400, 400))
    img = np.array(img).astype(np.int16)

    h2 = np.array([[0, 0, 0],
                   [0, 1, 1],
                   [0, 0, 0]])
    la = np.array([[0, 1, 0],
                   [1, -8, 1],
                   [0, 1, 0]])

    img_gray = rgb2gray(img)
    img_binary = get_binary(img_gray, 128)

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    # dilated_image = dilate(img_binary, h2)
    regions = get_regions(img_binary, 8)
    unique_region_indexes = np.unique(regions)
    unique_regions = []

    for region_idx in unique_region_indexes:
        if region_idx == 0:
            continue
        else:
            unique_regions.append(separate_region(regions, region_idx))

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img_binary, cmap=cm.Greys_r)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(regions)
    plt.show()

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(unique_regions[0], cmap=cm.Greys)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(unique_regions[1], cmap=cm.Greys)
    plt.show()

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(unique_regions[2], cmap=cm.Greys)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(unique_regions[3], cmap=cm.Greys)
    plt.show()

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    exit(0)
