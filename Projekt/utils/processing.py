import numpy as np
from matplotlib import pyplot as plt, cm
from skimage.transform import resize
import tensorflow as tf

model = tf.keras.models.load_model("cnna5")
model.trainable = False


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_binary(in_image: np.ndarray, threshold) -> np.ndarray:
    return 1 * (in_image < threshold)


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


def get_text(grid):
    image = np.array(grid)
    image = rgb2gray(image)
    image = get_binary(image, 128)

    regions = get_regions(image, 8)
    unique_region_indexes = np.unique(regions)
    unique_regions = []

    for region_idx in unique_region_indexes:
        if region_idx == 0:
            continue
        else:
            unique_regions.append(separate_region(regions, region_idx))

    rois = []
    for region in unique_regions:
        region_indexes = np.where(region == np.amax(region))
        row_min = np.amin(region_indexes[0])
        row_max = np.amax(region_indexes[0])
        col_min = np.amin(region_indexes[1])
        col_max = np.amax(region_indexes[1])
        roi = region[row_min:row_max, col_min:col_max]
        rois.append(roi)

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(image, cmap=cm.Greys_r)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(regions)
    plt.show()

    for idx, region in enumerate(rois):
        plt.figure(1, dpi=300)
        rois[idx] = np.pad(rois[idx], pad_width=2)
        rois[idx] = resize(rois[idx], (28, 28))
        plt.imshow(rois[idx], cmap=cm.Greys)
        plt.show()

    guess = ''
    for roi in rois:
        guess += str(np.argmax(model.predict(roi), axis=1))
    return guess

    # plt.figure(1, dpi=300)
    # plt.subplot(211)
    # plt.imshow(image, cmap=cm.Greys_r)
    # plt.figure(1, dpi=300)
    # plt.subplot(212)
    # plt.imshow(image, cmap=cm.Greys_r)
    # plt.show()
