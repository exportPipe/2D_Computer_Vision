from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_binary(in_image, threshold):
    return 1 * (in_image > threshold)


def get_regions(in_image, region_size):
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
                count = 0

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
    labels_R = list()
    for i in labels:
        labels_R.append({i})

    res = set()
    # Liste an Collisions (collisions)
    print(f'Kollisionen:    {collisions}')
    print(f'Label Sets      {labels_R}')
    r_a = 0
    r_b = 0
    for collision in collisions:
        for idx, entry in enumerate(labels_R):
            if collision[0] in entry:
                r_a = idx
            elif collision[1] in entry:
                r_b = idx
        if r_a != r_b:
            labels_R[r_a] = labels_R[r_a].union(labels_R[r_b])
            labels_R[r_b].clear()

    print(labels_R)

    # PASS 3 ----------------------
    for v in range(height):
        for u in range(width):
            if copy[u][v] > 1:
                for entry in labels_R:
                    if copy[u][v] in entry:
                        copy[u][v] = min(entry)

    return copy


if __name__ == "__main__":
    # read img
    img = io.imread("images/regionen2.png")
    # convert to numpy array
    img = np.array(img).astype(np.int16)

    # convert to binary - is working :)
    binary_img = get_binary(img, 150)

    # get regions
    regions = get_regions(binary_img, 4)
    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(binary_img, cmap=cm.Greys_r)
    plt.show()

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(regions)
    plt.show()

    exit(0)
