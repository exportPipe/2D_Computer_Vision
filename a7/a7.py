from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_binary(in_image, threshold):
    return 1 * (in_image < threshold)


def get_regions(in_image):
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
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        if not (i == 0 and j == 0):
                            neighbors.append(copy[u + i][v + j])

                # all neighbors are background ( = 0)
                # TODO: geht hier nie rein, wieso auch, warum sollten alle Nachbarn Hintergrund sein??
                if sum(neighbors) == 0:
                    print(f'seperated region at {u}-{v}')
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
                        break
                    count += 1
                count = 0

                if one_labeled:
                    print('one labeled')
                    in_image[u][v] = neighbors[0]
                    neighbors.clear()
                    continue
                elif several_labeled:
                    print('two labeled')
                    in_image[u][v] = neighbors[0]
                    for neighbor in neighbors:
                        if neighbor != neighbors[0] and neighbor > 1:
                            collisions.append((neighbors[count], neighbors[0]))
                        print(collisions)
                neighbors.clear()

    # test purpose only
    label = 5
    collisions = [(2, 3), (2, 4), (3, 4)]

    # PASS 2
    # list of all labels > 1
    labels = list()
    for i in range(2, label):
        labels.append(i)

    # list of lists of all labels > 1
    labels_R = list()
    for i in labels:
        labels_R.append([i])

    # TODO: finde beide Listen in labels_R, die je einen der beiden Kollisionswerte enthalten und merge sie
    # ...

    return copy


if __name__ == "__main__":
    # read img
    img = io.imread("images/regionen1.png")
    # convert to numpy array
    img = np.array(img).astype(np.int16)

    # convert to binary - is working :)
    binary_img = get_binary(img, 150)

    # get regions
    regions = get_regions(binary_img)
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
    plt.imshow(regions, cmap=cm.Greys_r)
    plt.show()

    exit(0)
