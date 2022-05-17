

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_binary(in_image, threshold):
    return 1 * (in_image > threshold)


def get_regions(in_image):
    # m
    label = 2

    c = list()
    tmp = list()

    width, height = in_image.shape
    # current pixel
    for v in range(1, height - 1):
        for u in range(1, width - 1):
            if in_image[u][v] == 1:
                # current neighbors
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        tmp.append(in_image[u][v])

                # all neighbors are background ( = 0)
                if sum(tmp) == 0:
                    print(f'seperated region at {u} {v}')
                    in_image[u][v] = label
                    label += 1
                    continue

                # TODO not working
                # exactly one is already labeled ( > 1)
                # or at least two are labeled
                # or all neighbors are labeled
                tmp.sort(reverse=True)
                print(tmp)
                one_labeled = False
                two_labeled = False
                all_labeled = False
                count = 0
                for i in tmp:
                    if count == 0 and i > 1:
                        one_labeled = True
                    if count > 0 and i > 1:
                        if one_labeled:
                            two_labeled = True
                        one_labeled = False
                    if count == len(tmp) and i > 1:
                        all_labeled = True
                    count += 1
                count = 1
                if one_labeled:
                    print('one labeled')
                    in_image[u][v] = tmp[0]
                    continue
                if two_labeled or all_labeled:
                    print('two labeled')
                    in_image[u][v] = tmp[0]
                if all_labeled:
                    print('all labeled')
                    if tmp[count] > 1 and tmp[count] != tmp[0]:
                        c.append((tmp[count], tmp[0]))
                    count += 1
                tmp.clear()
    return in_image


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
    plt.imshow(img, cmap=cm.Greys_r)
    plt.show()

    plt.figure(1, dpi=300)
    plt.subplot(211)
    plt.imshow(img, cmap=cm.Greys_r)
    plt.figure(1, dpi=300)
    plt.subplot(212)
    plt.imshow(binary_img, cmap=cm.Greys_r)
    plt.show()

    exit(0)
