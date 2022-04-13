import numpy
import skimage.io as skm
import matplotlib.pyplot as plt

from a1 import a1


def compute_cum_histo(image, asImage):
    if not asImage:
        histo = a1.compute_histogram(image, False, False)
    else:
        histo = numpy.zeros(256, dtype=int)
        for row in range(0, len(image)):
            for point in range(0, len(image[0])):
                image[row][point] = 0.3 * image[row][point][0] + 0.59 * image[row][point][1] \
                                    + 0.11 * image[row][point][2]
                histo[image[row][point][0]] += 1

    cum_histo = numpy.zeros(256, dtype=int)

    for i in range(0, len(histo)):
        for j in range(0, i + 1):
            cum_histo[i] += histo[j]
    return cum_histo


def match_histo(img_histo, ref_histo):
    k = len(img_histo)

    def cdf(histo):
        n = 0
        for i in range(0, k):
            n += histo[i]
        p = numpy.zeros(k, dtype=float)
        c = histo[0]
        p[0] = c / n
        for i in range(1, k):
            c += histo[i]
            p[i] = c / n
        return p

    pa = cdf(img_histo)
    pr = cdf(ref_histo)

    f = numpy.zeros(k, dtype=int)

    for a in range(0, k):
        j = k - 1
        while j >= 0 and pa[a] <= pr[j]:
            f[a] = j
            j -= 1

    return f


def point_operation(lut: dict):
    image = skm.imread(fname='images/bild01.jpg')
    plt.imshow(image)
    plt.show()

    for row in image:
        for point in row:
            point[0] = lut[point[0]]
            point[1] = lut[point[1]]
            point[2] = lut[point[2]]

    plt.imshow(image)
    plt.show()

    cum_histo_after = compute_cum_histo(image, True)
    plt.plot(cum_histo_after)
    plt.show()


if __name__ == '__main__':
    reference_histo = compute_cum_histo('images/bild02.jpg', False)
    cum_histo01 = compute_cum_histo('images/bild01.jpg', False)
    print(f"{cum_histo01} {reference_histo}")

    match_histo01 = match_histo(cum_histo01, reference_histo)
    point_operation(match_histo01)

    plt.plot(reference_histo)
    plt.show()

    reference_image = skm.imread(fname='images/bild02.jpg')
    plt.imshow(reference_image)
    plt.show()

# Frage 1:  Homogene Punktoperationen sind unabhängig von Bildkoordinaten (für alle Pixel gleich),
#           unhomogene Punktoperationen sind abhängig von Bildkoordinaten
# Frage 2:  Punktoperationen berechnen anhand von einzelnen Pixeln,
#           Filteroperationen berechnen anhand einer Menge von Pixeln
