from math import floor
import numpy
import skimage.io as skm
import matplotlib.pyplot as plt

from a1 import a1


def compute_cum_histo(image):
    histo = a1.compute_histogram(image, False, False)
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
            p[i] = round(c / n, 5)
        return p

    pa = cdf(img_histo)
    pr = cdf(ref_histo)
    print(pr)
    print(pa)

    f = numpy.zeros(k, dtype=int)

    for a in range(0, k):
        j = k - 1
        while j >= 0 and pa[a] <= pr[j]:
            f[a] = j
            j -= 1

    return f


if __name__ == '__main__':
    reference_histo = a1.compute_histogram('images/bild02.jpg', False, False)
    original_histo = a1.compute_histogram('images/bild01.jpg', False, False)

    match_histo01 = match_histo(original_histo, reference_histo)

    image = skm.imread('images/bild01.jpg')

    plt.plot(reference_histo)
    plt.show()
    plt.plot(original_histo)
    plt.show()

    exit(0)
