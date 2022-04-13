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
            p[i] = c / n
        return p

    pa = cdf(img_histo)
    pr = cdf(ref_histo)

    f = numpy.zeros(k, dtype=int)

    for a in range(0, k):
        j = k - 1
        while j >= 0 and pa[a] <= pr[a]:
            f[a] = j
            j -= 1

    return f


if __name__ == '__main__':
    reference_histo = compute_cum_histo('images/bild02.jpg')
    cum_histo01 = compute_cum_histo('images/bild01.jpg')
    print(f"{cum_histo01} {reference_histo}")

    match_histo01 = match_histo(cum_histo01, reference_histo)

    plt.plot(reference_histo)
    plt.show()

    exit(0)
