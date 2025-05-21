import numpy as np


NB_BINS = 40


def kl_div(p, q, nb_bins=NB_BINS):
    freq_p, _ = np.histogram(p, bins=nb_bins, range=(0.0, 1.0), density=True)
    freq_q, _ = np.histogram(q, bins=nb_bins, range=(0.0, 1.0), density=True)
    elem = freq_p * np.log(freq_p / freq_q)
    return np.sum(np.where((~np.isnan(elem)) & (freq_q != 0), elem, 0))
