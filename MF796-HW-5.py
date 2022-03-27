import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import eig


def descretization(S, K1, K2, min, max, r, T, sigma, N1, N2):

    return

def matrix(S, K1, K2, min, max, r, T, sigma, N1, N2):
    ss = np.arange(min, max+((max-min)/N2),((max-min)/N2))
    aa = 1 - (sigma * ss) ** 2 * (T/N1) / (((max-min)/N2) ** 2) - r * (T/N1)
    ll = ((sigma * ss) ** 2) / 2 * ((T/N1) / (((max-min)/N2) ** 2)) - (r * ss * (T/N1)) / (2 * ((max-min)/N2))
    uu = ((sigma * ss) ** 2) / 2 * ((T/N1) / (((max-min)/N2) ** 2)) + (r * ss * (T/N1)) / (2 * ((max-min)/N2))
    AA = np.diag(aa[1:N2])
    upperLimit = uu[1: N2 -1]
    lowerLimit = ll[2: N2]

    for i in range(len(upperLimit)):
        AA[i][i+1] = upperLimit[i]
        AA[i+1][i] = lowerLimit[i]

    return AA



if __name__ == '__main__':
