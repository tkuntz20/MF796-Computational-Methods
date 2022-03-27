import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import eig


def descretization(S, K1, K2, min, max, r, T, sigma, N1, N2):

    return

def matrix(S, K1, K2, min, max, r, T, sigma, N1, N2, type, option='european'):
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

    return AA, ss, uu

def descreteEulerCall(AA, ss, uu, S, K1, K2, min, max, r, T, sigma, N1, N2, type, option='european'):
    c = (ss - K1)[1: N2]
    c[c<0]=0
    cVector = c

    for i in range(N1):
        cVector = AA.dot(cVector)
        cVector[-1] = cVector[-1] + uu[N2 -1] * (max - K1 * np.exp(-r * i * (T/N1)))
        if type == 'American':
            cVector = [x if x > y else y for x, y in zip(cVector, c)]
    return np.interp(S, ss[1:N1], cVector)

def descreteEulerSpread(AA, ss, uu, S, K1, K2, min, max, r, T, sigma, N1, N2, type, option='european'):
    short = (ss - K1)[1:N2]
    long = (ss - K2)[1:N2]
    short[short < 0] = 0
    long[long < 0] = 0
    cVector = long - short
    c = cVector

    for i in range(N1):
        cVector = AA.dot(cVector)
        cVector[-1] = cVector[-1] + uu[N2 -1] * (max - (K1 - K2) * np.exp(-r * i * (T/N1)))
        if type == 'American':
            cVector = [x if x > y else y for x, y in zip(cVector, c)]
    return np.interp(S, ss[1:N1], cVector)

def euroCall(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + (sigma **2 ) / 2) * T) / (sigma * (T ** 0.5))
    d2 = d1 - sigma * (T ** 0.5)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)



if __name__ == '__main__':

    s0 = 276.11
    k1 = 285
    k2 = 290
    t = 144 / 252
    r = 0.0247
    smin = 0
    smax = 500
    Nt = 1000
    Ns = 250
    sigma = 0.1331
    smax_list = [500, 1000, 1500]
    Nt_list = [1000, 5000, 10000]
    call_price_BS = euroCall(s0, k1, r, sigma, t)
    print(call_price_BS)

