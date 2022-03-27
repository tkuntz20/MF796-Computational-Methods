import numpy as np
from scipy.optimize import root
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import eig


def matrix(S, K1, K2, min, max, r, T, sigma, N1, N2, type, option):
    hs = (max-min)/N2
    ht = T / N1
    ss = np.arange(min, max + hs, hs)
    aa = 1 - (sigma * ss) ** 2 * ht / (hs ** 2) - r * ht
    ll = ((sigma * ss) ** 2) / 2 * (ht / (hs ** 2)) - (r * ss * ht) / (2 * hs)
    uu = ((sigma * ss) ** 2) / 2 * (ht / (hs ** 2)) + (r * ss * ht) / (2 * hs)
    AA = np.diag(aa[1:N2])
    upperLimit = uu[1:N2 - 1]
    lowerLimit = ll[2:N2]

    for i in range(len(upperLimit)):
        AA[i][i+1] = upperLimit[i]
        AA[i+1][i] = lowerLimit[i]

    if type == 'call':
        c2 = (ss - K1)[1: N2]
        c2[c2 < 0] = 0
        cVector = c2

        for i in range(N1):
            cVector = AA.dot(cVector)
            cVector[-1] = cVector[-1] + uu[N2 - 1] * (max - K1 * np.exp(-r * i * ht))
            if option == 'American':
                cVector = [x if x > y else y for x, y in zip(cVector, c2)]

    elif type == 'callspread':
        short = (ss - K2)[1:N2]
        long = (ss - K1)[1:N2]
        short[short < 0] = 0
        long[long < 0] = 0
        cVector = long - short
        c = cVector

        for i in range(N1):
            cVector = AA.dot(cVector)
            cVector[-1] = cVector[-1] + uu[N2 -1] * (max - (K1 - K2) * np.exp(-r * i * ht))
            if option == 'American':
                cVector = [x if x > y else y for x, y in zip(cVector, c)]

    return np.interp(S, ss[1:N2], cVector), AA

def euroCall(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * (T ** 0.5))
    d2 = d1 - sigma * (T ** 0.5)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)



if __name__ == '__main__':

    S = 440
    K1 = 445
    K2 = 450
    T = 177/365
    r = 0.05
    min = 0
    max = 500
    Nt = 1000
    Ns = 250

    # finding volatility( call values pulled from Bloomberg)
    callK1 = 29.84
    callK2 = 26.79
    vol1 = root(lambda x: euroCall(S, K1, T, r, x) - callK1, 0.1).x
    vol2 = root(lambda x: euroCall(S, K2, T, r, x) - callK2, 0.1).x

    sigma = (vol1 + vol2) / 2
    print(f'sigma= : {sigma}')


    smax_list = [500, 1000, 1500]
    Nt_list = [1000, 5000, 10000]
    call_price_BS = euroCall(S, K1, T, r, sigma)
    print(call_price_BS)

    call_price_pde = []
    error = []

    for i in range(len(smax_list)):
        call_price_pde += [matrix(S, K1, K2, min, smax_list[i],r , T, sigma, Nt_list[i], Ns, 'call', 'European')[0]]
    print(f'the descrete:  {call_price_pde}')

    error = abs((call_price_pde - call_price_BS) / call_price_BS)
    print(f'error is:  {error}')

    AA = matrix(S, K1, K2, min, max, r, T, sigma, Nt, Ns, 'call', 'European')[1]
    eigenValue = eig(AA)[0]
    absEig = sorted(abs(eigenValue), reverse=True)
    firstEig = absEig[0]
    #print(f'eigen values   {eigenValue}')
    #print(f'absolute values  {absEig}')
    #print(f'first eigenvalue   {firstEig}')

    spreadEuro = matrix(S, K1, K2, min, max, r, T, sigma, Nt, Ns, 'callspread', 'European')[0]
    print(f'spread call value(euro):  {spreadEuro}')

    spreadAmerican = matrix(S, K1, K2, min, max, r, T, sigma, Nt, Ns, 'callspread', 'American')[0]
    print(f'spread call value(american):  {spreadAmerican}')

    print(f'premium:  {spreadAmerican - spreadEuro}')