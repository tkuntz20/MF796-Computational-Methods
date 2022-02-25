"""
Created on Sun Feb 20 21:05:52 2022
@author: Thomas Kuntz MF-796-HW-3
"""

import math
import numpy as np
import pandas as pd
import cmath
import scipy.stats as si
from scipy.stats import kurtosis, skew, mode, norm
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import root, minimize
from scipy import interpolate


class base:

    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def __repr__(self):
        return f'Base Class initial price level is: {self.S}, strike is: {self.K}, expiry is: {self.T}, interest rate is: {self.r}, volatility is: {self.sigma}.'

    def d1(self,S, K, T, r, sigma):
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def d2(self,S, K, T, r, sigma):
        return (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def euroCall(self, S, K, T, r, sigma):
        call = (S * si.norm.cdf(self.d1(S, K, T, r, sigma), 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(self.d2(S, K, T, r, sigma), 0.0, 1.0))
        return float(call)

    def euroPut(self, S, K, T, r, sigma):
        put = (K * np.exp(-r * T) * si.norm.cdf(-self.d2(S, K, T, r, sigma), 0.0, 1.0) - S * si.norm.cdf(-self.d1(S, K, T, r, sigma), 0.0, 1.0))
        return float(put)

    def discountFactor(self,f,t):
        return 1/(1 + f)**t

class euroGreeks(base):

    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def delta(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        deltaCall = si.norm.cdf(self.d1(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        deltaPut = si.norm.cdf(-self.d1(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        return deltaCall, -deltaPut

    def gamma(self):
        return (1 / np.sqrt(2 * np.pi) * np.exp(-self.d1(self.S, self.K, self.T, self.r, self.sigma) ** 2 * 0.5)) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * (1 / np.sqrt(2 * np.pi) * np.exp(-self.d1(self.S, self.K, self.T, self.r, self.sigma) ** 2 * 0.5)) * np.sqrt(self.T)

    def theta(self):
        density = 1 / np.sqrt(2 * np.pi) * np.exp(-self.d1(self.S, self.K, self.T, self.r, self.sigma) ** 2 * 0.5)
        cTheta = (-self.sigma * self.S * density) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        pTheta = (-self.sigma * self.S * density) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        return cTheta, pTheta

class FastFourierTransforms():

    def __init__(self, S, K, T, r, q, sigma, nu, kappa, rho, theta):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q  # dividend (in this case q=0)
        self.sigma = sigma
        self.nu = nu
        self.kappa = kappa
        self.rho = rho
        self.theta = theta

    def __repr__(self):
        return f'FFT Class initial price level is: {self.S}, strike is: {self.K}, expiry is: {self.T}, interest rate is: ' \
               f'{self.r},\n dividend is: {self.q}, volatility is: {self.sigma}, nu is: {self.nu}, kappa is: {self.kappa}, rho is: {self.rho}, theta is: {self.theta}'

    def helper(self, n):

        delta = np.zeros(len(n), dtype=complex)
        delta[n == 0] = 1
        return delta

    def CharaceristicHeston(self, u):

        sigma = self.sigma
        nu = self.nu
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        S = self.S
        r = self.r
        T = self.T

        i = complex(0, 1)
        Lambda = cmath.sqrt(sigma ** 2 * (u ** 2 + i * u) + (kappa - i * rho * sigma * u) ** 2)
        omega = np.exp(i * u * np.log(S) + i * u * (r - q) * T + kappa * theta * T * (kappa - i * rho * sigma * u) / sigma ** 2) / ((cmath.cosh(Lambda * T / 2) + (kappa - i * rho * sigma * u) / Lambda * cmath.sinh(Lambda * T / 2)) ** (2 * kappa * theta / sigma ** 2))
        phi = omega * np.exp(-(u ** 2 + i * u) * nu / (Lambda / cmath.tanh(Lambda * T / 2) + kappa - i * rho * sigma * u))
        return phi

    def heston(self, alpha, N, B, K):

        t = time.time()
        tau = B / (2 ** N)
        Lambda = (2 * math.pi / (2 ** N)) / tau
        dx = (np.arange(1, (2 ** N) + 1, dtype=complex) - 1) * tau
        chi = np.log(self.S) - Lambda * (2 ** N) / 2
        dy = chi + (np.arange(1, (2 ** N) + 1, dtype=complex) - 1) * Lambda
        i = complex(0, 1)
        chiDx = np.zeros(len(np.arange(1, (2 ** N) + 1, dtype=complex)), dtype=complex)
        for ff in range(0, (2 ** N)):
            u = dx[ff] - (alpha + 1) * i
            chiDx[ff] = self.CharaceristicHeston(u) / ((alpha + dx[ff] * i) * (alpha + 1 + dx[ff] * i))
        FFT = (tau / 2) * chiDx * np.exp(-i * chi * dx) * (
                    2 - self.helper(np.arange(1, (2 ** N) + 1, dtype=complex) - 1))
        ff = np.fft.fft(FFT)
        mu = np.exp(-alpha * np.array(dy)) / np.pi
        ffTwo = mu * np.array(ff).real
        List = list(chi + (np.cumsum(np.ones(((2 ** N), 1))) - 1) * Lambda)
        Kt = np.exp(np.array(List))
        Kfft = []
        ffT = []
        for gg in range(len(Kt)):
            if (Kt[gg] > 1e-16) & (Kt[gg] < 1e16) & (Kt[gg] != float("inf")) & (Kt[gg] != float("-inf")) & (
                    ffTwo[gg] != float("inf")) & (ffTwo[gg] != float("-inf")) & (ffTwo[gg] is not float("nan")):
                Kfft += [Kt[gg]]
                ffT += [ffTwo[gg]]
        spline = interpolate.splrep(Kfft, np.real(ffT))
        value = np.exp(-self.r * self.T) * interpolate.splev(K, spline).real

        tt = time.time()
        compTime = tt - t

        return (value, compTime)

    def strikeCalibration(self, size, strikesLst, K):
        x = np.zeros((len(size), len(strikesLst)))
        y = np.zeros((len(size), len(strikesLst)))
        a, b = np.meshgrid(size, strikesLst)

        for gg in range(len(size)):
            for pp in range(len(strikesLst)):
                Heston = self.heston(1, size[gg], strikesLst[pp], K)
                x[gg][pp] = Heston[0]
                y[gg][pp] = 1 / ((Heston[0]) ** 2 * Heston[1])

        return x, y, a, b

class breedenLitzenberger(base):

    def __init__(self,table,S, K, T, r, sigma):
        self.table = table
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def extractStrikes(self,sigma,expiry,delta):

        if delta >= 0:
            return root(lambda x: norm.ppf(delta)-(np.log(self.S/x)+(self.r+0.5*sigma**2)*expiry) / (sigma * np.sqrt(expiry)), self.S).x
        else:
            return root(lambda x: norm.ppf(delta+1)-(np.log(self.S/x)+(self.r+0.5*sigma**2)*expiry) / (sigma * np.sqrt(expiry)), self.S).x

    def densities(self,sigma1,sigma2,K,T):

        density = []
        h = 0.01
        for i in range(len(K)):
            put = base.euroPut(self.S,K[i],T,self.r,sigma1[0]*K[i] + sigma1[1])
            positive = base.euroPut(self.S,K[i]+h,T,self.r,sigma1[0]*(K[i]+h) + sigma1[1])
            negative = base.euroPut(self.S,K[i]-h,T,self.r,sigma1[0]*(K[i]-h) + sigma1[1])
            density += [np.exp(self.r * T) * (negative - 2*put + positive) / h**2]
        den2 = []
        for i in range(len(K)):
            put1 = base.euroPut(self.S,K[i],T,self.r,sigma2)
            positive1 = base.euroPut(self.S,K[i]+h,T,self.r,sigma2)
            negative1 = base.euroPut(self.S,K[i]-h,T,self.r,sigma2)
            den2 += [np.exp(self.r * T) * (negative1 - 2*put1 + positive1) / h**2]
        return density, den2






if __name__ == '__main__':      # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # european option inputs
    S = 100
    K = 100.15
    T = 200/365
    r = 0.0
    sigma = 0.1712

    Base = base(S, K, T, r, sigma)
    df = Base.discountFactor(r, T)
    print(f'{repr(Base)}\n')
    print(f'european d1 is:  {Base.d1(S, K, T, r, sigma)}')
    print(f'european d2 is:  {Base.d2(S, K, T, r, sigma)}')
    print(f'   The call is:  {Base.euroCall(S, K, T, r, sigma)}')
    print(f'    The put is:  {Base.euroPut(S, K, T, r, sigma)}')
    print(f'discount factor: {Base.discountFactor(r, T)}\n')

    Greeks = euroGreeks(S, K, T, r, sigma)
    cDelta, pDelta = Greeks.delta()
    print(f'call delta:  {cDelta}')
    print(f'put delta:   {pDelta}')
    print(f'gamma:       {Greeks.gamma()}')
    print(f'vega:        {Greeks.vega()}')
    print(f'put theta:   {Greeks.theta()[1]}')
    print(f'call theta:  {Greeks.theta()[0]}')
    print('------------------tests above------------------\n')

    # Volatility table-------------------------------------------------
    volTable = pd.DataFrame({'delta':[10, 25, 40, 50, 40, 25, 10], 'strike':['DP','DP','DP','D','DC','DC','DC'], '1M':[32.25,24.73,20.21,18.24,15.74,13.70,11.48], '3M':[28.36,21.78,18.18,16.45,14.62,12.56,10.94]})
    print(f'{volTable.head(7)}\n')

    # part (a)




    S = 100
    K = 100.15
    T = 200/365
    r = 0.0
    q = 0.0
    sigma = 0.1712
    alpha = 1
    N = 10
    B = 10000
    nu = 0.08
    kappa = 0.7
    rho = -0.4
    theta = 0.1
    FFT = FastFourierTransforms(S, K, T, r, q, sigma, nu, kappa, rho, theta)
    print(repr(FFT))

