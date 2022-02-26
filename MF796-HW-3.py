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
import re
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

    def __init__(self,S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def euroPayoff(self, density, S, K):
        payoff = 0
        for i in range(0, len(S)-2):
            payoff += density[i] * self.deltaPayoff(S[i], K)
        return payoff

    def strikeTransform(self,type, sigma, expiry, delta):
        transform = si.norm.ppf(delta)
        if type == 'P':
            K = 100 * np.exp(0.5 * sigma ** 2 * expiry + sigma * np.sqrt(expiry) * transform)
        else:
            K = 100 * np.exp(0.5 * sigma ** 2 * expiry - sigma * np.sqrt(expiry) * transform)
        return K

    def extractStrikes(self, type, sigma,expiry,delta):

        if type == 'P':
            return root(lambda x: norm.ppf(delta)-(np.log(self.S/x)+(self.r+0.5*sigma**2)*expiry) / (sigma * np.sqrt(expiry)), self.S).x
        else:
            return root(lambda x: norm.ppf(delta+1)-(np.log(self.S/x)+(self.r+0.5*sigma**2)*expiry) / (sigma * np.sqrt(expiry)), self.S).x

    def gammaTransform(self,S,K,T,r,sigma,h):
        value = base.euroCall(self,S, K, T, r, sigma)
        valuePlus = base.euroCall(self,S,K+h,T,r,sigma)
        valueDown = base.euroCall(self,S, K-h, T, r, sigma)
        return (valueDown - 2 * value + valuePlus) / h**2

    def riskNeutral(self,S,K,T,r,sigma,h):
        pdf = []
        for jj, vol in enumerate(sigma):
            p = np.exp(r*T) * self.gammaTransform(S, K[jj], T, r, vol,h)
            pdf.append(p)
        return pdf, K

    def constantVolatiltiy(self,S,T,r,sigma,h):
        K = np.linspace(70, 130, 150)
        pdf = []
        for k in K:
            p = np.exp(r*T) * self.gammaTransform(S, k, T, r, sigma,h)
            pdf.append(p)
        return pdf, K

    def deltaPayoff(self,S,K,type):
        if type == 'P':
            return 0 if S>K else 1
        else:
            return 0 if S<K else 1

    def deltaPrice(self,density,S,K,type):
        value = 0
        for i in range(0, len(S)-2):
            value += density[i] * self.deltaPayoff(S[i],K,type) * 0.1
        return value

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
    expiryStrike = ['10DP','25DP','40DP','50DP','40DC','25DC','10DC']
    vols = [[32.25,28.36],[24.73,21.78],[20.21,18.18],[18.24,16.45],[15.74,14.62],[13.70,12.56],[11.48,10.94]]
    volDictionary = dict(zip(expiryStrike, vols))
    volDF = pd.DataFrame.from_dict(volDictionary,orient='index',columns=['1M','3M'])
    #print(volDictionary)
    S = 100
    K = 0
    T = 0
    r = 0.0
    sigma = 0


    # part (a)
    BL = breedenLitzenberger(S, K, T, r, sigma)
    table = {}

    for row in volDictionary:
        delta = int(row[:2])/100
        type = row[-1]
        oneM = BL.strikeTransform(type,volDictionary[row][0]/100,1/12,delta)
        threeM = BL.strikeTransform(type,volDictionary[row][1]/100,3/12,delta)
        table[row] = [oneM,threeM]
    volTable = pd.DataFrame.from_dict(table,orient='index',columns=['1M','3M'])
    print(volTable)

    # part (b)
    strikeList = np.linspace(65, 112, 100)
    interp1M = np.polyfit(volTable['1M'], volDF['1M']/100,2)
    interp3M = np.polyfit(volTable['3M'], volDF['3M']/100,2)
    oneMvol = np.poly1d(interp1M)(strikeList)
    threeMvol = np.poly1d(interp3M)(strikeList)
    plt.plot(strikeList,oneMvol,color='r',label='1M vol')
    plt.plot(strikeList,threeMvol,color='b',label='3M vol')
    plt.xlabel('Strike Range')
    plt.ylabel('Volatilities')
    plt.title('Strike Against Volatility')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (c)
    pdf1 = BL.riskNeutral(S,strikeList,1/12,r,oneMvol,0.1)
    pdf2 = BL.riskNeutral(S,strikeList,3/12,r,threeMvol,0.1)
    plt.plot(pdf1[1], pdf1[0],label='1M volatility',linewidth=2)
    plt.plot(pdf2[1], pdf2[0], label='3M volatility',linewidth=2)
    plt.xlabel('Strike Range')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Densities')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (d)
    cpdf1 = BL.constantVolatiltiy(S, 1/12, r, 0.1824, 0.1)
    cpdf2 = BL.constantVolatiltiy(S, 3/12, r, 0.1645, 0.1)
    plt.plot(cpdf1[1], cpdf1[0], label='1M volatility',linewidth=2)
    plt.plot(cpdf2[1], cpdf2[0], label='3M volatility',linewidth=2)
    plt.xlabel('Strike Range')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Densities(const. vol)')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (e)
    S = np.linspace(65,112,len(pdf1))
    p1 = BL.deltaPrice(pdf1, S, 110,'P')
    p2 = BL.deltaPrice(pdf2, S,105,'C')
    v = (threeMvol+oneMvol)/2
    eupdf = BL.riskNeutral(100,strikeList,2/12,r,v,0.1)
    p3 = BL.euroPayoff(eupdf,S,100)
    print()
    print(f'1M European Digital Put Option with Strike 110:  {p1}')
    print(f'3M European Digital Call Option with Strike 105:  {p2}')
    print(f'2M European Call Option with Strike 100:  {p3}\n')

    # problem 2
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

