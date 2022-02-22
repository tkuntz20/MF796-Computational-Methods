# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:05:52 2022

@author: Thomas Kuntz
"""

import math
import numpy as np
import cmath
import scipy.stats as si
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.optimize import root
from scipy import interpolate

class europeanOption():
    
    def __init__(self,S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
    def __repr__(self):
        return f'initial level is {self.S}, strike is {self.K}, expiry is {self.T}, interest rate is {self.r}, volatility is {self.sigma}.'
    
    
    def euroCall(self,S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        return float(call)

class FastFourierTransforms():
    
    def __init__(self,S, K, T, r, q, sigma,nu,kappa,rho,theta):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q #dividend (in this case q=0)
        self.sigma = sigma
        self.nu = nu
        self.kappa = kappa
        self.rho = rho
        self.theta = theta
        
    def __repr__(self):
        return 
    
    def helper(self, n):
        
        delta = np.zeros(len(n),dtype = complex)
        delta[n==0] = 1
        return delta
    
    def CharaceristicHeston(self,u):
        
        sigma = self.sigma
        nu = self.nu
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        S = self.S
        r = self.r
        T = self.T
        
        i = complex(0,1)
        Lambda = cmath.sqrt(sigma**2*(u**2+i*u)+(kappa-i*rho*sigma*u)**2)
        omega = np.exp(i*u*np.log(S)+i*u*(r - q)*T+kappa*theta*T*(kappa-i*rho*sigma*u)/sigma**2) / ((cmath.cosh(Lambda*T/2) + (kappa-i*rho*sigma*u) / Lambda*cmath.sinh(Lambda*T/2))**(2*kappa*theta/sigma**2))
        phi = omega*np.exp(-(u**2+i*u)*nu/(Lambda/cmath.tanh(Lambda*T/2)+kappa-i*rho*sigma*u))
        return phi
    
    def heston(self,alpha,N,B,K):
        
        t = time.time()
        tau = B / (2**N)
        Lambda = (2 * math.pi / (2**N)) / tau
        dx = (np.arange(1,(2**N)+1,dtype = complex)-1) * tau
        chi = np.log(self.S) - Lambda * (2**N) / 2
        dy = chi + (np.arange(1,(2**N)+1,dtype = complex)-1) * Lambda
        i = complex(0,1)
        chiDx = np.zeros(len(np.arange(1,(2**N)+1,dtype = complex)),dtype = complex)
        for ff in range(0,(2**N)):
            u = dx[ff] - (alpha + 1) * i
            chiDx[ff] = self.CharaceristicHeston(u) / ((alpha + dx[ff] * i) * (alpha + 1 + dx[ff] * i))
        FFT = (tau/2) * chiDx * np.exp(-i * chi * dx) * (2 - self.helper(np.arange(1,(2**N)+1,dtype = complex)-1))
        ff = np.fft.fft(FFT)
        mu = np.exp(-alpha * np.array(dy)) / np.pi
        ffTwo = mu * np.array(ff).real
        List = list(chi + (np.cumsum(np.ones(((2**N), 1))) - 1) * Lambda)
        Kt = np.exp(np.array(List))
        Kfft = [] 
        ffT = []
        for gg in range(len(Kt)):
            if( Kt[gg]>1e-16 )&(Kt[gg] < 1e16)& ( Kt[gg] != float("inf"))&( Kt[gg] != float("-inf")) &( ffTwo[gg] != float("inf"))&(ffTwo[gg] != float("-inf")) & (ffTwo[gg] is not  float("nan")):
                Kfft += [Kt[gg]]
                ffT += [ffTwo[gg]]
        spline = interpolate.splrep(Kfft , np.real(ffT))
        value =  np.exp(-self.r*self.T)*interpolate.splev(K, spline).real

        tt = time.time()
        compTime = tt-t

        return(value,compTime)
    
    def alpha(self,lst,N,B,K):
        
        alphas = np.array([self.heston(alpha, N, B, K)[0] for alpha in lst])
        plt.plot(lst, alphas)
        return plt.show()
    
    def strikeCalibration(self, size, strikesLst,K):
        x = np.zeros((len(size),len(strikesLst)))
        y = np.zeros((len(size),len(strikesLst)))
        a, b = np.meshgrid(size, strikesLst)
        
        for gg in range(len(size)):
            for pp in range(len(strikesLst)):
                Heston = self.heston(1, size[gg], strikesLst[pp],K)
                x[gg][pp] = Heston[0]
                y[gg][pp] = 1 / ((Heston[0]) ** 2 * Heston[1])
        
        return x, y, a, b
    
class VolatilitySurfaces():
    
    def __init__(self, stockLst, strikeLst, expiryLst,S, K, T, r, sigma):
        self.stockLst = stockLst
        self.strikeLst = strikeLst
        self.expiryLst = expiryLst
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
    def strikeVolatility(self):
        volatility = []
        euC = europeanOption(self.S, self.K, self.T, self.r, self.sigma)
        for gg in range(len(self.strikeLst)):
            volPoint = root(lambda x: ((euC.euroCall(self.S, self.strikeLst[gg], self.T, self.r, x))) - self.stockLst[gg],0.3)
            volatility += [volPoint.x]
        return volatility, self.strikeLst

    def expiryVolatility(self):
        volatility = []
        euC = europeanOption(self.S, self.K, self.T, self.r, self.sigma)
        for gg in range(len(self.expiryLst)):
            volPoint = root(lambda x: ((euC.euroCall(S, K, self.expiryLst[gg], r, x))) - self.stockLst[gg],0.3)
            volatility += [volPoint.x]
        return volatility, self.expiryLst



if __name__ == '__main__':      # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    

    S = 250
    K = 250
    T = 0.5
    r = 0.02
    q = 0
    sigma = 0.2
    alpha = 1
    N = 10
    B = 250
    nu = 0.08
    kappa = 0.7
    rho = -0.4
    theta = 0.1
    
    # analytic sanity check
    analyticEuros = europeanOption(S, K, T, r, sigma)
    value = analyticEuros.euroCall(S, K, T, r, sigma)
    print(f'{repr(analyticEuros)}\n')
    print(f'the call value is:  {value}\n')
    
    # A i)
    Heston = FastFourierTransforms(S, K, T, r, q, sigma,nu,kappa,rho,theta)
    fft = Heston.heston(alpha, N, B, K)
    print(f'The FFT equivalent price is: {fft[0]}\n')
    print(f'Runtime is:  {fft[1]}\n')
    
    
    # alphas = np.linspace(0.1,10,num = 50)
    # alphaPlot = Heston.alpha(alphas,N,B,K)
    
    # alphass = [0.01, 0.02, 0.25, 0.5, 0.75, 0.9, 1 ,1.25, 1.5, 1.75, 2, 5, 7, 10, 15, 20, 30, 38]
    # plot = []
    # for i in alphass:
    #     plot += [Heston.heston(i,N,B,K)[0]]
    # print(plot)
    # plt.plot(plot)
    
    
    # # A ii)
    # sizeN = np.array([6,7,8,9,10,11,12,13,14,15])
    # strikesLst = np.linspace(150,350,100)
    # strikesPlot = Heston.strikeCalibration(sizeN, strikesLst,K)
    # fig1 = plt.figure()
    # ax = Axes3D(fig1)
    # ax.plot_surface(strikesPlot[2], strikesPlot[3], strikesPlot[0].T, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
    
    # print(f'k=260    {Heston.heston(alpha,N,B,260)}\n')
    
    # sizeN = np.array([6,7,8,9,10,11,12,13,14,15])
    # strikesLst = np.linspace(150,350,100)
    # strikesPlot = Heston.strikeCalibration(sizeN, strikesLst,260)
    # fig1 = plt.figure()
    # ax = Axes3D(fig1)
    # ax.plot_surface(strikesPlot[2], strikesPlot[3], strikesPlot[0].T, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
    
    # B i)
    S = 150
    K = 150
    T = 0.25
    r = 0.025
    q = 0
    sigma = 0.4
    alpha = 1
    N = 10
    B = 150
    nu = 0.09
    kappa = 0.5
    rho = 0.25
    theta = 0.12
    
    expiryLst = np.linspace(1/12,3,72)
    
    strikeLst = np.linspace(50,250,72)
    stockLst = []
    for i in strikeLst:
        stockLst += [Heston.heston(alpha, N, B, i)[0]]
    stockLst = np.array(stockLst)
    
    volSurface = VolatilitySurfaces(stockLst, strikeLst, expiryLst, S, K, T, r, sigma)
    
    strikeVol, lst = volSurface.strikeVolatility()
    
    strikeVol = np.array(strikeVol)
    plt.plot(lst, strikeVol)
    plt.show()
    
    # B ii)
    expiryLst = np.linspace(1/12,3,72)
    value = []
    for i in expiryLst:
        model = FastFourierTransforms(S, K, i, r, q, sigma,nu,kappa,rho,theta)
        value += [model.heston(alpha, N, B, K)[0]]
    value = np.array(value)
    
    expiryVol, lst1 = volSurface.expiryVolatility()
    expiryVol = np.array(expiryVol)
    plt.plot(lst1, expiryVol)
    plt.show()
        
    
    
    
    