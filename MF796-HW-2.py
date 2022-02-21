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
    
    
    def euroCall(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        call = (self.S * si.norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2, 0.0, 1.0))
        return call, d1, d2

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
        
        delta = np.zeros(len(n),dtype=complex)
        delta[n==0] = 1
        return delta
    
    def CharaceristicHeston(self,u):
        
        imaginary = complex(0,1)
        Lambda = np.sqrt((self.sigma ** 2) * (u ** 2 + imaginary * u) + (self.kappa * imaginary * self.rho * self.sigma * u) ** 2)
        omega = np.exp(imaginary * u * np.log(self.S) + imaginary * u * (self.r - self.q) * self.T + ((self.kappa * self.theta * self.T * (self.kappa - imaginary* self.rho * self.sigma * u)) / (self.sigma ** 2))) / (cmath.cosh(Lambda * self.T / 2) + (self.kappa - imaginary* self.rho * self.sigma * u) / (Lambda) * cmath.sinh(Lambda * self.T / 2)) ** (2 * self.kappa * self.theta / self.sigma ** 2)
        #       np.exp(ii        * u * np.log(S0)     + ii        * u *(r-0)*T+kappa*theta*T*(kappa-ii*rho*sigma*u)/sigma**2)/(cmath.cosh(l*T/2)+(kappa-ii*rho*sigma*u)/l*cmath.sinh(l*T/2))**(2*kappa*theta/sigma**2)
        phi = omega * np.exp(-(u ** 2 + imaginary * u) * self.nu / (Lambda / cmath.tanh(Lambda * self.T / 2) + self.kappa - imaginary * self.rho * self.sigma * u))
        
        return phi
    
    def heston(self,alpha,N,M):
        
        t = time.time()
        imaginary = complex(0,1)
        
        n = N**2
        tau = M / n
        lambdaTau = 2 ** np.pi / n
        Lambda = lambdaTau / tau
        
        x = np.arange(1,n+1,dtype=complex)
        dx = (x-1) * tau
        y = np.arange(1,n+1,dtype=complex)
        chi = np.log(self.S) - Lambda * n /2
        dy = chi + (y-1) * Lambda
        CHIdx = np.zeros(len(x),dtype=complex)
        
        for i in range(0,n):
            u = dx[i] - (alpha + 1) ** imaginary  # I think its (alpha + 1), but it might be (alpha - 1)
            CHIdx [i] = self.CharaceristicHeston(u) / ((alpha + dx[i] * imaginary) * (alpha + 1 + dx[i] * imaginary))
        
        FFTinput = (tau / 2) * CHIdx * np.exp(-imaginary * chi * dx) * (2 - self.helper(x-1))
        FFT = np.fft.fft(FFTinput)
        
        mu = np.exp(-alpha * np.array(dy)) / np.pi
        FFTtwo = mu * np.array(FFT).real 
        List = list(chi + (np.cumsum(np.ones((n,1))) - 1) * Lambda)
        Kt = np.exp(np.array(List))
        
        Kfft = []
        ito = []
        for i in range(len(Kt)):
            if (Kt[i] != float('inf')) & (Kt[i] != float('-inf')) & ( Kt[i]>1e-16 ) & (Kt[i] < 1e16) & ( FFTtwo[i] != float("inf")) & (FFTtwo[i] != float("-inf")) & (FFTtwo[i] is not  float("nan")):
           #if( Kt[i]>1e-16 )&(Kt[i] < 1e16)& ( Kt[i] != float("inf"))&( Kt[i] != float("-inf")) &( zz2[i] != float("inf"))&(zz2[i] != float("-inf")) & (zz2[i] is not  float("nan")):
                Kfft += [Kt[i]]
                ito += [FFTtwo[i]]
        curve = interpolate.splrep(Kfft, np.real(ito))
        value = np.exp(-self.r * self.T) * interpolate.splev(self.K, curve).real
        tt = time.time()
        compTime = tt - t
        
        return value, compTime


if __name__ == '__main__':      # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    

    S = 250
    K = 250
    T = 0.5
    r = 0.02
    q = 0
    sigma = 0.2
    alpha = 1
    N = 9
    M = 250
    nu = 0.08
    kappa = 0.7
    rho = -0.4
    theta = 0.1
    
    
    # analytic sanity check
    analyticEuros = europeanOption(S, K, T, r, sigma)
    value, d1, d2 = analyticEuros.euroCall()
    print(f'the call value is:  {value}\n')
    print(f'{repr(analyticEuros)}\n')
    
    
    Heston = FastFourierTransforms(S, K, T, r, q, sigma,nu,kappa,rho,theta)
    fft = Heston.heston(alpha, N, M)
    print(f'The FFT price is: {fft[0]}')
    
    
    
    
    
    
    