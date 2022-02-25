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
from matplotlib import cm
import time
from mpl_toolkits.mplot3d import Axes3D
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

        return value, compTime
    
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
        euCC = europeanOption(self.S, self.K, self.T, self.r, self.sigma)
        for gg in range(len(self.expiryLst)):
            volPoint = root(lambda x: ((euCC.euroCall(S, K, self.expiryLst[gg], r, x))) - self.stockLst[gg],0.3)
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
    
    alphass = [0.01, 0.02, 0.25, 0.5, 0.75, 0.9, 1 ,1.25, 1.5, 1.75, 2, 5, 7, 10, 15, 20, 30, 38]
    plot = []
    for i in alphass:
        plot += [Heston.heston(i,N,B,K)[0]]
    plt.plot(plot,marker='*')
    plt.title("Alpha Values")
    plt.xlabel("Alphas")
    plt.ylabel("FFT Heston Call Price")
    plt.grid(linestyle = '--', linewidth = 1)
    default_x_ticks = range(len(alphass))
    plt.xticks(default_x_ticks, alphass)
    plt.show()
    
    
    # A ii)
    sizeN = np.array([6,7,8,9,10,11,12,13,14])
    strikesLst = np.linspace(150,350,100)
    strikesPlot = Heston.strikeCalibration(sizeN, strikesLst,K)
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    ax.plot_surface(strikesPlot[2], strikesPlot[3], strikesPlot[0].T, rstride=1, cstride=1, cmap=cm.coolwarm)
    plt.title("Call Option Price for N & B ranges, K=250")
    ax.set_xlabel("N for 2^N")
    ax.set_ylabel("strikes = B")
    ax.set_zlabel("Call Option Prices")
    ax.view_init(10, 80)
    plt.show()
    
    fig2 = plt.figure()
    ax1 = Axes3D(fig2)
    ax1.plot_surface(strikesPlot[2], strikesPlot[3], strikesPlot[1].T, rstride=1, cstride=1, cmap=cm.coolwarm)
    plt.title("Efficiency of N & B ranges, K=250")
    ax1.set_xlabel("N for 2^N")
    ax1.set_ylabel("strikes = B")
    ax1.set_zlabel("efficiency")
    ax1.view_init(10, 80)
    plt.show()
    
    
    print(f'k=260    {Heston.heston(alpha,N,B,260)}\n')
    
    sizeN = np.array([6,7,8,9,10,11,12,13,14])
    strikesLst = np.linspace(150,350,100)
    Heston260 = FastFourierTransforms(S, K, T, r, q, sigma,nu,kappa,rho,theta)
    strikesPlot260 = Heston260.strikeCalibration(sizeN, strikesLst,260)
    
    fig3 = plt.figure()
    ax2 = Axes3D(fig3)
    ax2.plot_surface(strikesPlot260[2], strikesPlot260[3], strikesPlot260[0].T, rstride=1, cstride=1, cmap=cm.coolwarm)
    plt.title("Call Option Price for N & B ranges, K=260")
    ax2.set_xlabel("N for 2^N")
    ax2.set_ylabel("strikes = B")
    ax2.set_zlabel("Call Option Prices")
    ax2.view_init(10, 83)
    plt.grid(linestyle = '--', linewidth = 1)
    plt.show()
    
    fig4 = plt.figure()
    ax3 = Axes3D(fig4)
    ax3.plot_surface(strikesPlot260[2], strikesPlot260[3], strikesPlot260[1].T, rstride=1, cstride=1, cmap=cm.coolwarm)
    plt.title("Efficiency of N & B ranges, K=260")
    ax3.set_xlabel("N for 2^N")
    ax3.set_ylabel("strikes = B")
    ax3.set_zlabel("efficiency")
    ax3.view_init(10, 83)
    plt.show()
    
    # B i)
    S = 150
    K = 150
    T = 0.25
    r = 0.025
    q = 0
    sigma = 0.4
    alpha = 1
    N = 9
    B = 1000
    nu = 0.09
    kappa = 0.5
    rho = 0.25
    theta = 0.12
    
    bHeston = FastFourierTransforms(S, K, T, r, q, sigma, nu, kappa, rho, theta)
    expiryLst = 0
    strikeLst = np.linspace(50,250,71)
    stockLst = []
    for i in strikeLst:
        stockLst += [bHeston.heston(alpha, N, B, i)[0]]
    stockLst = np.array(stockLst)
    
    volSurfaceK = VolatilitySurfaces(stockLst, strikeLst, expiryLst, S, K, T, r, sigma)
    strikeVol, lst = volSurfaceK.strikeVolatility()
    
    strikeVol = np.array(strikeVol)
    plt.plot(lst, strikeVol,color='y')
    plt.title("I.V. of Strike")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.show()
    
    # B ii)
    expiryLst = np.linspace(1/12,3,71)
    strikeLst = 0
    value = []
    for i in expiryLst:
        model1 = FastFourierTransforms(S, K, i, r, q, sigma,nu,kappa,rho,theta)
        value += [model1.heston(alpha, N, B, K)[0]]
    value = np.array(value)
    
    volSurfaceT = VolatilitySurfaces(value, strikeLst, expiryLst, S, K, T, r, sigma)
    expiryVol, lst1 = volSurfaceT.expiryVolatility()
    expiryVol = np.array(expiryVol)
    plt.plot(lst1, expiryVol,color='r')
    plt.title("I.V. Term Structure")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Implied Volatility")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.show()
    
    # # B iii)
    # # nu changes
    
    S = 150
    K = 150
    T = 0.25
    r = 0.025
    q = 0
    sigma = 0.4
    alpha = 1
    N = 9
    B = 1000
    nu = [0.005,0.01,0.05,0.075,0.09,0.25]
    kappa = 0.5
    rho = 0.25
    theta = 0.12
    
    
    for nU in nu:
        nuHeston = FastFourierTransforms(S, K, T, r, q, sigma, nU, kappa, rho, theta)
        expiryLst = 0
        strikeLst = np.linspace(50,250,71)
        stockLst = []
        for i in strikeLst:
            stockLst += [nuHeston.heston(alpha, N, B, i)[0]]
        stockLst = np.array(stockLst)
        
        nuVolSurfaceK = VolatilitySurfaces(stockLst, strikeLst, expiryLst, S, K, T, r, sigma)
        strikeVol, lst = nuVolSurfaceK.strikeVolatility()
        
        strikeVol = np.array(strikeVol)
        plt.plot(lst, strikeVol,label=f'nu={nU}')
        plt.title(f"I.V. of Strike, nu")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.legend()
    plt.show()
    
    for nU in nu:
        nuHeston = FastFourierTransforms(S, K, T, r, q, sigma, nU, kappa, rho, theta)    
        # B ii)
        expiryLst = np.linspace(1/12,3,71)
        strikeLst = 0
        value = []
        for i in expiryLst:
            modelnu = FastFourierTransforms(S, K, i, r, q, sigma,nU,kappa,rho,theta)
            value += [modelnu.heston(alpha, N, B, K)[0]]
        value = np.array(value)
        
        nuVolSurfaceT = VolatilitySurfaces(value, strikeLst, expiryLst, S, K, T, r, sigma)
        expiryVol, lst1 = nuVolSurfaceT.expiryVolatility()
        expiryVol = np.array(expiryVol)
        plt.plot(lst1, expiryVol,label=f'nu={nU}')
        plt.title(f"I.V. Term Structure, nu")
        plt.xlabel("Time to Expiry")
        plt.ylabel("Implied Volatility")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.legend()
    plt.show()
    
    # kappa changes
    
    S = 150
    K = 150
    T = 0.25
    r = 0.025
    q = 0
    sigma = 0.4
    alpha = 1
    N = 9
    B = 1000
    nu = 0.09
    kappa = [0.05,0.1,0.25,0.5,1,1.5]
    rho = 0.25
    theta = 0.12
    
    for KAPPA in kappa:
        kappaHeston = FastFourierTransforms(S, K, T, r, q, sigma, nu, KAPPA, rho, theta)
        expiryLst = 0
        strikeLst = np.linspace(50,250,71)
        stockLst = []
        for i in strikeLst:
            stockLst += [kappaHeston.heston(alpha, N, B, i)[0]]
        stockLst = np.array(stockLst)
        
        kappaVolSurfaceK = VolatilitySurfaces(stockLst, strikeLst, expiryLst, S, K, T, r, sigma)
        strikeVol, lst = kappaVolSurfaceK.strikeVolatility()
        
        strikeVol = np.array(strikeVol)
        plt.plot(lst, strikeVol,label=f'kappa={KAPPA}')
        plt.title(f"I.V. of Strike, kappa")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.legend()
    plt.show()
        
        # B ii)
    for KAPPA in kappa:
        kappaHeston = FastFourierTransforms(S, K, T, r, q, sigma, nu, KAPPA, rho, theta)
        expiryLst = np.linspace(1/12,3,71)
        strikeLst = 0
        value = []
        for i in expiryLst:
            modelkappa = FastFourierTransforms(S, K, i, r, q, sigma,nu,KAPPA,rho,theta)
            value += [modelkappa.heston(alpha, N, B, K)[0]]
        value = np.array(value)
        
        kappaVolSurfaceT = VolatilitySurfaces(value, strikeLst, expiryLst, S, K, T, r, sigma)
        expiryVol, lst1 = kappaVolSurfaceT.expiryVolatility()
        expiryVol = np.array(expiryVol)
        plt.plot(lst1, expiryVol,label=f'kappa={KAPPA}')
        plt.title(f"I.V. Term Structure, kappa")
        plt.xlabel("Time to Expiry")
        plt.ylabel("Implied Volatility")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.legend()
    plt.show()
    
    # rho changes
    
    S = 150
    K = 150
    T = 0.25
    r = 0.025
    q = 0
    sigma = 0.4
    alpha = 1
    N = 9
    B = 1000
    nu = 0.09
    kappa = 0.5
    rho = [0.05,0.25,0.5,0.75,1]
    theta = 0.12
    
    for RHO in rho:
        rhoHeston = FastFourierTransforms(S, K, T, r, q, sigma, nu, kappa, RHO, theta)
        expiryLst = 0
        strikeLst = np.linspace(50,250,71)
        stockLst = []
        for i in strikeLst:
            stockLst += [rhoHeston.heston(alpha, N, B, i)[0]]
        stockLst = np.array(stockLst)
        
        rhoVolSurfaceK = VolatilitySurfaces(stockLst, strikeLst, expiryLst, S, K, T, r, sigma)
        strikeVol, lst = rhoVolSurfaceK.strikeVolatility()
        
        strikeVol = np.array(strikeVol)
        plt.plot(lst, strikeVol,label=f'rho={RHO}')
        plt.title(f"I.V. of Strike, rho")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.legend()
    plt.show()
        
        # B ii)
    for RHO in rho:
        rhoHeston = FastFourierTransforms(S, K, T, r, q, sigma, nu, kappa, RHO, theta)
        expiryLst = np.linspace(1/12,3,71)
        strikeLst = 0
        value = []
        for i in expiryLst:
            modelrho = FastFourierTransforms(S, K, i, r, q, sigma,nu,kappa,RHO,theta)
            value += [modelrho.heston(alpha, N, B, K)[0]]
        value = np.array(value)
        
        rhoVolSurfaceT = VolatilitySurfaces(value, strikeLst, expiryLst, S, K, T, r, sigma)
        expiryVol, lst1 = rhoVolSurfaceT.expiryVolatility()
        expiryVol = np.array(expiryVol)
        plt.plot(lst1, expiryVol,label=f'rho={RHO}')
        plt.title(f"I.V. Term Structure, rho")
        plt.xlabel("Time to Expiry")
        plt.ylabel("Implied Volatility")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.legend()
    plt.show()
    
    # theta changes
    
    S = 150
    K = 150
    T = 0.25
    r = 0.025
    q = 0
    sigma = 0.4
    alpha = 1
    N = 9
    B = 1000
    nu = 0.09
    kappa = 0.5
    rho = 0.25
    theta = [0.01,0.05,0.12,0.2,0.3]
    
    
    for THETA in theta:
        thetaHeston = FastFourierTransforms(S, K, T, r, q, sigma, nu, kappa, rho, THETA)
        expiryLst = 0
        strikeLst = np.linspace(50,250,71)
        stockLst = []
        for i in strikeLst:
            stockLst += [thetaHeston.heston(alpha, N, B, i)[0]]
        stockLst = np.array(stockLst)
        
        thetaVolSurfaceK = VolatilitySurfaces(stockLst, strikeLst, expiryLst, S, K, T, r, sigma)
        strikeVol, lst = thetaVolSurfaceK.strikeVolatility()
        
        strikeVol = np.array(strikeVol)
        plt.plot(lst, strikeVol,label=f'theta={THETA}')
        plt.title(f"I.V. of Strike, theta")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.legend()
    plt.show()
        
        # B ii)
    for THETA in theta:
        thetaHeston = FastFourierTransforms(S, K, T, r, q, sigma, nu, kappa, rho, THETA)
        expiryLst = np.linspace(1/12,3,71)
        strikeLst = 0
        value = []
        for i in expiryLst:
            modeltheta = FastFourierTransforms(S, K, i, r, q, sigma,nu,kappa,rho,THETA)
            value += [modeltheta.heston(alpha, N, B, K)[0]]
        value = np.array(value)
        
        thetaVolSurfaceT = VolatilitySurfaces(value, strikeLst, expiryLst, S, K, T, r, sigma)
        expiryVol, lst1 = thetaVolSurfaceT.expiryVolatility()
        expiryVol = np.array(expiryVol)
        plt.plot(lst1, expiryVol,label=f'theta={THETA}')
        plt.title(f"I.V. Term Structure, theta")
        plt.xlabel("Time to Expiry")
        plt.ylabel("Implied Volatility")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.legend()
    plt.show()
    