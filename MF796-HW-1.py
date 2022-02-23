# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:32:55 2022

@author: Thomas Kuntz
"""

# Thomas Kuntz MF796 hw-1

import pandas as pd
import pandas_datareader
import numpy as np
from math import *
import QuantLib as ql
import scipy as sp
import scipy.stats as si
import statsmodels.api as sm
import seaborn as sns
import sympy as sy
from tabulate import tabulate
from pandas_datareader import data
import matplotlib.pyplot as plt
from sympy.stats import Normal, cdf
import urllib.request
import zipfile

# Analytic call price
def euroCall(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call, d1, d2

# Left-point Riemann rule
def euroCallLeftRiemann(S, K, T, r, sigma,N,a):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    sums = 0
    for i in range(1,N+1):
        x = a + (d1-a)/N * (i-1)
        sums += si.norm.pdf(x,0,1)
    integrated1 = sums * (d1-a)/N
    
    sums1 = 0
    for i in range(1,N+1):
        x = a + (d2-a)/N * (i-1)
        sums1 += si.norm.pdf(x,0,1)
    integrated2 = sums1 * (d2-a)/N
    
    return S * integrated1 - K * np.exp(-r * T) * integrated2

def euroCallMidPtRiemann(S, K, T, r, sigma,N,a):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    W = (d1-a)/N
    sums = 0
    for i in range(1,N+1):
        x = a + W * (i + 1/2)
        sums += si.norm.pdf(x,0,1)
    integrated1 = sums * W
    
    W1 = (d2-a)/N
    sums1 = 0
    for i in range(1,N+1):
        x = a + W1 * (i + 1/2)
        sums1 += si.norm.pdf(x,0,1)
    integrated2 = sums1 * W1
    
    return S * integrated1 - K * np.exp(-r * T) * integrated2

def euroCallGaussian(S, K, T, r, sigma,N,a):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    nodes, w = np.polynomial.legendre.leggauss(N)
    #print(nodes)
    sums = 0
    t = 0.5*(nodes+1)*(d1+10)-10
    sums = sum(w*si.norm.pdf(t))*0.5*(d1+10)
    
    sums1 = 0
    t = 0.5*(nodes+1)*(d2+10)-10
    sums1 = sum(w*si.norm.pdf(t))*0.5*(d2+10)
    
    return S * sums - K * np.exp(-r * T) * sums1

# vanilla contingent claim 
def vanillaOption(S,K,T,r,sigma,a,b,N,method):
    W = (b-a)/N
    sums = 0
    for i in range(N):
        x = a+W*(i-0.5)
        sums += vanillaIntegral(S,K,T,r,sigma,x)
    return sums*W

def vanillaIntegral(S,K,T,r,sigma,x):  #(S,K,T,r,sigma,x)
    return (np.exp(-r*T)*(x-K)/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-(S+r*T))**2/(2*sigma**2)))

# contingent claim - part(b)
def ContingentCall(S,K1,K2,T1,T2,r,sigma1,sigma2,N,method,rho):
    W1 = ((S + 10*sigma1) - K1) / N  #3*sigma1
    W2 = (K2 - (S - 10*sigma2)) / N
    sums = 0
    for i in range(N):
        for z in range(N):
            sums += method(S,K1,K2,T1,T2,r,sigma1,sigma2,K1 + (i + 0.5) * W1, (S - 10*sigma2) + (z + 0.5) * W2, rho)
    return sums * W2 * W1

def contingentIntegral(S,K1,K2,T1,T2,r,sigma1,sigma2,s1,s2,rho):
    return np.exp(-r*T1) * (s1-K1) / (2*np.pi*sigma1*sigma2*np.sqrt(1-rho**2)) * np.exp((-(s1-S)**2/(sigma1**2)-(s2-S)**2/(sigma2**2)+2*rho*(s2-S)*(s1-S)/(sigma1*sigma2))/(2*(1-rho**2))) #integral 




if __name__ == '__main__': 
    Call, d1, d2 = euroCall(10,12,3/12,0.01,0.2)
    print("Part (a) the analytic price is :",Call)
    
    LRcall1 = euroCallLeftRiemann(10, 12, 3/12, 0.01, 0.2,5,-10)
    print(f'call value {LRcall1}, error value = {abs(Call-LRcall1)}')
    
    LRcall2 = euroCallLeftRiemann(10, 12, 3/12, 0.01, 0.2,10,-10)
    print(f'call value {LRcall2}, error value = {abs(Call-LRcall2)}')
    
    LRcall3 = euroCallLeftRiemann(10, 12, 3/12, 0.01, 0.2,50,-10)
    print(f'call value {LRcall3}, error value = {abs(Call-LRcall3)}')
    print('+++++++++++++++++++++++++++++++++++Mid-point values+++++++++++++++++++++++++++++++++++++++++')
    MPcall1 = euroCallMidPtRiemann(10, 12, 3/12, 0.01, 0.2,5,-10)
    print(f'call value {MPcall1}, error value = {abs(Call-MPcall1)}')
    
    MPcall2 = euroCallMidPtRiemann(10, 12, 3/12, 0.01, 0.2,10,-10)
    print(f'call value {MPcall2}, error value = {abs(Call-MPcall2)}')
    
    MPcall3 = euroCallMidPtRiemann(10, 12, 3/12, 0.01, 0.2,50,-10)
    print(f'call value {MPcall3}, error value = {abs(Call-MPcall3)}')
    print('+++++++++++++++++++++++++++++++++++Gaussian values+++++++++++++++++++++++++++++++++++++++++')
    
    gaussCall1 = euroCallGaussian(10, 12, 3/12, 0.01, 0.2,5,-10)
    print(f'call value {gaussCall1}, error value = {abs(Call-gaussCall1)}')
    
    gaussCall2 = euroCallGaussian(10, 12, 3/12, 0.01, 0.2,10,-10)
    print(f'call value {gaussCall2}, error value = {abs(Call-gaussCall2)}')
    
    gaussCall3 = euroCallGaussian(10, 12, 3/12, 0.01, 0.2,50,-10)
    print(f'call value {gaussCall3}, error value = {abs(Call-gaussCall3)}')
    print('+++++++++++++++++++++++++++++++++++error calculations+++++++++++++++++++++++++++++++++++++++++')
    #error calculations
    Lpoint=[]
    Mpoint=[]
    gausian=[]
    for m in list(range(1,30)):
        LRcall = euroCallLeftRiemann(10, 12, 3/12, 0.01, 0.2,m,-10)
        Lpoint.append(abs(Call-LRcall))
        MPcall = euroCallMidPtRiemann(10, 12, 3/12, 0.01, 0.2,m,-10)
        Mpoint.append(abs(Call-MPcall))
        gaussCall = euroCallGaussian(10, 12, 3/12, 0.01, 0.2,m,-10)
        gausian.append(abs(Call-gaussCall))
        
    plt.plot(Lpoint[4:], label='error left-point')
    x = np.arange(4,30)
    plt.plot(1/x**2, label='1/X^2')
    plt.plot(1/x**3, label='1/X^3')
    plt.legend()
    plt.title('Left-Point Errors')
    plt.show()
    
    plt.plot(Mpoint[4:], label='error midpoint')
    plt.plot(1/x**2, label='1/X^2')
    plt.plot(1/x**3, label='1/X^3')
    plt.legend()
    plt.title('Mid-Point Errors')
    plt.show()
    
    plt.plot(gausian[4:], label='error Gauss')
    plt.plot(1/x**x, label='1/X^X')
    plt.plot(1/x**(x*2), label='1/X^2X')
    plt.legend()
    plt.title('Gaussian-node Errors')
    plt.show()
    
    print('+++++++++++++++++++++++++++++++++++Vanilla call+++++++++++++++++++++++++++++++++++++++++')
    x = 0.0
    vanillaIntegral(380,380,1,0,20,x)
    vanillaCall =  vanillaOption(380,380,1,0,20,380,380 + 3*20,10000,vanillaIntegral)
    print(vanillaCall)
    
    print('+++++++++++++++++++++++++++++++++++contingent call+++++++++++++++++++++++++++++++++++++++++')
    s1= 0.0
    s2= 0.0
    rho = [0.95,0.8,0.5,0.2]
    for i in rho:
        contingentIntegral(380,380,375,1,6/12,0.0,20,15,s1,s2,i)
        contingentCall = ContingentCall(380,380,375,1,6/12,0.0,20,15,100,contingentIntegral,i)
        print(f'contingent option, p = {i}:  {contingentCall}')
        
    contingentIntegral(380,380,375,1,6/12,0.0,20,15,s1,s2,0.95)
    contingentCall4 = ContingentCall(380,380,370,1,6/12,0.0,20,15,100,contingentIntegral,0.95)
    print(f'contingent option, p = 0.95, 6-Mo K = 370:  {contingentCall4}')
    
    contingentCall5 = ContingentCall(380,380,360,1,6/12,0.0,20,15,100,contingentIntegral,0.95)
    print(f'contingent option, p = 0.95, 6-Mo K = 360:  {contingentCall5}')
        
    contingentCall6 = ContingentCall(380,380,360,1,6/12,0.0,20,15,100,contingentIntegral,0.0)
    print(f'contingent option, p = 0.0, 6-Mo K = 360, and K2 needs to be changed to (S + 10*sigma1) in the weighting:  {contingentCall6}')
    
    
    
    
    
    