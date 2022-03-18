"""
Created on Sun Mar. 14 21:05:52 2022
@author: Thomas Kuntz MF-796-HW-4
"""

import cvxpy as opt
import cvxopt as cvx
import scipy as si
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def importData(df):
    print(df.head())
    print(df.tail())
    #print('count of null values',df.isnull().sum())
    #print(df.describe())
    df = df.dropna()
    print(df.describe())
    return df

def minVarPort(mean_rets, CC):
    assets = len(mean_rets)
    args = (mean_rets, CC)
    consts = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(assets))
    res = si.optimize.minimize(portSD, assets*[1./assets,], args=args, method='SLSQP', bounds=bounds, constraints=consts)
    return res

def portSD(w, means, CC):
    port_sd = np.sqrt(np.dot(w.T, np.dot(CC, w)))
    return port_sd

if __name__ == '__main__':

    # problem 1, part a):
    df = pd.read_csv('mf796-hw4-data.csv',index_col='date',infer_datetime_format=True)
    data = importData(df)
    data.isnull().sum()

    # part b)
    dfPct = data.pct_change()
    dfPct = dfPct.dropna()
    print(dfPct.head(10))

    # part c)
    plt.figure(figsize=(17, 11))
    sns.heatmap(dfPct.cov(), cmap="Blues", annot=False)
    plt.show()
    cov_matrix = dfPct.cov()
    eigenValues, eigenVectors = np.linalg.eig(cov_matrix)

    plt.plot(eigenValues)
    plt.show()

    plus = sum(count > 0 for count in eigenValues)
    minus = sum(count < 0 for count in eigenValues)
    print(f'the number of positive eigen values is: {plus}, and the number of negative is: {minus}')

    # part d)
    evSum = eigenValues.sum()
    evCumSum = eigenValues.cumsum()
    temp = evCumSum / evSum
    cum1 = len(temp[temp < 0.5]) + 1
    cum2 = len(temp[temp < 0.9]) + 1
    print(f'the 50% {cum1}, and the 90% {cum2}')

    # part e)
    rets = eigenVectors[:, :cum2]
    resids = dfPct - np.dot(dfPct.values.dot(rets), rets.T)
    plt.plot(resids)
    plt.show()

    # problem 2
    G = np.zeros([2,len(dfPct.columns)])
    G[0, :18] = 1
    G[1, :] = 1
    C = cov_matrix.values
    R = dfPct.mean(axis=0).values
    a=1
    c = np.array([1, 0.1])
    invC = np.linalg.inv(C)
    Lambda = np.linalg.inv(np.dot(G, invC.dot(G.T))).dot(G.dot(invC).dot(R) - 2 * a * c)
    w = 1/2/a*invC.dot((R - G.T.dot(Lambda)))
    plt.plot(range(len(w)), w)
    plt.show()

    # problem 3 part a)
    df1 = pd.read_csv('DataForProblem3.csv',index_col='Date',infer_datetime_format=True)
    df1 = importData(df1)
    plt.plot(df1)
    plt.show()
    df1.isnull().sum()
    #print(df1.head(6))
    #print(df1.tail(6))

    # attempt 1
    P = 1000000   # total wealth
    GG = np.zeros([2, len(df1.columns)])
    GG[0, :9] = 1
    GG[1, :] = 1
    CC = df1.cov()
    #print(CC)
    RR = df1.mean(axis=0).values
    aa = 0.5
    cc = 1
    invCC = np.linalg.inv(CC)
    LL = np.linalg.inv(np.dot(GG, invCC.dot(GG.T))).dot(GG.dot(invCC).dot(RR) - 2 * aa * cc)
    ww = 1 / 2 / aa * invCC.dot((RR - GG.T.dot(LL)))
    plt.plot(range(len(ww)), ww)
    plt.show()
    print(f'The weights: \n{ww} \n and the sum of the weights: {np.sum(ww)}')




    # attempt 2
    print('-------------attempt 2---------------')
    N = 10000
    lstwts = np.zeros((N, len(df1.columns)))
    port_rets = np.zeros((N))
    port_risk = np.zeros((N))


    for i in range(N):
        wts = np.random.uniform(size = len(df1.columns))
        wts = wts/np.sum(wts)
        lstwts[i,:] = wts

        port_ret = np.sum(df1.mean() * wts)
        port_ret = (port_ret + 1)

        port_rets[i] = port_ret

        port_sd = np.sqrt(np.dot(wts.T, np.dot(CC, wts)))
        port_risk[i] = port_sd

    VaR = lstwts[port_risk.argmin()]
    print(VaR)
    print(port_risk.min())
    plt.scatter(port_risk, port_rets)
    plt.show()

    print('-------------------take 3--------------------')
    CC = df1.cov()
    names = df1.columns
    means = df1.mean(axis=0).values
    vari = minVarPort(means, CC)
    resss = pd.DataFrame([round(x,6) for x in vari['x']], index=names).T
    print(vari)
    print()
    sums = np.sum(vari['x'])
    print(sums)
    # this is the optimal weighting
    print(resss)

