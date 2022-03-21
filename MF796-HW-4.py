"""
Created on Sun Mar. 14 21:05:52 2022
@author: Thomas Kuntz MF-796-HW-4
"""

import cvxopt as opt
from cvxopt import blas, solvers
import scipy as si
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
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

def helper(w, aa ,temp):
    return -1*(temp*RR.dot(w) - aa*np.dot(w,CC.dot(w)))

def MinVarPort(means, CC):
    assets = len(means)
    args = (means, CC)
    consts = ({'type': 'ineq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 0.5)
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
    plt.title('Line Plot of Returns')
    plt.xlabel('time')
    plt.ylabel('Value')
    # plt.grid(linestyle='--', linewidth=0.75)
    plt.plot(data.pct_change())
    plt.show()
    # part b)
    dfPct = data.pct_change()
    dfPct = dfPct.dropna()
    #print(dfPct.head(10))

    # part c)
    plt.figure(figsize=(17, 11))
    sns.heatmap(dfPct.cov(), cmap="Blues", annot=False)
    plt.show()
    cov_matrix = dfPct.cov()
    eigenValues, eigenVectors = np.linalg.eig(cov_matrix)

    plt.title('Line Plot of Eigenvalues')
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.grid(linestyle='--', linewidth=0.75)
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
    plt.title('Line Plot of Residuals')
    plt.xlabel('time')
    plt.ylabel('Value')
    #plt.grid(linestyle='--', linewidth=0.75)
    plt.plot(resids)
    plt.show()

    # problem 2
    G = np.zeros([2, len(dfPct.columns)])
    G[1, :] = 1
    G[0, :17] = 1
    C = cov_matrix.values
    R = dfPct.mean(axis=0).values
    a = 1
    c = np.array([1, 0.1])
    invC = np.linalg.inv(C)
    Lambda = np.linalg.inv(np.dot(G, invC.dot(G.T))).dot(G.dot(invC).dot(R) - (2 * a * c))
    w = 1/2/a*invC.dot((R - G.T.dot(Lambda)))
    plt.title('Plot of Weights')
    plt.xlabel('time')
    plt.ylabel('Value')
    plt.grid(linestyle='--', linewidth=0.75)
    plt.bar(range(len(w)), w)
    plt.show()

    # problem 3 part a)
    df1 = pd.read_csv('DataForProblem3.csv',index_col='Date',infer_datetime_format=True)
    df1 = importData(df1)
    df1.isnull().sum()
    P = 1000000  # total wealth
    GG = np.zeros([2, len(df1.columns)])
    GG[1, :] = 1
    GG[0, :9] = 1
    CC = df1.cov().values
    COR = df1.corr()
    Cinv = np.linalg.inv(CC)
    RR = df1.mean(axis=0).values
    aa = 0.5
    cc = np.array([1, 0.1])



    print('-------------------take 3--------------------')
    CC = df1.cov()
    names = df1.columns
    means = df1.mean(axis=0).values
    vari = MinVarPort(means, CC)
    resss = pd.DataFrame([round(x,6) for x in vari['x']], index=names).T
    print(vari)
    print()
    sums = np.sum(vari['x'])
    print(sums)
    # this is the optimal weighting
    print(resss)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},{'type': 'ineq', 'fun': lambda x: x})
    w0 = np.ones(10)/10
    bounds = (0,1)
    opt1 = si.optimize.minimize(helper, w0,args=bounds, constraints=cons)
    print(opt1)
    print(sum(opt1.x))
    plt.title('Minimum Variance Port.')
    plt.xlabel('Sectors')
    plt.ylabel('Weights')
    plt.grid(linestyle='--', linewidth=0.75)
    plt.scatter(range(len(opt1.x)), opt1.x)
    plt.show()


    # part B
    bounds2 = (0,0.5)
    opt2 = si.optimize.minimize(helper,w0,args=bounds2,constraints=cons)
    print(opt2)
    print(sum(opt2.x))
    plt.title('Optimal Port.')
    plt.xlabel('Sectors')
    plt.ylabel('Weights')
    plt.grid(linestyle='--', linewidth=0.75)
    plt.scatter(range(len(opt2.x)), opt2.x)
    plt.show()

    # part C
    bounds3 = (1, 0)
    cons2 = ({'type': 'ineq', 'fun': lambda x: -(np.sum(x) - 1)}, {'type': 'ineq', 'fun': lambda x: x})
    opt3 = si.optimize.minimize(helper, w0, args=bounds3, constraints=cons2)
    print(opt3)
    print(sum(opt3.x))
    plt.title('Max Return Port.')
    plt.xlabel('Sectors')
    plt.ylabel('Weights')
    plt.grid(linestyle='--', linewidth=0.75)
    plt.scatter(range(len(opt3.x)), opt3.x)
    plt.show()