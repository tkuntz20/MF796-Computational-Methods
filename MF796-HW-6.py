
import math
import numpy as np
import cmath
import matplotlib.pyplot as plt
import time
from scipy import interpolate

class SimulationInHeston():

    def __init__(self, S, T, r, type):
        self.S = S
        self.T = T
        self.r = r
        self.type = type

    def euroCall(self, S, K):
        if type == 'p':
            payoff = np.maximum(K-S[:,-1], 0)
        else:
            payoff = np.maximum(S[:,-1]-K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price

    def upAndOut(self, S, K1, K2):
        if type == 'p':
            max = np.max(S, axis=1)
            dummy = np.where(max < K2, 1, 0)
            payoff = np.maximum(K1 - S[:, -1], 0) * dummy
        else:
            max = np.max(S, axis=1)
            dummy = np.where(max < K2, 1, 0)
            payoff = np.maximum(S[:, -1] - K1, 0) * dummy
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price

    def upAndOutVr(self, S, K1, K2, euro):
        euPayoff = np.maximum(S[:, -1] - K1, 0)
        max = np.max(S, axis=1)
        dummy = np.where(max < K2, 1, 0)
        upOutPayoff = np.maximum(S[:, -1] - K1, 0) * dummy
        CC = np.cov(euPayoff, upOutPayoff)
        cc = -CC[0][1] / CC[0][0]
        payoff = np.mean(upOutPayoff) + cc * (euPayoff - euro)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price

    def simulatedHeston(self, paramsH, paramsE, N, M, seed=None):
        kappa, theta, sigma, rho, nu = paramsH
        S, T, r, q = paramsE
        dt = T/M
        if seed is not None:
            np.random.seed(seed)

        mu = np.array([0, 0])
        CC = np.array([[dt, dt * rho], [dt * rho, dt]])
        dW = np.random.multivariate_normal(mu, CC, M * N)
        dW1 = dW[:, 0].reshape(N, M)
        dW2 = dW[:, 1].reshape(N, M)

        ss = np.zeros([N, M])
        vv = np.zeros([N, M])
        ss[:, 0] = S
        vv[:, 0] = nu

        for i in range(1,M):
            dv_t = kappa * (theta - np.maximum(vv[:, i-1], 0)) * dt + sigma * np.maximum(vv[:, i-1], 0) ** 0.5 * dW2[:,i-1]
            vv[:,i] = vv[:, i-1] + dv_t
            dS_t = (r-q) * ss[:, i-1] * dt + np.maximum(vv[:, i-1], 0) ** 0.5 * ss[:, i-1] * dW1[:, i-1]
            ss[:, i] = ss[:, i-1] + dS_t
        return ss


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

        return value

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

if __name__ == '__main__':

    kappa = 3.52
    theta = 0.052
    sigma = 1.18
    rho = -0.77
    nu = 0.034
    S = 282
    K = 285
    T = 1
    q = 0.0177
    r = 0.015
    paramsH = [kappa, theta, sigma, rho, nu]
    paramsE = [S, T, r, q]

    # part b
    M = 252
    N = 1000000

    # part c
    sih = SimulationInHeston(S, T, r, type='c')
    simulated = sih.simulatedHeston(paramsH, paramsE, N, M)
    euroC = sih.euroCall(simulated, K)
    FFT = FastFourierTransforms(S, K, T, r, q, sigma, nu, kappa, rho, theta)
    euroFFT = FFT.heston(1.5, N=10, B=K, K=K)
    print(f' Euro Call value via Simulation =  {euroC}')
    print(f' Euro Call Value via FFt =         {euroFFT}\n')

    # part d
    K1 = 285
    K2 = 315
    uao = sih.upAndOut(simulated, K1, K2, )
    print(f' Up and Out call values =   {uao}')

    Ns = np.logspace(1, 4.9, 400)
    UpAndOut = []
    for sample_N in Ns:
        sample_index = np.random.choice(N, int(sample_N), replace=False)
        sample_path = simulated[sample_index, :]
        price = sih.upAndOut(sample_path, K1, K2)
        UpAndOut.append(price)

    Error = [abs(p - uao) for p in UpAndOut]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_title('Option Value vs. # paths')
    ax.set_ylabel('Option Value')
    ax.set_xlabel('# Paths (N)')
    ax.plot(Ns, UpAndOut, color='purple')
    plt.grid(linestyle='--', linewidth=1)
    plt.show()

    # part e
    euro = sih.euroCall(simulated, K)
    Ns = np.logspace(1,4.9,100)
    UpAndOut = []
    UpAndOut_cv = []
    for sample_N in Ns:
        sample_index = np.random.choice(N, int(sample_N), replace=False)
        sample_path = simulated[sample_index, :]
        price = sih.upAndOut(sample_path, K1, K2)
        price_cv = sih.upAndOutVr(sample_path, K1, K2, euro)
        UpAndOut.append(price)
        UpAndOut_cv.append(price_cv)

    R_Error = [abs(p - uao) for p in UpAndOut]
    R_Error_vc = [abs(p - uao) for p in UpAndOut_cv]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_title('Error vs. # Paths')
    ax.set_ylabel('Error')
    ax.set_xlabel('# Paths (N)')
    ax.plot(Ns, R_Error, color='b')
    ax.plot(Ns, R_Error_vc, color='y')
    ax.legend(['Error w/o vc', 'Error w/ vc'])
    plt.grid(linestyle='--', linewidth=1)
    plt.show()



