from scipy import interpolate
import numpy as np
import cmath
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')


class FFT:
    def __init__(self, params, T, S0=267.15, r = 0.015, q = 0.0177):

        self.kappa = params[0]
        self.theta = params[1]
        self.sigma = params[2]
        self.rho = params[3]
        self.v0 = params[4]

        self.S0 = S0
        self.r = r
        self.T = T
        self.q = q

    def Heston_fft(self, alpha, K, B, n=11, Ks = None):

        time_begin = time.time()

        ii = complex(0, 1)
        N = 2**n
        dv = B / N
        dk = 2 * np.pi / N / dv
        beta = np.log(self.S0) - dk * N / 2

        vj = np.arange(0, N, dtype=complex) * dv
        km = beta + np.arange(0, N) * dk   # ln(K)

        delta_j_1 = np.zeros(N)
        delta_j_1[0] = 1

        Psi_vj = np.zeros(N, dtype=complex)

        for j in range(0, N):
            u = vj[j] - (alpha + 1) * ii
            numer = np.exp(-ii * beta * vj[j]) * self.Heston_cf(u)
            denom = 2 * (alpha + vj[j] * ii) * (alpha + 1 + vj[j] * ii)
            Psi_vj[j] = numer / denom

        x = (2 - delta_j_1) * dv * Psi_vj
        z = np.fft.fft(x)

        # Option price of a series of K
        Mul = np.exp(-alpha * np.array(km)) / np.pi
        Calls = np.exp(-self.r * self.T) * Mul * np.array(z).real

        # To get the price of specified K
        K_list = list(np.exp(km))
        Call_list = list(Calls)
        tck = interpolate.splrep(K_list, Call_list)
        if Ks is not None:
            price = interpolate.splev(Ks, tck).real
        else:
            price = interpolate.splev(K, tck).real

        # calculat the running time
        time_end = time.time()
        run_time = time_end - time_begin

        return price

    def Heston_cf(self, u):
        sigma = self.sigma
        kappa = self.kappa
        rho = self.rho
        S0 = self.S0
        r = self.r
        T = self.T
        theta = self.theta
        v0 = self.v0

        ii = complex(0, 1)

        lmbd = cmath.sqrt(sigma**2 * (u**2 + ii * u) +
                          (kappa - ii * rho * sigma * u)**2)
        w_nume = np.exp(ii * u * np.log(S0) + ii * u * (r - self.q) * T +
                        kappa * theta * T * (kappa - ii * rho * sigma * u) / sigma**2)
        w_deno = (cmath.cosh(lmbd * T / 2) + (kappa - ii * rho * sigma * u) /
                  lmbd * cmath.sinh(lmbd * T / 2))**(2 * kappa * theta / sigma**2)
        w = w_nume / w_deno
        y = w * np.exp(-(u**2 + ii * u) * v0 / (lmbd /
                                                cmath.tanh(lmbd * T / 2) + kappa - ii * rho * sigma * u))

        return y

    def NB_plot(self, n_list, B_list, alpha, K, true_price, plot=True):

        xx, yy = np.meshgrid(n_list, B_list)
        pp = np.zeros([len(n_list), len(B_list)])
        tt = np.zeros([len(n_list), len(B_list)])
        tt_dict = {}

        for i, n in enumerate(n_list):
            for j, B in enumerate(B_list):
                price, time, _ = self.Heston_fft(alpha, n, B, K)
                pp[i, j] = price
                tt[i, j] = 1 / ((price - true_price)**2 * time)
                tt_dict[str(n) + '_' + str(B)] = tt[i, j]

        if plot:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(xx, yy, pp.T, rstride=1, cstride=1, cmap='rainbow')
            plt.title("European Call Option Price v.s N & B")
            ax.set_xlabel("N")
            ax.set_ylabel("B")
            ax.set_zlabel("FFT European Call Option Price")

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(xx, yy, tt.T, rstride=1, cstride=1, cmap='rainbow')
            plt.title("FFT Efficiency v.s N & B")
            ax.set_xlabel("N")
            ax.set_ylabel("B")
            ax.set_zlabel("FFT Efficiency")

        max_eff = max(tt_dict, key=lambda x: tt_dict[x])
        max_eff_n = max_eff.split('_')[0]
        max_eff_B = max_eff.split('_')[1]
        print(
            'The point of Max Efficiency: n={}, B={}'.format(
                max_eff_n, max_eff_B))