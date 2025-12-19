import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
kB = 8.61733034e-5


class SetTestData:
    def __init__(self):
        self.K = 2
        self.beta = 0.0833
        self.temp0 = 300
        self.temp1 = 850
        self.P0 = np.array([6, 4])
        self.tau0 = np.array([6.2, 500.0])
        self.Ea = np.array([1.0, 0.6])
        self.dT = 1
        self.T = np.arange(self.temp0, self.temp1, self.dT)
        self.N = self.T.size
        self.Y_dat = generate_data(self.T, self.N, self.K,
                                   self.P0, self.tau0,
                                   self.Ea, self.beta, self.temp0)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.T, self.Y_dat, label='generated data')
        ax.set_xlabel('Temperature [K]')
        ax.set_ylim([0.0, self.Y_dat.max()*1.1])
        ax.set_ylabel('Spectrum Intensity/beta')
        ax.legend()
        plt.show()

    def info(self):
        print("[Information summary of contained data] ")
        print("Mixing number K: ", self.K)
        print("Heating ratio beta: ", self.beta)
        print("Temperature range: ", "from ",self.temp0, "to", self.temp1,"[K]")
        print("Initial Polarization P0: ", self.P0)
        print("Total Initial Polarization P0: ", self.P0.sum())
        print("Exponential pre-factor tau0[us]: ", self.tau0)
        print("Activation Energy Ea[eV]: ", self.Ea)
        print("Temperature interval dT[K]: ", self.dT)
        self.plot()


def f(T, E):
    return np.exp(-E / (kB * T))


def generate_data(T, N, K, P0, tau0, Ea, beta, temp0):
    y = np.zeros(N)
    for i in range(N):
        for k in range(K):
            y[i] += P0[k] / (tau0[k] * 1e-6) * np.exp(-Ea[k] / kB / T[i]
                                                      - 1 / beta / (tau0[k] * 1e-6)
                                                      * integrate.quad(f, temp0, T[i], Ea[k])[0])
    return y/beta


def test():
    test_data = SetTestData()
    test_data.info()


if __name__ == "__main__":
    # execute only if run as a script
    test()
