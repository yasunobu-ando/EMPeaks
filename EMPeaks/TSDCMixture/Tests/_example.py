from ._test_data import TestData
from .._tsdc_mixture import TSDCMixtureModel
import matplotlib.pyplot as plt
import numpy as np
import sys


class Example:
    def __init__(self, K=2):
        if (K < 1) or (K > 5):
            print("In this example, only a number of 1 to 5 is acceptable.")
            sys.exit()
        self.K = K
        self.intensity = TestData(K).intensity
        self.T = TestData(K).T
        self.param = TestData(K).param
        self.model = TSDCMixtureModel(K)


    def leastsq(self):
        print("Start example of leastsq estimation.")
        plt.plot(self.T, self.intensity)
        plt.show()
        print("\n Start sampling process.")
        info = self.model.sampling(self.T, self.intensity, method='leastsq', trial=10, criteria='rmse')
        self.model.plot(self.T, self.intensity/self.model.beta)
        print("average time: {:}s".format(info['time_hist'].mean()))

    def leastsq_tau0(self):
        print("Start example of leastsq_tau0 estimation.")
        plt.plot(self.T, self.intensity)
        plt.show()
        print("\n Start sampling process.")
        info = self.model.sampling(self.T, self.intensity, method='leastsq_tau0', trial=10, criteria='rmse')
        self.model.plot(self.T, self.intensity/self.model.beta)
        print("average time: {:}s".format(info['time_hist'].mean()))

    def adapted_em(self):
        print("Start example of Adapted EM estimation.")
        plt.plot(self.T, self.intensity)
        plt.show()
        print("\n Start sampling process.")
        info = self.model.sampling(self.T, self.intensity, method='adapted_em', r_eps=1.0e-9, trial=1)
        self.model.plot(self.T, self.intensity)
        print("average time: {:}s".format(info['time_hist'].mean()))

    def experimental_data(self, K, fname):
        T, intensity = np.loadtxt(fname, delimiter=',', unpack=True, skiprows=1)
        intensity = intensity*1000
        T = T + 273.15
        plt.plot(T, intensity)
        plt.show()
        model = TSDCMixtureModel(K=K, background='none', Ea_min=0.01, Ea_max=5.0, k_ramp=0)
        info = model.sampling(T, np.abs(intensity), method='leastsq', trial=50, criteria='rmse')
        model.plot(T, intensity)
        plt.show()
        model = TSDCMixtureModel(K=K, background='none', Ea_min=0.01, Ea_max=5.0, k_ramp=0)
        info = model.sampling(T, np.abs(intensity), method='adapted_em', r_eps=1e-6, trial=50)
        info = model.fit(T, np.abs(intensity), method='adapted_em', r_eps=1e-9)
        model.plot(T, intensity)
        plt.show()
