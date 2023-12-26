from ._test_data import TestData
from .._tsdc_mixture import TSDCMixtureModel
import matplotlib.pyplot as plt
import numpy as np


class Example:
    def __init__(self, K=2):
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
        self.model.plot(self.T, self.intensity)
        print("average time: {:}s".format(info['time_hist'].mean()))

    def leastsq_tau0(self):
        print("Start example of leastsq_tau0 estimation.")
        plt.plot(self.T, self.intensity)
        plt.show()
        print("\n Start sampling process.")
        info = self.model.sampling(self.T, self.intensity, method='leastsq_tau0', trial=10, criteria='rmse')
        self.model.plot(self.T, self.intensity)
        print("average time: {:}s".format(info['time_hist'].mean()))

    def adapted_em(self):
        print("Start example of Adapted EM estimation.")
        plt.plot(self.T, self.intensity)
        plt.show()
        print("\n Start sampling process.")
        info = self.model.sampling(self.T, self.intensity, method='adapted_em', r_eps=1.0e-9, trial=1)
        self.model.plot(self.T, self.intensity)
        print("average time: {:}s".format(info['time_hist'].mean()))

    def experimental_data(self, fname):
        T, intensity = np.loadtxt(fname, delimiter=',', unpack=True)
        T = T + 273.15
        model = TSDCMixtureModel(K=5, background='none', Ea_min=0.01, Ea_max=5.0, k_ramp=0)
        info = model.sampling(T, np.abs(intensity), method='leastsq', trial=50, criteria='rmse')
        model.plot(T, intensity)
        plt.show()
        plt.ylim(0, 10000)
        model = TSDCMixtureModel(K=5, background='none', Ea_min=0.01, Ea_max=5.0, k_ramp=0)
        info = model.sampling(T, np.abs(intensity), method='adapted_em', r_eps=1e-6, trial=100)
        info = model.fit(T, np.abs(intensity), method='adapted_em', r_eps=1e-9)
        model.plot(T, intensity)
        plt.show()
        model = TSDCMixtureModel(K=5, background='none', Ea_min=0.01, Ea_max=2.0, k_ramp=0)
        info = model.sampling(T, np.abs(intensity), method='adapted_em', r_eps=1e-6, trial=100)
        info = model.fit(T, np.abs(intensity), method='adapted_em', r_eps=1e-9)
        model.plot(T, intensity)
        plt.show()
