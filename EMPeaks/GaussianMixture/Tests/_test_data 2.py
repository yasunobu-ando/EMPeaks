""" TestData Generation from 2 component GMM. """
# License: BSD-3-clause
# Copyright © 2020 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu Ando <yasunobu.ando@aist.go.jp>


import numpy as np
import matplotlib.pyplot as plt


class TestData:
    def __init__(self, n, bins=500):
        self.mu = np.array([3.5, 6.5])
        self.sigma = np.array([1.4, 0.7])
        self.pi = np.array([0.3, 0.7])
        self.n = n
        self.bins = bins

        tmp = np.random.rand(n)
        z = np.round(1.0 / (1.0 + np.exp(-1000 * (tmp - self.pi[0]))), 0)
        x = np.zeros(n)
        for i in range(n):
            if z[i] == 0:
                x[i] = np.random.normal(self.mu[0], np.sqrt(self.sigma[0]))
            else:
                x[i] = np.random.normal(self.mu[1], np.sqrt(self.sigma[1]))

        self.data1D = x
        weights, bins = np.histogram(self.data1D, bins=bins, range=(0, 10))
        self.data2D = np.array([bins[:-1], weights])

    def show_info(self):
        print("-- test_data information ---")
        print("model:               Gaussian mixture model of K = 2")
        print("mu(means):          ", self.mu)
        print("sigma^2(variances): ", self.sigma)
        print("pi(mixing ratio):   ", self.pi)
        print("n(data number):     ", self.n)
        print("bins:               ", self.bins)
        print("\n")

    def plot(self, dpi=100):
        fig = plt.figure(dpi=dpi)
        plt.scatter(self.data2D[0], self.data2D[1], c='gray', marker='o')


if __name__ == "__main__":
    TestData(100).show_info()