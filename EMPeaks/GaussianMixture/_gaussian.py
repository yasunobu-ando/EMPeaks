import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class Gaussian:
    def __init__(self, x_min=-100, x_max=100, sigma_min=0.1, sigma_max=10):
        self.x_min = x_min
        self.x_max = x_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mu, self.sigma = self.init_param()

    def set_param(self, **param):
        """
        x_min, x_maxの代償関係についてのチェックは未実装
        sigma_min, sigma_maxの代償関係についてのチェックは未実装
        """
        if not param:
            return self
        param_keys = set(param.keys())
        instance_keys = set(self.__dict__.keys())
        for key in (instance_keys & param_keys):
            self.__setattr__(key, param[key])
        return

    def init_param(self):
        mu = np.random.uniform(self.x_min, self.x_max)
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        return mu, sigma

    def init_model(self):
        self.mu = np.random.uniform(self.x_min, self.x_max)
        self.sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        return

    def predict(self, x):
        z = np.sqrt(2.0 * np.pi * self.sigma**2)
        x_p = (x - self.mu) / (np.sqrt(2.0) * self.sigma)
        return np.exp(-x_p**2) / z

    def log_likelihood(self, x, intensity):
        return np.sum(intensity * np.log(self.predict(x) + 1e-200))

    def plot(self, x, n_factor=1.0):
        plt.plot(x, n_factor * self.predict(x))
        return

    def maximum_likelihood_estimation(self, x, intensity):
        self.mu = np.sum(intensity * x) / (np.sum(intensity) + 1e-100)
        self.sigma = np.sqrt(np.sum(intensity * (x - self.mu)**2) / (np.sum(intensity) + 1e-100))
        if self.sigma < 1.0e-5:
            print("sigma becomes 0. Reset the parameter again.")
            self.init_model()
        return

    def cdf(self, x):
        return norm.cdf(x, self.mu, self.sigma)
