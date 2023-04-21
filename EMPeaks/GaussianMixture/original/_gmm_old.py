# License: BSD-3-clause
# Copyright © 2020 National Institute of Advanced Industrial Science and Technology (AIST)

import numpy as np


class GaussianMixtureModel:
    def __init__(self, data2d, K=2):
        self.dat = data2d
        self.gmm_K_ = K
        self.gmm_mu_ = np.random.uniform(self.dat[0].min(), self.dat[0].max(), self.gmm_K_)
        self.gmm_sigma_ = np.random.uniform(0, 1.0, self.gmm_K_)
        self.gmm_pi_ = self.get_pi()
        self.param = [{'mu': self.gmm_mu_[k], 'sigma': self.gmm_sigma_[k], 'pi': self.gmm_pi_[k]} for k in range(K)]

    def init_params(self):
        self.gmm_mu_ = np.random.uniform(self.dat[0].min(), self.dat[0].max(), self.gmm_K_)
        self.gmm_sigma_ = np.random.uniform(0, 1.0, self.gmm_K_)
        self.gmm_pi_ = self.get_pi()
        self.param = [{'mu': self.gmm_mu_[k],
                       'sigma': self.gmm_sigma_[k],
                       'pi': self.gmm_pi_[k]} for k in range(self.gmm_K_)]

    def get_pi(self):
        pi = np.zeros(self.gmm_K_)
        max_value = 1.0
        for i in range(self.gmm_K_-1):
            pi[i] = np.random.uniform(0, max_value, 1)
            max_value -= pi[i]
        pi[-1] = max_value
        return pi

    def get_gmm_param(self):
        print("--Parameters of GMM--")
        print("mu(means):          ", self.gmm_mu_)
        print("sigma^2(variance):  ", self.gmm_sigma_)
        print("pi(mixing ratio):   ", self.gmm_pi_)
        print("\n")
