# License: BSD-3-clause
# Copyright © 2020-2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu ANDO

from EMPeaks.EMCore._em_core import EMCore
from EMPeaks.GaussianMixture._gaussian import Gaussian
from EMPeaks.Background import UniformModel, SquareRootModel, LinearModel, TriangleModel, RampModel
import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import copy
import time


class GaussianMixtureModel(EMCore):
    def __init__(self, K=2, x_min=-300, x_max=300, sigma_min=0.1, sigma_max=50, background='none', k_ramp=5):
        super().__init__(K=K, x_min=x_min, x_max=x_max, background=background, k_ramp=k_ramp)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model[0:K] = [Gaussian(x_min, x_max, sigma_min, sigma_max) for k in range(self.K)]

    def set_single_params(self, **param):
        # setting parameters for each single Gaussian model.
        param_set = {"mu", "sigma"}
        single_params = self.extract_single_params(param_set, **param)
        self.model = [Gaussian(self.x_min, self.x_max, self.sigma_min, self.sigma_max) for k in range(self.K)]
        [self.model[k].set_param(**single_params[k]) for k in range(self.K)]
        return

    def export_single_params(self, _tmp_param):
        """ parameters are sorted in the order of the first element in param_set."""
        param_set = {"mu", "sigma"}
        _tmp = {}
        for param in list(param_set):
            _tmp[param] = [self.model[k].__dict__[param] for k in range(self.K)]

        _tmp_index = np.array(_tmp[list(param_set)[0]]).argsort()

        for param in list(param_set):
            _tmp_param[param] = list(np.array(_tmp[param])[_tmp_index])

        return _tmp_param, _tmp_index

    def add_hist_model(self, info, hist_model, trial):
        info.update({'mu_hist': np.array([hist_model[i]['mu'] for i in range(trial)]),
                     'sigma_hist': np.array([hist_model[i]['sigma'] for i in range(trial)])})
        return

    def print_param_summary(self, param):
        print('   mu:        ' + ('{:5.3f} eV       ' * len(param['mu'])).format(*param['mu']))
        print('   sigma:     ' + ('{:6.3e}          ' * len(param['sigma'])).format(*param['sigma']))
        print('   N_tot:   {:6.3e} '.format(self.N_tot))
        print('   N:         ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi'] * self.N_tot))
        print('   pi:        ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
        return

# class Sharley():
#     def __init__(self, K, x_min, x_max):
#         self.K = K
#         self.x_min = x_min
#         self.x_max = x_max
#         self.dx = 1.0
#         self.pi = np.ones(K)/K
#         self.peak_model = GaussianMixtureModel(K, x_min, x_max)
#
#     def predict(self, x):
#         p = np.sum([self.peak_model.pi[k] * self.peak_model.model[k].cdf(x) for k in range(self.K)], axis=0)
#         z = integrate.trapz(p, x)
#         return p/z
#
#     def _LL(self, x, weight):
#         t = np.log(self.predict(x))
#         self.LL = (t * weight).sum()
#         return self.LL
#
#     def maximum_likelihood_estimation(self, x, weight):
#         return
