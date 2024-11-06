# License: BSD-3-clause
# Copyright © 2024 Institute of Science Tokyo
# Author: Yasunobu ANDO

from scipy import optimize
from scipy.special import voigt_profile
from scipy.special import erf
from scipy.integrate import quad
import numpy as np


class Voigt:
    def __init__(self, x_min=-100, x_max=100, sigma_min=0.01, sigma_max=10, gamma_min=0.01, gamma_max=10):
        self.x_min = x_min
        self.x_max = x_max
        self.interval = (self.x_min, self.x_max)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.x0 = np.random.uniform(x_min, x_max)
        self.sigma = np.random.uniform(sigma_min, sigma_max)
        self.gamma = np.random.uniform(gamma_min, gamma_max)
        self.fix_x0 = False
        self.fix_sigma = False
        self.fix_gamma = False

    def set_param(self, **param):
        """
        x0_min, x0_maxの代償関係についてのチェックは未実装
        gamma_min, gamma_maxの代償関係についてのチェックは未実装
        """
        if not param:
            return self
        param_keys = set(param.keys())
        instance_keys = set(self.__dict__.keys())
        for key in (instance_keys & param_keys):
            self.__setattr__(key, param[key])
        return

    def init_model(self):
        if not self.fix_x0:
            self.x0 = np.random.uniform(self.x_min, self.x_max)
        if not self.fix_sigma:
            self.sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        if not self.fix_gamma:   
           self.gamma = np.random.uniform(self.gamma_min, self.gamma_max)
        return

    def predict(self, x):
        """
        x0: center of Voigt Profile
        sigma: gaussian standard derivation
        gamma: Lorentzian Half width Half Maximum
        """
        return voigt_profile(x-self.x0, self.sigma, self.gamma)/self._Z()

    def _Z(self):
        """ Normalization Factor Z """
        def integrand(x):
            return voigt_profile(x-self.x0, self.sigma, self.gamma)
        return quad(integrand, self.x_min, self.x_max)[0]
        
    def cdf(self, x):
        """ Cummulative distribution function """
        def integrand(x):
            return voigt_profile(x-self.x0, self.sigma, self.gamma)
        return quad(integrand, self.x_min, x)[0]

    def _LL(self, x, intensity):
        t = np.log(self.predict(x) + 1.0e-200)
        self.LL = (t * intensity).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, intensity):
        if [self.fix_x0, self.fix_sigma, self.fix_gamma] == [True, True, True]:
            return
        elif [self.fix_x0, self.fix_sigma, self.fix_gamma] == [False, False, False]:
            self.full_optimization(x, intensity)
        elif [self.fix_x0, self.fix_sigma, self.fix_gamma] == [True, False, False]:
            self.full_optimization_fix_x0(x, intensity)
        elif [self.fix_x0, self.fix_sigma, self.fix_gamma] == [False, True, False]:
            self.full_optimization_fix_sigma(x, intensity)
        elif [self.fix_x0, self.fix_sigma, self.fix_gamma] == [False, False, True]:
            self.full_optimization_fix_gamma(x, intensity)
        elif [self.fix_x0, self.fix_sigma, self.fix_gamma] == [True, True, False]:
            self.full_optimization_fix_x0_sigma(x, intensity)
        elif [self.fix_x0, self.fix_sigma, self.fix_gamma] == [True, False, True]:
            self.full_optimization_fix_x0_gamma(x, intensity)
        elif [self.fix_x0, self.fix_sigma, self.fix_gamma] == [False, True, True]:
            self.full_optimization_fix_sigma_gamma(x, intensity)
 
    def full_optimization(self, x, intensity):
        def func_ll(param):
            self.x0 = param[0]
            self.sigma = param[1]
            self.gamma = param[2]
            return - self._LL(x, intensity)

        x0 = np.sum(x*intensity)/np.sum(intensity)
        sigma = self.sigma
        gamma = self.gamma

        init = [x0, sigma, gamma]
        info = optimize.minimize(func_ll, x0=init,
                                    bounds=[(self.x_min, self.x_max), 
                                            (self.sigma_min, self.sigma_max), 
                                            (self.gamma_min, self.gamma_max)],
                                    method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.sigma = info['x'][1]
        self.gamma = info['x'][2]
        return   

    def full_optimization_fix_x0(self, x, intensity):
        def func_ll(param):
            self.sigma = param[0]
            self.gamma = param[1]
            return - self._LL(x, intensity)

        init = [self.sigma, self.gamma]
        info = optimize.minimize(func_ll, x0=init,
                                    bounds=[(self.sigma_min, self.sigma_max), 
                                            (self.gamma_min, self.gamma_max)],
                                    method='L-BFGS-B')
        self.sigma = info['x'][0]
        self.gamma = info['x'][1]
        return   

    def full_optimization_fix_sigma(self, x, intensity):
        def func_ll(param):
            self.x0 = param[0]
            self.gamma = param[1]
            return - self._LL(x, intensity)

        x0 = np.sum(x*intensity)/np.sum(intensity)

        init = [x0, self.gamma]
        info = optimize.minimize(func_ll, x0=init,
                                    bounds=[(self.x_min, self.x_max), 
                                            (self.gamma_min, self.gamma_max)],
                                    method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.gamma = info['x'][1]
        return   
    
    def full_optimization_fix_gamma(self, x, intensity):
        def func_ll(param):
            self.x0 = param[0]
            self.sigma = param[1]
            return - self._LL(x, intensity)

        x0 = np.sum(x*intensity)/np.sum(intensity)

        init = [x0, self.sigma]
        info = optimize.minimize(func_ll, x0=init,
                                 bounds=[(self.x_min, self.x_max), 
                                        (self.sigma_min, self.sigma_max)], 
                                    method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.sigma = info['x'][1]
        return   

    def full_optimization_fix_x0_sigma(self, x, intensity):
        def func_ll(param):
            self.gamma = param[0]
            return - self._LL(x, intensity)

        #sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        #gamma = np.random.uniform(self.gamma_min, self.gamma_max)

        init = [self.gamma]
        info = optimize.minimize(func_ll, x0=init,
                                    bounds=[(self.gamma_min, self.gamma_max)],
                                    method='L-BFGS-B')
        self.gamma = info['x'][0]
        return
    
    def full_optimization_fix_sigma_gamma(self, x, intensity):
        def func_ll(param):
            self.x0 = param[0]
            return - self._LL(x, intensity)

        x0 = np.sum(x*intensity)/np.sum(intensity)

        init = [x0, self.sigma, self.gamma]
        info = optimize.minimize(func_ll, x0=init,
                                    bounds=[(self.x_min, self.x_max)],
                                    method='L-BFGS-B')
        self.x0 = info['x'][0]
        return   
       
    def full_optimization_fix_x0_gamma(self, x, intensity):
        def func_ll(param):
            self.sigma = param[0]
            return - self._LL(x, intensity)

        init = [self.sigma]
        info = optimize.minimize(func_ll, x0=init,
                                    bounds=[(self.sigma_min, self.sigma_max)],
                                    method='L-BFGS-B')
        self.sigma = info['x'][0]
        return   
