# License: BSD-3-clause
# Copyright © 2020-2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu ANDO

from scipy import optimize
from scipy.stats import norm
from scipy.stats import cauchy
import numpy as np


class PseudoVoigt:
    def __init__(self, x_min=-100, x_max=100, gamma_min=0.01, gamma_max=75, eta_min=0.5, eta_max=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.interval = (self.x_min, self.x_max)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.x0 = np.random.uniform(x_min, x_max)
        self.gamma = np.random.uniform(gamma_min, gamma_max)
        self.eta = np.random.uniform(eta_min, eta_max)
        self.fix_x0 = False
        self.fix_gamma = False
        self.fix_eta = False

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
        if not self.fix_gamma:
            self.gamma = np.random.uniform(self.gamma_min, self.gamma_max)
        if not self.fix_eta:
            np.random.uniform(self.eta_min, self.eta_max)
        return

    def predict(self, x):
        _sigma = self.gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        standard_model = self.eta * norm.pdf(x, self.x0, _sigma) \
            + (1 - self.eta) * cauchy.pdf(x, self.x0, self.gamma / 2.0)
        truncated_model = standard_model/self._Z()
        return truncated_model
    
    def _Z(self):
        return self._cdf(self.x_max)  - self._cdf(self.x_min)

    def _cdf(self, x):
        """ Cummulative distribution function of standard model, not truncated one. """
        _sigma = self.gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        cdf = self.eta * norm.cdf(x, self.x0, _sigma)
        cdf += (1.0 - self.eta) * 1.0/np.pi \
                * (np.arctan((x - self.x0)/(self.gamma/2.0)) 
                - np.arctan((self.x_min - self.x0)/(self.gamma/2.0)))
        return cdf

    def _LL(self, x, intensity):
        t = np.log(self.predict(x) + 1.0e-20)
        self.LL = (t * intensity).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, intensity, max_iter=5000, eps=1e-8):
        if self.fix_eta:
            if self.fix_x0 * self.fix_gamma:
                return
            elif self.fix_x0:
                self.full_optimization_fix_x0_eta(x, intensity)
            elif self.fix_gamma:
                self.full_optimization_fix_gamma_eta(x, intensity)
            else:
                self.full_optimization_fix_gamma_eta(x, intensity)
                self.full_optimization_fix_x0_eta(x, intensity)
                self.full_optimization_fix_eta(x, intensity)
        # self.full_optimization(x, intensity, max_iter=max_iter, eps=eps)
        else:
            if self.fix_x0:
                self.full_optimization_fix_gamma_eta(x, intensity)
                self.full_optimization_fix_x0_eta(x, intensity)
                self.full_optimization_fix_x0(x, intensity)
            elif self.fix_gamma:
                self.full_optimization_fix_gamma_eta(x, intensity)
                self.full_optimization_fix_x0_gamma(x, intensity)
                self.full_optimization_fix_gamma(x, intensity)
            else:
                self.full_optimization_fix_x0_eta(x, intensity)
                self.full_optimization_fix_x0_gamma(x, intensity)
                self.full_optimization_fix_gamma_eta(x, intensity)
                self.full_optimization(x, intensity)
        return

    def full_optimization_fix_eta(self, x, intensity):
        def func_ll(param):
            self.x0 = param[0]
            self.gamma = param[1]
            return - self._LL(x, intensity)

        init = [self.x0, self.gamma]
        ll = [self._LL(x, intensity)]
        info = optimize.minimize(func_ll, 
                                 x0=init,
                                 bounds=[(self.x_min, self.x_max), (self.gamma_min, self.gamma_max)],
                                 method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.gamma = info['x'][1]
        return

    def full_optimization_fix_gamma_eta(self, x, intensity):
        def func_ll(param):
            self.x0 = param
            return - self._LL(x, intensity)

        init = [self.x0]
        ll = [self._LL(x, intensity)]
        info = optimize.minimize(func_ll, 
                                 x0=init,
                                 bounds=[(self.x_min, self.x_max)],
                                 method='L-BFGS-B')
        self.x0 = info['x'][0]
        return

    def full_optimization_fix_x0_eta(self, x, intensity):
        def func_ll(param):
            self.gamma = param
            return - self._LL(x, intensity)

        init = [self.gamma]
        ll = [self._LL(x, intensity)]
        info = optimize.minimize(func_ll, 
                                 x0=init,
                                 bounds=[(self.gamma_min, self.gamma_max)],
                                 method='L-BFGS-B')
        self.gamma = info['x'][0]
        return

    def full_optimization_fix_x0_gamma(self, x, intensity):
        def func_ll(param):
            self.eta = param
            return - self._LL(x, intensity)

        init = [self.eta]
        ll = [self._LL(x, intensity)]
        info = optimize.minimize(func_ll, 
                                 x0=init,
                                 bounds=[(self.eta_min, self.eta_max)],
                                 method='L-BFGS-B')
        self.eta = info['x'][0]
        return

    def full_optimization(self, x, intensity):
        def func_ll(param):
            self.x0 = param[0]
            self.gamma = param[1]
            self.eta = param[2]
            return - self._LL(x, intensity)

        def derivative(param):
            self.x0 = param[0]
            self.gamma = param[1]
            self.eta = param[2]
            return -np.array([self._LL_m(x,intensity), 
                             self._LL_g(x,intensity), 
                             self._LL_eta(x,intensity)])

        x0 = np.sum(x*intensity)/np.sum(intensity)
        init = [x0, self.gamma, self.eta]
        info = optimize.minimize(func_ll, x0=init, #jac=derivative,
                                            bounds=[(self.x_min, self.x_max), 
                                            (self.gamma_min, self.gamma_max),
                                            (self.eta_min, self.eta_max)],
                                    method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.gamma = info['x'][1]
        self.eta = info['x'][2]
        return   

    def full_optimization_fix_x0(self, x, intensity):
        def func_ll(param):
            self.gamma = param[0]
            self.eta = param[1]
            return - self._LL(x, intensity)

        init = [self.gamma, self.eta]
        ll = [self._LL(x, intensity)]
        info = optimize.minimize(func_ll, x0=init,
                                bounds=[(self.gamma_min, self.gamma_max), 
                                        (self.eta_min, self.eta_max)],
                                method='L-BFGS-B')
        self.gamma = info['x'][0]
        self.eta = info['x'][1]

        return
    
    def full_optimization_fix_gamma(self, x, intensity):
        def func_ll(param):
            self.x0 = param[0]
            self.eta = param[1]
            return - self._LL(x, intensity)

        def derivative(param):
            self.x0 = param[0]
            self.eta = param[1]
            return -np.array([self._LL_m(x,intensity), 
                             self._LL_eta(x,intensity)])

        x0 = np.sum(x*intensity)/np.sum(intensity)
        init = [x0, self.eta]
        info = optimize.minimize(func_ll, x0=init, jac=derivative,
                                bounds=[(self.x_min, self.x_max), 
                                        (self.eta_min, self.eta_max)],
                                method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.eta = info['x'][1]
        return   
    
    def _LL_m(self, x,intensity):
        """ derivative of LL in respect to x_0 """
        _N = self.eta * self._N_m(x)
        _L = (1-self.eta) * self._L_m(x)
        return (intensity/self.predict(x)*(_N - _L)).sum()

    def _LL_g(self, x,intensity):
        """ derivative of LL in respect to gamma """
        _N = self.eta * self._N_g(x)
        _L = (1-self.eta) * self._L_g(x)
        return (intensity/self.predict(x)*(_N - _L)).sum()

    def _LL_eta(self, x, intensity):
        """ derivative of LL in respect to eta """
        _sigma = self.gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        _N = norm.pdf(x, self.x0, _sigma)
        _L = cauchy.pdf(x, self.x0, self.gamma / 2.0)
        return (intensity/self.predict(x)*(_N - _L)).sum()
        
    def _N_m(self,x):
        """ derivative of Gaussian in respect to x_0 """
        _sigma = self.gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return norm.pdf(x, self.x0, _sigma) * 2.0 * np.log(2.0)*(x - self.x0)/(self.gamma/2)**2
    
    def _N_g(self,x):
        """ derivative of Gaussian in respect to gamma """
        _sigma = self.gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return 1.0/self.gamma**3 * (-self.gamma**2 + 8.0*np.log(2.0)*(x-self.x0)**2) \
               * norm.pdf(x, self.x0, _sigma) 
    
    def _L_m(self, x):
        return 2*(x-self.x0)/((x-self.x0)**2+(self.gamma/2)**2) * cauchy.pdf(x, self.x0, self.gamma / 2.0)
    
    def _L_g(self, x):
        return (1.0/self.gamma - np.pi*cauchy.pdf(x, self.x0, self.gamma / 2.0)) \
            * cauchy.pdf(x, self.x0, self.gamma / 2.0)
    
