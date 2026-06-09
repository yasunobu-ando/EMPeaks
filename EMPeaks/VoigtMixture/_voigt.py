# License: BSD-3-clause
# Copyright © 2024 Institute of Science Tokyo
# Author: Yasunobu ANDO

from scipy import optimize
from scipy.special import voigt_profile
from scipy.special import erf
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
import numpy as np

# スマートCDF用のガウス・ルジャンドル求積（100点）
_GX, _GW = leggauss(100)
# [-1, 1] を [0, 1] にマッピング
_U = 0.5 * _GX + 0.5
_W = 0.5 * _GW

def _tail_integral(limit, direction, x0, sigma, gamma):
    """
    半無限区間のガウス求積 (変数変換 t = limit + direction * u / (1-u))
    direction = -1: (-∞, limit] を積分
    direction = +1: [limit, ∞) を積分
    """
    x = limit + direction * (_U / (1.0 - _U))
    jac = 1.0 / ((1.0 - _U)**2)
    y = voigt_profile(x - x0, sigma, gamma)
    return np.sum(_W * y * jac)

def _smart_cdf(x, x0, sigma, gamma):
    """ピークを跨がないスマートな累積分布関数"""
    if x <= x0:
        return _tail_integral(x, -1, x0, sigma, gamma)
    else:
        return 1.0 - _tail_integral(x, 1, x0, sigma, gamma)


def _voigt_fwhm(sigma, gamma):
    """Voigt プロファイルの FWHM 近似値（Thompson et al.）"""
    f_G = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma  # ガウス FWHM
    f_L = 2.0 * gamma                                 # ローレンツ FWHM
    return 0.5346 * f_L + np.sqrt(0.2166 * f_L**2 + f_G**2)


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
        """
        規格化定数 (x_min から x_max までの積分値)
        スマートCDF（裾野引き算方式）により計算
        """
        return _smart_cdf(self.x_max, self.x0, self.sigma, self.gamma) \
             - _smart_cdf(self.x_min, self.x0, self.sigma, self.gamma)
        
    def cdf(self, x):
        """
        累積分布関数
        スマートCDF（裾野引き算方式）により計算
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        f_min = _smart_cdf(self.x_min, self.x0, self.sigma, self.gamma)
        results = np.array([
            _smart_cdf(xi, self.x0, self.sigma, self.gamma) - f_min
            for xi in x_arr
        ])
        if np.ndim(x) == 0:
            return results.item()
        return results

    def _LL(self, x, intensity):
        t = np.log(self.predict(x) + 1.0e-200)
        self.LL = (t * intensity).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, intensity):
        if [self.fix_x0, self.fix_sigma, self.fix_gamma] == [True, True, True]:
            return
        elif [self.fix_x0, self.fix_sigma, self.fix_gamma] == [False, False, False]:
            self.full_optimization_fix_x0_sigma(x, intensity)
            self.full_optimization_fix_x0_gamma(x, intensity)
            self.full_optimization_fix_sigma_gamma(x, intensity)
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

        init = [x0]
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
