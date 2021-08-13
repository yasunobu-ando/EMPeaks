import numpy as np
from scipy import integrate, optimize, stats
import matplotlib.pyplot as plt


class Lorentzian:
    def __init__(self, x_min=300, x_max=1000, gamma_min=0.1, gamma_max=1000):
        self.x_min = x_min
        self.x_max = x_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.init_model()

    def set_param(self, **param):
        """
        Xp_min, Xp_maxの大小関係についてのチェックは未実装
        Ea_min, Ea_maxの大小関係についてのチェックは未実装
        """
        if not param:
            return self
        param_keys = set(param.keys())
        instance_keys = set(self.__dict__.keys())
        for key in (instance_keys & param_keys):
            self.__setattr__(key, param[key])
        return

    def init_model(self):
        self.x0 = np.random.uniform(self.x_min, self.x_max)
        self.gamma = np.random.uniform(0, 100)

    def set_param_empirical(self, x, intensity):
        x0_mu = np.sum(intensity * x) / np.sum(intensity)
        x0_sigma = np.sqrt(np.sum(intensity * (x ** 2)) / np.sum(intensity) - x0_mu ** 2)

        self.x0 = np.random.normal(x0_mu, x0_sigma, 1)
        self.gamma = np.random.uniform(0, 100)
        return

    def predict(self, x):
        prob = 1.0/(np.pi * self._Z()) * (1.0 * self.gamma)/((x-self.x0)**2 + (0.50 * self.gamma)**2)
        return prob / integrate.trapz(prob, x)

    def log_likelihood(self, x, intensity):
        return np.sum(intensity * np.log(self.predict(x) + 1e-200))

    def plot(self, x, n_factor=1.0):
        plt.plot(x, n_factor * self.predict(x))
        return

    def _Z(self):
        """ Normalization factor of interval [x_min, x_max] """
        return 1.0/np.pi*np.arctan((self.x_max - self.x0)/(self.gamma/2.0)) \
               - 1.0/np.pi*np.arctan((self.x_min - self.x0)/(self.gamma/2.0))

    def cdf(self, x):
        """ Cumulative distribution function """
        return 1.0/self._Z() * 1.0/np.pi*(np.arctan((x - self.x0)/(self.gamma/2.0))
                                          - np.arctan((self.x_min - self.x0)/(self.gamma/2.0)))

    def _logZ_x0(self):
        """ Derivative of Log Z with respect to x0. """
        A = 1.0 / (np.pi * self._Z()) * (1.0 * self.gamma) / ((self.x_max - self.x0) ** 2 + (0.50 * self.gamma) ** 2)
        B = 1.0 / (np.pi * self._Z()) * (1.0 * self.gamma) / ((self.x_min - self.x0) ** 2 + (0.50 * self.gamma) ** 2)
        return -A + B

    def _logZ_gamma(self):
        """ Derivative of Log Z with respect to gamma. """
        _a = self.x_min
        _b = self.x_max
        return 1.0/(2.0 * np.pi * self._Z()) * (
                (self.x0 - _b)/((self.gamma/2)**2 + (self.x0 - _b)**2)
                - (self.x0 - _a)/((self.gamma/2)**2 + (self.x0 - _a)**2)
                )

    def _LL_x0(self, x, intensity):
        """ Derivative of log-likelihood with respect to x0. """
        N = intensity.sum()
        return -np.sum((2.0*intensity*(x - self.x0))/((x - self.x0)**2 + (self.gamma/2.0)**2)) \
               - N * self._logZ_x0()

    def _LL_gamma(self, x, intensity):
        """ Derivative of log-likelihood with respect to gamma. """
        N = intensity.sum()
        value = N/self.gamma
        value += - np.sum(intensity*(self.gamma/2.0)/((x - self.x0)**2 + (self.gamma/2.0)**2))
        value += - N * self._logZ_gamma()
        return value

    def _opt_gamma(self, x0, _interval_g, x, intensity):
        """ Finding gamma for given x0. """
        def _f(gamma):
            self.gamma = gamma
            self.x0 = x0
            return self._LL_gamma(x, intensity)
        return optimize.brentq(_f, _interval_g[0], _interval_g[1])

    def maximum_likelihood_estimation(self, x, intensity, n_partition_x0=100):
        #print('MLE via root search method.')
        #self.root_search(x, intensity, n_partition_x0)
        #print('MLE via bfgs method.')
        self.minimize_bfgs(x, intensity)
        return

    def root_search(self, x, intensity, n_partition_x0=100):
        _interval_x0 = (self.x_min, self.x_max)
        _interval_gamma = (self.gamma_min, self.gamma_max)
        _LL = []

        def _f1(gamma, x0):
            self.gamma = gamma
            self.x0 = x0
            return self._LL_gamma(x, intensity)

        def _f2(x0):
            _g = self._opt_gamma(x0, _interval_gamma, x, intensity)
            self.gamma = _g
            return self._LL_x0(x, intensity)

        _x0 = np.linspace(_interval_x0[0], _interval_x0[-1], n_partition_x0)
        _init_g = np.array([_f1(_interval_gamma[0], x0) for x0 in _x0])
        _finish_g = np.array([_f1(_interval_gamma[-1], x0) for x0 in _x0])

        try:
            _interval_x0 = (_x0[np.sign(_init_g * _finish_g) == - 1][0], _x0[np.sign(_init_g * _finish_g) == - 1][-1])
        except IndexError:
            print("IndexError: _interval_x0 cannot be constructed.")
            print("Maximum likelihood estimation is failed. Reset parameters again.")
            self.x0 = np.random.uniform(self.x_min, self.x_max)
            self.gamma = np.random.uniform(self.gamma_min, self.gamma_max)

            return

        _x = np.linspace(_interval_x0[0], _interval_x0[-1], n_partition_x0)
        _y = np.array([_f2(x) for x in _x])
        #print(_x, _y)

        sign = np.array([np.sign(_y[i + 1] * _y[i]) for i in range(_x.size - 1)])
        sect = np.where(sign == -1)[0]
        _opt_x0 = [optimize.brentq(_f2, _x[i], _x[i + 1]) for i in sect[0:sect.size]]
        _opt_gamma = [self._opt_gamma(x0, _interval_gamma, x, intensity) for x0 in _opt_x0]
        for i in range(len(_opt_x0)):
            self.x0 = _opt_x0[i]
            self.gamma = _opt_gamma[i]
            _LL.append(self.log_likelihood(x, intensity))

        if len(_opt_x0) is 0:
            print("Maximum likelihood estimation is failed.")
            print("Reset parameters again.")
            self.x0 = np.random.uniform(self.x_min, self.x_max)
            self.gamma = np.random.uniform(self.gamma_min, self.gamma_max)
            return
        elif len(_opt_x0) is not 1:
            print("More than one local minima are found as follows.")
            print(_opt_x0)

        self.x0 = _opt_x0[np.argmax(_LL)]
        self.gamma = _opt_gamma[np.argmax(_LL)]

        return

    def minimize_bfgs(self, x, intensity):
        def func_ll(param):
            self.x0 = param[0]
            self.gamma = param[1]
            return - self.log_likelihood(x, intensity)

        self.x0 = np.sum(x * intensity) / np.sum(intensity)
        self.gamma = np.sqrt(2.0 * np.log(2.0) * np.sum(x ** 2 * intensity) / np.sum(intensity))

        init = [self.x0, self.gamma]
        info = optimize.minimize(func_ll, x0=init, bounds=[(-200, 200), (0.1, 2000)], method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.gamma = info['x'][1]
        return
