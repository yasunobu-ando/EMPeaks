from scipy import optimize
from scipy.stats import norm
from scipy.stats import cauchy
import numpy as np


class PseudoVoigt:
    def __init__(self, x_min=-100, x_max=100, gamma_min=0.01, gamma_max=10):
        self.x_min = x_min
        self.x_max = x_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.x0 = np.random.uniform(x_min, x_max)
        self.gamma = np.random.uniform(gamma_min, gamma_max)
        self.eta = np.random.uniform()
        self.interval = (self.x_min, self.x_max)

    def set_param(self, **param):
        """
        Tp_min, Tp_maxの代償関係についてのチェックは未実装
        Ea_min, Ea_maxの代償関係についてのチェックは未実装
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
        self.eta = np.random.uniform()

    def predict(self, x):
        _sigma = self.gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return self.eta * norm.pdf(x, self.x0, _sigma) \
            + (1 - self.eta) / self._Z(self.x0, self.gamma) * cauchy.pdf(x, self.x0, self.gamma / 2.0)

    def _Z(self, x0, gamma):
        """ Normalization factor of Lorentzian component. """
        _Z = 1.0 / np.pi * np.arctan((self.x_max - x0) / (gamma / 2.0))
        _Z += - 1.0 / np.pi * np.arctan((self.x_min - x0) / (gamma / 2.0))
        return _Z

    def cdf(self, x):
        _sigma = self.gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        cdf = self.eta * norm.cdf(x, self.x0, _sigma)
        cdf += (1.0 - self.eta)/self._Z(self.x0, self.gamma) \
            * 1.0/np.pi*(np.arctan((x - self.x0)/(self.gamma/2.0))
                         - np.arctan((self.x_min - self.x0)/(self.gamma/2.0)))
        return cdf

    def _LL(self, x, intensity):
        t = np.log(self.predict(x) + 1.0e-20)
        self.LL = (t * intensity).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, intensity, max_iter=5000, eps=1e-8):
        # self.full_optimization(x, intensity, max_iter=max_iter, eps=eps)
        self.conditional_max(x, intensity, max_iter=max_iter, eps=eps)
        return

    def _LogZ_m(self, m, g):
        return 1.0 / self._Z(m, g) * (
                - cauchy.pdf(self.interval[-1], m, g / 2)
                + cauchy.pdf(self.interval[0], m, g / 2)
        )

    def _LogZ_g(self, m, g):
        _a = self.interval[0]
        _b = self.interval[-1]
        return 1.0 / (2 * np.pi * self._Z(m, g)) * (
                + (m - _b) / ((g / 2) ** 2 + (m - _b) ** 2)
                - (m - _a) / ((g / 2) ** 2 + (m - _a) ** 2)
        )

    def _qm(self, m, g, x, intensity):
        _N1 = (intensity * self.gamma1).sum()
        _N2 = (intensity * self.gamma2).sum()

        A = 8 * np.log(2) / (g ** 2) * np.sum(self.gamma1 * intensity * x) \
            + np.sum(self.gamma2 * intensity * (2 * x) / ((x - m) ** 2 + (g / 2) ** 2)) \
            - _N2 * self._LogZ_m(m, g)
        B = 8 * np.log(2) / (g ** 2) * _N1 \
            + np.sum(self.gamma2 * intensity * 2 / ((x - m) ** 2 + (g / 2) ** 2))

        return A / B

    def _qg(self, m, g, x, intensity):
        _N1 = (intensity * self.gamma1).sum()
        _N2 = (intensity * self.gamma2).sum()

        A = - g ** 4 * np.sum(self.gamma2 * intensity / 2.0 / ((x - m) ** 2 + (g / 2) ** 2))
        A += - _N2 * g ** 3 * self._LogZ_g(m, g)
        A += (_N2 - _N1) * g ** 2
        A += 8 * np.log(2) * np.sum(self.gamma1 * intensity * (x - m) ** 2)

        return A

    def _opt_g(self, m, _interval_g, x, intensity):
        def _fg(g, m):
            return self._qg(m, g, x, intensity)

        return optimize.brentq(_fg, _interval_g[0], _interval_g[1], args=(m))

    def _e_step(self, x):
        _sigma = self.gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        _gamma1 = self.eta * norm.pdf(x, self.x0, _sigma) / self.predict(x)
        _gamma2 = (1.0 - self.eta) * cauchy.pdf(x, self.x0, self.gamma / 2.0) / self.predict(x)
        self.gamma1 = _gamma1
        self.gamma2 = _gamma2
        return

    def _m_step(self, x, intensity):
        N = 100
        _interval_m = (self.x_min, self.x_max)
        _interval_g = (0.1, self.x_max - self.x_min)
        self.eta = np.sum(self.gamma1 * intensity) / np.sum(intensity)

        def _fm(m):
            _g = self._opt_g(m, _interval_g, x, intensity)
            return self._qm(m, _g, x, intensity) - m

        _m = np.linspace(_interval_m[0], _interval_m[-1], N)
        _init_g = np.array([self._qg(m, _interval_g[0], x, intensity) for m in _m])
        _finish_g = np.array([self._qg(m, _interval_g[-1], x, intensity) for m in _m])
        _interval_m = (_m[np.sign(_init_g * _finish_g) == - 1][0],
                       _m[np.sign(_init_g * _finish_g) == - 1][-1])
        _m = np.linspace(_interval_m[0], _interval_m[-1], N)
        _y = np.array([_fm(m) for m in _m])

        sign = np.array([np.sign(_y[i + 1] * _y[i]) for i in range(_m.size - 1)])
        sect = np.where(sign == -1)[0]
        root = np.array([optimize.brentq(_fm, _m[i], _m[i + 1]) for i in sect[0:sect.size]])
        root2 = np.array([self._opt_g(m, _interval_g, x, intensity) for m in root])

        list_ll = np.array([])
        for i in range(len(root)):
            _sigma = root2[i] / (2 * np.sqrt(2 * np.log(2)))
            t = self.eta * norm.pdf(x, root[i], _sigma) + (1 - self.eta) \
                * cauchy.pdf(x, root[i], root2[i] / 2)
            list_ll = np.append(list_ll, (np.log(t) * intensity).sum())

        self.x0 = root[np.argmax(list_ll)]
        self.gamma = root2[np.argmax(list_ll)]
        return

    def _cm_step_x0_gamma(self, x, intensity):
        N = 10
        _interval_m = (self.x_min, self.x_max)
        _interval_g = (0.1, self.x_max - self.x_min)

        def _fm(m):
            _g = self._opt_g(m, _interval_g, x, intensity)
            return self._qm(m, _g, x, intensity) - m

        _m = np.linspace(_interval_m[0], _interval_m[-1], N)
        _init_g = np.array([self._qg(m, _interval_g[0], x, intensity) for m in _m])
        _finish_g = np.array([self._qg(m, _interval_g[-1], x, intensity) for m in _m])
        _interval_m = (_m[np.sign(_init_g * _finish_g) == - 1][0],
                       _m[np.sign(_init_g * _finish_g) == - 1][-1])
        _m = np.linspace(_interval_m[0], _interval_m[-1], N)
        _y = np.array([_fm(m) for m in _m])

        sign = np.array([np.sign(_y[i + 1] * _y[i]) for i in range(_m.size - 1)])
        sect = np.where(sign == -1)[0]
        root = np.array([optimize.brentq(_fm, _m[i], _m[i + 1]) for i in sect[0:sect.size]])
        root2 = np.array([self._opt_g(m, _interval_g, x, intensity) for m in root])

        list_ll = np.array([])
        for i in range(len(root)):
            _sigma = root2[i] / (2 * np.sqrt(2 * np.log(2)))
            t = self.eta * norm.pdf(x, root[i], _sigma) + (1 - self.eta) \
                * cauchy.pdf(x, root[i], root2[i] / 2)
            list_ll = np.append(list_ll, (np.log(t) * intensity).sum())

        self.x0 = root[np.argmax(list_ll)]
        self.gamma = root2[np.argmax(list_ll)]
        return

    def _cm_step_eta(self, x, intensity):
        eta = np.arange(0, 1, 0.01)
        ll = []
        for i in range(eta.size):
            self.eta = eta[i]
            ll.append(self._LL(x, intensity))
        self.eta = eta[np.argmax(np.array(ll))]
        return

    def conditional_max(self, x, intensity, eps, max_iter):
#        print("### Start MLE of Pseudo-Voigt component via adapted ECM algorithm.")
#        eps_x0_gamma = eps
#        self.x0 = np.sum(x * intensity)/np.sum(intensity)
#        self.gamma = np.sum(x**2 * intensity)/np.sum(intensity)
#        self.eta = 1.0
#        ll = [self._LL(x, intensity)]
        self._e_step(x)
        self._cm_step_x0_gamma(x, intensity)
        self._cm_step_eta(x, intensity)
#        for i in range(max_iter):
#            for j in range(max_iter):
#                self._e_step(x)
#                self._cm_step_x0_gamma(x, intensity)
#                ll.append(self._LL(x, intensity))
#                if (ll[-1] - ll[-2]) / np.abs(ll[-2]) < eps_x0_gamma:
#                    break
#
#            self._cm_step_eta(x, intensity)
#            ll.append(self._LL(x, intensity))
#            print("     iteration {:}, LL: {:}, residual: {:}".format(i, ll[-1], (ll[-1]-ll[-2]) / np.abs(ll[-2])))
#            if (ll[-1] - ll[-2])/np.abs(ll[-2]) < eps:
#                return
#
#        print("Does not converged.")
        return

    def full_optimization(self, x, intensity, eps, max_iter):
        #print("### Start MLE of PseudoVoigt component via iterative optimization between (x0, gamma) and eta.")

        def func_ll(param):
            self.x0 = param[0]
            self.gamma = param[1]
            return - self._LL(x, intensity)

        def func_eta(eta, param):
            self.x0 = param[0]
            self.gamma = param[1]
            self.eta = eta
            return - self._LL(x, intensity)

        #self.x0 = np.sum(x * intensity)/np.sum(intensity)
        #self.gamma = np.sqrt(2.0 * np.log(2.0) * np.sum(x ** 2 * intensity) / np.sum(intensity))
        #self.eta = 0.5

        init = [self.x0, self.gamma]
        ll = [self._LL(x, intensity)]
        eta_grid = np.arange(0, 1.0, 0.01)
        for i in range(max_iter):
            info = optimize.minimize(func_ll, x0=init,
                                     bounds=[(self.x_min, self.x_max), (self.gamma_min, self.gamma_max)],
                                     method='L-BFGS-B')
            init = info['x']
            index = np.nanargmin([func_eta(eta, init) for eta in eta_grid])
            self.eta = eta_grid[index]
            #info = optimize.minimize(func_eta, x0=[self.eta], args=(info['x']), bounds=[(0, 1.0)], method='L-BFGS-B')
            #self.eta = info['x']
            self.x0 = init[0]
            self.gamma = init[1]
            ll.append(self._LL(x, intensity))
            res = (ll[-1] - ll[-2]) / np.abs(ll[-2])
            #print("     iteration {:}, LL: {:}, residual: {:}".format(i, ll[-1], (ll[-1]-ll[-2]) / np.abs(ll[-2])))
            if res < eps:
                return
        print("MLE Does not converged.")
        return
