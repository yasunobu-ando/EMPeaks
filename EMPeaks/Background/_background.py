# License: BSD-3-clause
# Copyright © 2020-2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu ANDO

import numpy as np
import sys
from scipy import stats, optimize
from scipy.integrate import quad
from scipy.interpolate import BSpline


class UniformModel:
    """
    UniformModel(self, x, w):
    x: sampling point or data bins
    w: weight or intensity of the data
    """
    def __init__(self, x0, x1):
        self.x0 = x0
        self.width = x1 - x0

    def predict(self, x):
        return stats.uniform.pdf(x, loc=self.x0, scale=self.width)

    def log_likelihood(self, x, weight):
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        return


class LinearModel:
    """
    LinearModel(self, x, w):
    リニアモデルには、一様バックグラウンド成分が含まれています。
    オプションで、勾配の符号を決定する必要があります。
    x: sampling point or data bins
    w: weight or intensity of the data
    """
    def __init__(self, x0, x1, sign='positive', s_tri=np.random.rand()):
        self.x0 = x0
        self.x1 = x1
        self.s_tri = s_tri
        self.s_uni = 1.0 - self.s_tri
        if sign == 'positive':
            self.sign = +1
        elif sign == 'negative':
            self.sign = -1
        self.LL = 0.0

    def predict(self, x):
        s1 = self.s_tri
        s2 = 1 - s1
        a = self.sign * (2.0 * s1) / (self.x1 - self.x0) ** 2
        b = s2 / (self.x1 - self.x0)
        # when sign is +1, x0 is selected. Otherwise, sing = -1, x1 is selected.
        c = self.x0 ** ((1+self.sign)/2) * self.x1 ** ((1-self.sign)/2)
        return a*(x-c) + b

    def log_likelihood(self, x, weight):
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        def score(s_tri, x, weight):
            eps = 1e-5
            s_uni = 1 - s_tri
            a = (2.0 * s_tri) / ((self.x1 - self.x0) ** 2) * self.sign
            b = s_uni / (self.x1 - self.x0)
            c = self.x0 ** ((1 + self.sign) / 2) * self.x1 ** ((1 - self.sign) / 2)

            a_prime = 2.0 / ((self.x1 - self.x0) ** 2) * self.sign
            b_prime = -1 / (self.x1 - self.x0)
            score = np.sum(weight * (a_prime * (x - c) + b_prime) / (a * (x - c) + b + eps))
            return score

        try:
            self.s_tri = optimize.brentq(score, 0,   1.0 - 1.0e-5, args=(x, weight))
            self.s_uni = 1.0 - self.s_tri
            return

        except ValueError:
            print("ValueError at MLE of Linear background model")
            self.s_tri = (score(0, x, weight) >= score(1, x, weight))
            self.s_uni = (score(0, x, weight) < score(1, x, weight))
            return


class RampModel:
    def __init__(self, x0, x1, x2):
        if not x0 < x1 < x2:
            print("Error !!! x1 must between x0 an x2.")
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.LL = 0.0

    def predict(self, x):
        y1 = 2.0 / (2.0 * self.x2 - self.x1 - self.x0)
        a = y1/(self.x1 - self.x0)
        return np.minimum( a * (x - self.x1) + y1, y1 * np.ones(x.size)) * ( x <= self.x2) * (self.x0 < x)

    def log_likelihood(self, x, weight):
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        return


class TriangleModel:
    """
    TriangleModel:
    """
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1
        self.LL = 0.0

    def predict(self, x):
        y1 = 2.0 / (self.x1 - self.x0)
        a = y1 / (self.x1 - self.x0)
        return (a * (x - self.x1) + y1) * (self.x0 < x) * (x <= self.x1)

    def log_likelihood(self, x, weight):
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        return


class SquareRootModel:
    """
    SquareRootModel(self, x0, x1):
    x: sampling point or data bins
    w: weight or intensity of the data
    """
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1

    def predict(self, x):
        return np.sqrt(x) /(2.0/3.0 * (self.x1**(1.5) - self.x0**(1.5)))

    def log_likelihood(self, x, weight):
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        return


class LinearModelEM:
    """
    UniformModel(self, x, w):
    x: sampling point or data bins
    w: weight or intensity of the data
    """
    def __init__(self, x0, x1):
        self.x0 = x0
        self.width = x1 - x0
        self.pi_uni = stats.uniform.rvs(0, 1, 1)
        self.pi_tri = 1.0 - self.pi_uni
        self.LL = 0.0

    def predict(self, x):
        return self.pi_tri * 2.0 * stats.triang.pdf(x, 0.5, loc = self.x0, scale=2*self.width) \
                + self.pi_uni * stats.uniform.pdf(x, loc=self.x0, scale=self.width)

    def log_likelihood(self, x, weight):
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        eps = 1.0e-3
        max_iter = 10000
        ll = [self._LL(x, weight)]
        for i in range(max_iter):
            self._e_step(x)
            self._m_step(x, weight)
            ll.append(self._LL(x, weight))
            #print(ll[-1], ll[-1] - ll[-2])
            if np.abs(ll[-1] - ll[-2]) < eps:
                #print(self.pi_uni, self.pi_tri)
                return
        print("Warning!!! LinearBackground does not converge.")
        print(self.pi_uni, self.pi_tri)
        return

    def _e_step(self, x):
        self._gamma_tri = self.pi_tri * 2.0 * stats.triang.pdf(x, 0.5, loc=self.x0,
                                                               scale=2 * self.width) / self.predict(x)
        self._gamma_uni = self.pi_uni * stats.uniform.pdf(x, loc=self.x0,
                                                          scale=self.width) / self.predict(x)

    def _m_step(self, x, weight):
        self.pi_tri = np.sum(self._gamma_tri * weight) / np.sum(weight)
        self.pi_uni = np.sum(self._gamma_uni * weight) / np.sum(weight)


class TanhModel:
    def __init__(self, x0, a, x_min=-10.0, x_max=10.0):
        self.x_min = x_min
        self.x_max = x_max
        self.x0 = x0
        self.a = a
        self.eps = 1.0e-10

    def normalization_factor(self):
        return 1 / self.a * (+ np.log(np.cosh(self.a * (self.x_max - self.x0)))
                             - np.log(np.cosh(self.a * (self.x_min - self.x0)))) + (self.x_max - self.x_min)

    def predict(self, x):
        Z = self.normalization_factor()
        y = (np.tanh(self.a * (x - self.x0)) + 1.0)
        return y / Z

    def log_likelihood(self, x, intensity):
        return np.sum(intensity * np.log(self.predict(x) + self.eps))

    def maximum_likelihood_estimation(self, x, intensity):
        def func(param):
            self.x0 = param[0]
            self.a = param[1]
            return - self.log_likelihood(x, intensity)

        init = [self.x0, self.a]
        info = optimize.minimize(func, x0=init, bounds=[(self.x_min, self.x_max),
                                                        (0, np.inf)], method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.a = info['x'][1]
        return


class TanhLinearModel:
    def __init__(self, x0, a, x_min=-10.0, x_max=10.0):
        self.x_min = x_min
        self.x_max = x_max
        self.x0 = x0
        self.a = a
        self.eps = 1e-10

    def normalization_factor(self):
        def func(x):
            return -(x - self.x_max) * (np.tanh(self.a * (x - self.x0)) + 1.0)

        return quad(func, self.x_min, self.x_max)[0]

    def predict(self, x):
        Z = self.normalization_factor()
        y = -(x - self.x_max) * (np.tanh(self.a * (x - self.x0)) + 1.0)
        return y / Z

    def log_likelihood(self, x, intensity):
        return np.sum(intensity * np.log(self.predict(x) + self.eps))

    def maximum_likelihood_estimation(self, x, intensity):
        def func(param):
            self.x0 = param[0]
            self.a = param[1]
            return - self.log_likelihood(x, intensity)

        init = [self.x0, self.a]
        info = optimize.minimize(func, x0=init, bounds=[(self.x_min, self.x_max),
                                                        (0, np.inf)], method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.a = info['x'][1]
        return


class TanhQuadModel:
    def __init__(self, x0, a, x_min=-10.0, x_max=10.0):
        self.x_min = x_min
        self.x_max = x_max
        self.x0 = x0
        self.a = a
        self.eps = 1e-10

    def normalization_factor(self):
        def func(x):
            return -(x - self.x_max) * (np.tanh(self.a * (x - self.x0)) + 1.0)

        return quad(func, self.x_min, self.x_max)[0]

    def predict(self, x):
        Z = self.normalization_factor()
        y = (x - self.x_max) ** 2 * (np.tanh(self.a * (x - self.x0)) + 1.0)
        return y / Z

    def log_likelihood(self, x, intensity):
        return np.sum(intensity * np.log(self.predict(x) + self.eps))

    def maximum_likelihood_estimation(self, x, intensity):
        def func(param):
            self.x0 = param[0]
            self.a = param[1]
            return - self.log_likelihood(x, intensity)

        init = [self.x0, self.a]
        info = optimize.minimize(func, x0=init, bounds=[(self.x_min, self.x_max),
                                                        (0, np.inf)], method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.a = info['x'][1]

        return


class TanhCubicModel:
    def __init__(self, x0, a, x_min=-10.0, x_max=10.0):
        self.x_min = x_min
        self.x_max = x_max
        self.x0 = x0
        self.a = a
        self.eps = 1e-10

    def normalization_factor(self):
        def func(x):
            return -(x - self.x_max) * (np.tanh(self.a * (x - self.x0)) + 1.0)

        return quad(func, self.x_min, self.x_max)[0]

    def predict(self, x):
        Z = self.normalization_factor()
        y = -(x - self.x_max) ** 3 * (np.tanh(self.a * (x - self.x0)) + 1.0)
        return y / Z

    def log_likelihood(self, x, intensity):
        return np.sum(intensity * np.log(self.predict(x) + self.eps))

    def maximum_likelihood_estimation(self, x, intensity):
        def func(param):
            self.x0 = param[0]
            self.a = param[1]
            return - self.log_likelihood(x, intensity)

        init = [self.x0, self.a]
        info = optimize.minimize(func, x0=init, bounds=[(self.x_min, self.x_max),
                                                        (0, np.inf)], method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.a = info['x'][1]

        return


class SplineBasisModel:
    def __init__(self, x0, x1, degree_spline, n_section, basis_index):
        if basis_index >= (n_section + degree_spline):
            print("ERROR!!!!")
            sys.exit(0)
        if basis_index < 0:
            print("ERROR!!!! Basis_index should be positive integer or zero.")
            sys.exit(0)

        self.x0 = x0
        self.x1 = x1
        self.x_scale = (self.x1 - self.x0) / (n_section)
        print(self.x0)

        self.degree_spline = degree_spline
        self.n_section = n_section
        self.basis_index = basis_index
        self.knot_vector = np.array([min(n_section,
                                         max(0, i - degree_spline + self.basis_index))
                                     for i in range(degree_spline + 2)]) * self.x_scale + self.x0
        print(self.knot_vector)
        self.LL = 0.0

        print("*** SplineBasisModel for background ***")
        print("range of basis index is [0, {:})".format(n_section + degree_spline))

    def predict(self, x):
        type_index = len(set(self.knot_vector))
        z = (type_index - 1) / (self.degree_spline + 1) * self.x_scale
        b = BSpline.basis_element(self.knot_vector, extrapolate=False)
        return np.nan_to_num(b(x)) / z

    def _LL(self, x, weight):
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        return