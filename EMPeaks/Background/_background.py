import numpy as np
from scipy import stats, optimize #, integrate


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

    def _LL(self, x, weight):
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

    def _LL(self, x, weight):
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

    def _LL(self, x, weight):
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


class LinearModel:
    """
    LinearModel(self, x, w):
    x: sampling point or data bins
    w: weight or intensity of the data
    """
    def __init__(self, x0, x1, s_tri=np.random.rand()):
        self.x0 = x0
        self.x1 = x1
        self.s_tri = s_tri
        self.s_uni = 1.0 - self.s_tri
        self.LL = 0.0

    def predict(self, x):
        s1 = self.s_tri
        s2 = 1 - s1
        a = (2.0 * s1) / (self.x1 - self.x0) ** 2
        b = s2 / (self.x1 - self.x0) - a * self.x0
        return a*x + b

    def _LL(self, x, weight):
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        def score(s_tri, x, weight):
            eps = 1e-5
            s_uni = 1 - s_tri
            a = (2.0 * s_tri) / (self.x1 - self.x0) ** 2
            b = s_uni / (self.x1 - self.x0) - a * self.x0

            a_prime = 2.0 / (self.x1 - self.x0) ** 2
            b_prime = -1 / (self.x1 - self.x0) - a_prime * self.x0

            return np.sum(weight * (a_prime * x + b_prime) / (a * x + b + eps))

        try:
            self.s_tri = optimize.brentq(score, 0, 1.0 - 1.0e-5, args=(x, weight))
            self.s_uni = 1.0 - self.s_tri
            return

        except ValueError:
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

    def _LL(self, x, weight):
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        return


class TriangleModel:
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1
        self.LL = 0.0

    def predict(self, x):
        y1 = 2.0 / (self.x1 - self.x0)
        a = y1 / (self.x1 - self.x0)
        return (a * (x - self.x1) + y1) * (self.x0 < x) * (x <= self.x1)

    def _LL(self, x, weight):
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        return
