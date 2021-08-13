import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import gamma
from scipy.integrate import trapz


class DoniachSunjic:
    def __init__(self, x_min=-100, x_max=100, gamma_min=0.1, gamma_max=10, alpha_min=0.0, alpha_max=0.3):
        self.x_min = x_min
        self.x_max = x_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.x0, self.gamma, self.alpha = self.init_param()

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
        x0 = np.random.uniform(self.x_min, self.x_max)
        gamma = np.random.uniform(self.gamma_min, self.gamma_max)
        alpha = np.random.uniform(self.alpha_min, self.alpha_max)
        return x0, gamma, alpha

    def init_model(self):
        self.x0 = np.random.uniform(self.x_min, self.x_max)
        self.gamma = np.random.uniform(self.gamma_min, self.gamma_max)
        self.alpha = np.random.uniform(self.alpha_min, self.alpha_max)
        return

    def predict(self, x):
        pdf = gamma(1-self.alpha)/((x-self.x0)**2 + self.gamma**2)**((1-self.alpha)/2.0)
        pdf = pdf * np.cos((np.pi * self.alpha)/2 + (1 - self.alpha) * np.arctan((x-self.x0)/self.gamma))
        z = trapz(pdf, x)
        return pdf/z

    def log_likelihood(self, x, intensity):
        return np.sum(intensity * np.log(self.predict(x) + 1e-200))

    def plot(self, x, n_factor=1.0):
        plt.plot(x, n_factor * self.predict(x))
        return

    def maximum_likelihood_estimation(self, x, intensity):
        self.full_optimization(x, intensity)
        # self.conditional_max(x, intensity, 1.0e-5, 100)
        return

    def full_optimization(self, x, intensity):
        def func(param):
            self.x0 = param[0]
            self.gamma = param[1]
            self.alpha = param[2]
            return - self.log_likelihood(x, intensity)

#        self.x0 = np.sum(x * intensity)/np.sum(intensity)
#        self.gamma = np.sqrt(2.0 * np.log(2.0) * np.sum(x ** 2 * intensity) / np.sum(intensity))
#        self.alpha = 0.1

        init = [self.x0, self.gamma, self.alpha]
        info = optimize.minimize(func, x0=init, bounds=[(self.x_min, self.x_max),
                                                        (self.gamma_min, self.gamma_max),
                                                        (self.alpha_min, self.alpha_max)], method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.gamma = info['x'][1]
        self.alpha = info['x'][2]
        return

    def conditional_max(self, x, intensity, eps, max_iter):
        #print("### Start MLE of PseudoVoigt component via iterative optimization between (x0, gamma) and alpha.")
        def func_ll(param):
            self.x0 = param[0]
            self.gamma = param[1]
            return - self.log_likelihood(x, intensity)

        def func_alpha(alpha, param):
            self.x0 = param[0]
            self.gamma = param[1]
            self.alpha = alpha
            return - self.log_likelihood(x, intensity)

#        self.x0 = np.sum(x * intensity)/np.sum(intensity)
#        self.gamma = np.sqrt(2.0 * np.log(2.0) * np.sum(x ** 2 * intensity) / np.sum(intensity))
#        self.alpha = 0.1

        init = [self.x0, self.gamma]
#        ll = [self.log_likelihood(x, intensity)]
        alpha_grid = np.arange(self.alpha_min, self.alpha_max, 0.001)

        info = optimize.minimize(func_ll, x0=init,
                                 bounds=[(self.x_min, self.x_max), (self.gamma_min, self.gamma_max)],
                                 method='L-BFGS-B')
        self.x0 = info['x'][0]
        self.gamma = info['x'][1]
        index = np.nanargmin([func_alpha(alpha, info['x']) for alpha in alpha_grid])
        self.alpha = alpha_grid[index]

        return

    #        for i in range(max_iter):
#             info = optimize.minimize(func_ll, x0=init, bounds=[(-200, 200), (0.1, 2000)], method='L-BFGS-B')
#             init = info['x']
#             index = np.argmin([func_alpha(alpha, init) for alpha in alpha_grid])
#             self.alpha = alpha_grid[index]
#             #info = optimize.minimize(func_eta, x0=[self.eta], args=(info['x']), bounds=[(0, 1.0)], method='L-BFGS-B')
#             #self.alpha = info['x']
#             self.x0 = init[0]
#             self.gamma = init[1]
#             ll.append(self.log_likelihood(x, intensity))
#             res = (ll[-1] - ll[-2]) / np.abs(ll[-2])
#             #print("     iteration {:}, LL: {:}, residual: {:}".format(i, ll[-1], (ll[-1]-ll[-2]) / np.abs(ll[-2])))
#             if res < eps:
#                 return
#         print("MLE Does not converged.")

    def cdf(self, x):
        cdf = np.zeros(x.size)
        pdf = self.predict(x)
        dx = x[1] - x[0]
        cdf[0] = 0.0
        for i in range(x.size - 1):
            cdf[i+1] = cdf[i] + pdf[i] * dx
        return 1 - cdf
