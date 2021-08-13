from EMPeaks.DoniachSunjicMixture._doniachsunjic import DoniachSunjic
from EMPeaks.GaussianMixture._gmm2 import GaussianMixtureModel
# from ..Background import UniformModel, SquareRootModel, LinearModel, TriangleModel, RampModel
# from scipy import integrate
# from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
# import copy
# import time


class DoniachSunjicMixtureModel(GaussianMixtureModel):
    """
    class Mixture(K, Background)
    K: mixture component of Pseudo Voigt Function
    Background: default 'uniform': including Uniform background model
                'none' : No background model is included.

    """
    def __init__(self, K=2, x_min=-300, x_max=300, gamma_min=0.1, gamma_max=50, alpha_min=0.0, alpha_max=0.3,
                 background='none', k_ramp=0):
        super().__init__(K=K, x_min=x_min, x_max=x_max, background=background, k_ramp=k_ramp)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.model[0:K] = [DoniachSunjic(x_min, x_max, gamma_min, gamma_max, alpha_min, alpha_max)
                           for k in range(self.K)]

    def set_single_params(self, **param):
        # setting parameters for each single Gaussian model.
        param_set = {"x0", "gamma", "alpha"}
        single_params = self.extract_single_params(param_set, **param)
        self.model = [DoniachSunjic(self.x_min, self.x_max, self.gamma_min, self.gamma_max) for k in range(self.K)]
        [self.model[k].set_param(**single_params[k]) for k in range(self.K)]
        return

    def export_single_params(self, _tmp_param):
        """ parameters are sorted in the order of the first element in param_set."""
        param_set = {"x0", "gamma", "alpha"}
        _tmp = {}
        for param in list(param_set):
            _tmp[param] = [self.model[k].__dict__[param] for k in range(self.K)]

        _tmp_index = np.array(_tmp[list(param_set)[0]]).argsort()

        for param in list(param_set):
            _tmp_param[param] = list(np.array(_tmp[param])[_tmp_index])

        return _tmp_param, _tmp_index

    def add_hist_model(self, info, hist_model, trial):
        info.update({'x0_hist': np.array([hist_model[i]['x0'] for i in range(trial)]),
                     'gamma_hist': np.array([hist_model[i]['gamma'] for i in range(trial)]),
                     'alpha_hist': np.array([hist_model[i]['alpha'] for i in range(trial)])
                     })
        return

    def print_param_summary(self, param):
        print('   x0:        ' + ('{:5.3f} eV       ' * len(param['x0'])).format(*param['x0']))
        print('   gamma:     ' + ('{:6.3e}          ' * len(param['gamma'])).format(*param['gamma']))
        print('   alpha:     ' + ('{:6.3e}          ' * len(param['alpha'])).format(*param['alpha']))
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
#         self.peak_model = DoniachSunjicMixtureModel(K, x_min, x_max)
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


def main():
    lmm = DoniachSunjicMixtureModel()
    test = DoniachSunjicMixtureModel()

    test.model[0].x0 = -100
    test.model[1].x0 = +150

    x = np.arange(test.x_min, test.x_max)
    y = test.predict(x) * 10000

    lmm.sampling(x, y, trial=10, r_eps=1.e-6)
    lmm.fit(x, y, r_eps=1.e-8)

    plt.plot(x, y)
    plt.plot(x, lmm.predict(x) * lmm.N_tot)
    plt.show()


if __name__ == '__main__':
    main()