from EMPeaks.LorentzianMixture._lorentz import Lorentzian
from EMPeaks.GaussianMixture._gmm2 import GaussianMixtureModel
from EMPeaks.Background import UniformModel, SquareRootModel, LinearModel, TriangleModel, RampModel
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import copy
import time


class LorentzianMixtureModel(GaussianMixtureModel):
    """
    class Mixture(K, Background)
    K: mixture component of Lorentzian
    """
    def __init__(self, K=2, x_min=-300, x_max=300, gamma_min=0.1, gamma_max=500,
                 background='none', k_ramp=0):
        super().__init__(K=K, x_min=x_min, x_max=x_max, background=background, k_ramp=k_ramp)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.model[0:K] = [Lorentzian(x_min, x_max, gamma_min, gamma_max) for k in range(self.K)]


    def set_single_params(self, **param):
        # setting parameters for each single Gaussian model.
        param_set = {"x0", "gamma"}
        single_params = self.extract_single_params(param_set, **param)
        self.model = [Lorentzian(self.x_min, self.x_max, self.gamma_min, self.gamma_max) for k in range(self.K)]
        [self.model[k].set_param(**single_params[k]) for k in range(self.K)]
        return

    def export_single_params(self, _tmp_param):
        """ parameters are sorted in the order of the first element in param_set."""
        param_set = {"x0", "gamma"}
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
                     })
        return

    def print_param_summary(self, param):
        print('   x0:       ' + ('{:5.3f} eV        ' * len(param['x0'])).format(*param['x0']))
        print('   gamma:     ' + ('{:6.3e}          ' * len(param['gamma'])).format(*param['gamma']))
        print('   N_tot:   {:6.3e} '.format(self.N_tot))
        print('   N:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi'] * self.N_tot))
        print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
        return


def main():
    lmm = LorentzianMixtureModel()
    test = LorentzianMixtureModel()

    x = np.arange(test.x_min, test.x_max)
    y = test.predict(x) * 10000

    lmm.sampling(x, y, trial=10)
    lmm.fit(x, y)

    plt.plot(x, y)
    plt.plot(x, lmm.predict(x) * lmm.N_tot)
    plt.show()


if __name__ == '__main__':
    main()
