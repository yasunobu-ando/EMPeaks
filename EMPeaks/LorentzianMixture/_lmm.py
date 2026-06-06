# License: BSD-3-clause
# Copyright © 2020-2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu ANDO

from EMPeaks.EMCore._em_core import EMCore
from EMPeaks.LorentzianMixture._lorentz import Lorentzian
from ..Background import UniformModel, SquareRootModel, LinearModel, TriangleModel, RampModel
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import copy
import time


class LorentzianMixtureModel(EMCore):
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

    def adapted_em(self, x, intensity, max_iter, r_eps, stdout):
        from EMPeaks.EMCore._backend import get_backend
        if get_backend() == "rust" and self.background == "none":
            return self._adapted_em_rust_lo(x, intensity, max_iter, r_eps, stdout)
        return self._adapted_em_python(x, intensity, max_iter, r_eps, stdout)

    def _adapted_em_rust_lo(self, x, intensity, max_iter, r_eps, stdout):
        import empeaks_rust_core
        import time

        start = time.time()
        print("<< Start fitting via Adapted EM Algorithm. [Rust backend / Lorentzian] >>")

        ll_0 = self.log_likelihood(x, intensity)
        ll_hist = [ll_0]
        res_hist = [0.0]
        total_iter = max_iter
        t_tot = 0.0
        ll = ll_0
        residual = 0.0

        x_f64 = x.astype(np.float64)

        for it in range(max_iter):
            self.e_step(x)

            N_k = np.array([np.sum(intensity * self._gamma[k]) for k in range(self.K_all)])
            self.pi = (N_k + self.Dirichlet_alpha - 1) / np.sum(N_k + self.Dirichlet_alpha - 1)
            self.pi[self.pi < 0] = 0.0
            self.pi = self.pi / np.sum(self.pi)

            for k in range(self.K):
                w = (intensity * self._gamma[k]).astype(np.float64)
                if np.sum(w) == 0:
                    continue
                x0_new, gamma_new = empeaks_rust_core.lorentzian_mle(
                    x_f64, w,
                    self.model[k].x0, self.model[k].gamma,
                    self.model[k].x_min, self.model[k].x_max,
                )
                self.model[k].x0 = x0_new
                self.model[k].gamma = gamma_new

            ll = self.log_likelihood(x, intensity)
            residual = (ll - ll_0) / np.abs(ll_0)
            ll_hist.append(ll)
            res_hist.append(residual)

            if stdout and it % 10 == 0:
                t2 = time.time()
                print("> iteration #{:3d}, LL={:10.8e}, residual={:4.3e}, elapsed time: {:5.2f} s"
                      .format(it, ll, residual, t2 - start))

            if residual < 0.0:
                print("Warning!!!! residual is negative!!!  Parameters are initialized again.")
                self.init_param_uniform()
            elif residual < r_eps:
                t_tot = time.time() - start
                total_iter = it + 1
                print('Convergence is achieved at iter. {:3d}, elapsed time {:5.2f} s'
                      .format(it, t_tot))
                print('   LogLikelihood:     {:12.8e}\n        residual:      {:12.8e}'
                      .format(ll, residual))
                break

            ll_0 = ll
        else:
            t_tot = time.time() - start
            print('>>> Convergence is not achieved within {:3d} iterations, elapsed time: {:5.2f} s'
                  .format(max_iter, t_tot))
            print('   LogLikelihood:      {:12.8e}\n        residual:       {:12.8e}'
                  .format(ll, residual))

        rmse = self.leastsq_for_normalization_factor(x, intensity, stdout)
        param = self.export_param()

        run_info = {
            'total_iter': total_iter,
            'total_time': t_tot,
            'time/iter': t_tot / max(total_iter, 1),
            'LL': ll,
            'LL_hist': ll_hist,
            'LL_residual': residual,
            'LL_residual_hist': res_hist,
            'RMSE': rmse,
        }
        if stdout:
            print('Estimated model parameters and scores are following:')
            self.print_param_summary(param)
            print('   LL:      {:12.8e}\n'
                  '   RMSE:     {:12.8e}\n'.format(run_info['LL'], run_info['RMSE']))
        return run_info

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
