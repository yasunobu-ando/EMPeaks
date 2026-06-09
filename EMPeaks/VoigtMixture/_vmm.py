# License: BSD-3-clause
# Copyright © 2024 Institute of Science Tokyo
# Author: Yasunobu ANDO

from EMPeaks.VoigtMixture._voigt import Voigt
from EMPeaks.EMCore._em_core import EMCore
from EMPeaks.Background import UniformModel, SquareRootModel, LinearModel, TriangleModel, RampModel
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import copy
import time


class VoigtMixtureModel(EMCore):
    """
    class Mixture(K, Background)
    K: mixture component of Lorentzian
    Background: default 'uniform': including Uniform background model
                'none' : No background model is included.
    """
    def __init__(self, K=2, 
                 x_min=-300, x_max=300, 
                 sigma_min=0.1, sigma_max=70,
                 gamma_min=0.1, gamma_max=70,
                 background='none', k_ramp=0,
                 degree_spline=3, n_section=5):
        super().__init__(K=K, x_min=x_min, x_max=x_max, background=background, k_ramp=k_ramp, degree_spline=degree_spline, n_section=n_section)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max    
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.model[0:K] = [Voigt(x_min, x_max, 
                                 sigma_min, sigma_max, 
                                 gamma_min, gamma_max) for k in range(self.K)]

    def set_single_params(self, **param):
        # setting parameters for each single Gaussian model.
        param_set = {"x0", "sigma", "gamma"}
        single_params = self.extract_single_params(param_set, **param)
        self.model = [Voigt(self.x_min, self.x_max, 
                            self.sigma_min, self.sigma_max, 
                            self.gamma_min, self.gamma_max) for k in range(self.K)]
        [self.model[k].set_param(**single_params[k]) for k in range(self.K)]
        return

    def export_single_params(self, _tmp_param):
        """ parameters are sorted in the order of the first element in param_set."""
        param_set = {"x0", "sigma", "gamma"}
        _tmp = {}
        for param in list(param_set):
            _tmp[param] = [self.model[k].__dict__[param] for k in range(self.K)]

        _tmp_index = np.array(_tmp[list(param_set)[0]]).argsort()

        for param in list(param_set):
            _tmp_param[param] = list(np.array(_tmp[param])[_tmp_index])

        return _tmp_param, _tmp_index

    def add_hist_model(self, info, hist_model, trial):
        info.update({'x0_hist': np.array([hist_model[i]['x0'] for i in range(trial)]),
                     'sigma_hist': np.array([hist_model[i]['sigma'] for i in range(trial)]),
                     'gamma_hist': np.array([hist_model[i]['gamma'] for i in range(trial)]),
                     })
        return

    def print_param_summary(self, param):
        print('   x0:       ' + ('{:5.3f} eV        ' * len(param['x0'])).format(*param['x0']))
        print('   sigma:     ' + ('{:6.3e}          ' * len(param['sigma'])).format(*param['sigma']))
        print('   gamma:     ' + ('{:6.3e}          ' * len(param['gamma'])).format(*param['gamma']))
        print('   N_tot:   {:6.3e} '.format(self.N_tot))
        print('   N:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi'] * self.N_tot))
        print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
        return

    def adapted_em(self, x, intensity, max_iter, r_eps, stdout):
        from EMPeaks.EMCore._backend import get_backend
        if (get_backend() == "rust"
                and self.background in {"none", "uniform", "squareroot", "linear"}):
            return self._adapted_em_rust_voigt(x, intensity, max_iter, r_eps, stdout)
        return self._adapted_em_python(x, intensity, max_iter, r_eps, stdout)

    def _adapted_em_rust_voigt(self, x, intensity, max_iter, r_eps, stdout):
        import empeaks_rust_core

        start = time.time()
        print("<< Start fitting via Adapted EM Algorithm. [Rust backend / Voigt] >>")

        x0_arr    = np.array([self.model[k].x0    for k in range(self.K)], dtype=np.float64)
        sigma_arr = np.array([self.model[k].sigma  for k in range(self.K)], dtype=np.float64)
        gamma_arr = np.array([self.model[k].gamma  for k in range(self.K)], dtype=np.float64)
        pi        = self.pi.astype(np.float64).copy()
        da        = self.Dirichlet_alpha.astype(np.float64).copy()

        fix_x0    = [bool(self.model[k].fix_x0)    for k in range(self.K)]
        fix_sigma = [bool(self.model[k].fix_sigma)  for k in range(self.K)]
        fix_gamma = [bool(self.model[k].fix_gamma)  for k in range(self.K)]

        bg_type   = {"none": 0, "uniform": 1, "squareroot": 2, "linear": 3}.get(self.background, 0)
        s_tri_in  = float(self.model[-1].s_tri) if self.background == "linear" else 0.0

        total_iter, ll_hist, res_hist, s_tri_out = empeaks_rust_core.run_voigt_em_loop(
            x.astype(np.float64), intensity.astype(np.float64),
            x0_arr, sigma_arr, gamma_arr, pi, da,
            fix_x0, fix_sigma, fix_gamma,
            self.x_min, self.x_max,
            self.sigma_min, self.sigma_max,
            self.gamma_min, self.gamma_max,
            max_iter, r_eps, bg_type, s_tri_in,
        )

        for k in range(self.K):
            self.model[k].x0    = float(x0_arr[k])
            self.model[k].sigma = float(sigma_arr[k])
            self.model[k].gamma = float(gamma_arr[k])
        self.pi[:self.K_all] = pi
        if self.background == "linear":
            self.model[-1].s_tri = s_tri_out
            self.model[-1].s_uni = 1.0 - s_tri_out

        t_tot = time.time() - start
        ll = float(ll_hist[-1]) if len(ll_hist) > 0 else float('nan')
        residual = float(res_hist[-1]) if len(res_hist) > 0 else float('nan')

        converged = total_iter < max_iter
        if converged:
            print('Convergence is achieved at iter. {:3d}, elapsed time {:5.2f} s'
                  .format(total_iter, t_tot))
            print('   LogLikelihood:     {:12.8e}\n'
                  '        residual:      {:12.8e}'.format(ll, residual))
        else:
            print('>>> Convergence is not achieved within {:3d} iterations, '
                  'elapsed time: {:5.2f} s'.format(max_iter, t_tot))
            print('   LogLikelihood:      {:12.8e}\n'
                  '        residual:       {:12.8e}'.format(ll, residual))

        rmse = self.leastsq_for_normalization_factor(x, intensity, stdout)
        param = self.export_param()

        run_info = {
            'total_iter': total_iter,
            'total_time': t_tot,
            'time/iter': t_tot / max(total_iter, 1),
            'LL': ll,
            'LL_hist': list(ll_hist),
            'LL_residual': residual,
            'LL_residual_hist': list(res_hist),
            'RMSE': rmse,
        }
        if stdout:
            print('Estimated model parameters and scores are following:')
            self.print_param_summary(param)
            print('   LL:      {:12.8e}\n'
                  '   RMSE:     {:12.8e}\n'.format(run_info['LL'], run_info['RMSE']))
        return run_info


def main():
    vmm = VoigtMixtureModel()
    test = VoigtMixtureModel()
    test.model[0].x0 = -100
    test.model[1].x0 = +150

    x = np.arange(test.x_min, test.x_max)
    y = test.predict(x) * 20000

    vmm.sampling(x, y, trial=10, r_eps=1.0e-7)
    vmm.fit(x, y)

    plt.plot(x, y)
    plt.plot(x, vmm.predict(x))
    # plt.show()


if __name__ == '__main__':
    main()