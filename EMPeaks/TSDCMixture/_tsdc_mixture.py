# License: BSD-3-clause
# Copyright © 2020-2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu ANDO

from ._tsdc import TSDC
from EMPeaks.EMCore._em_core import EMCore
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import time
import copy


class TSDCMixtureModel(EMCore):
    """
    class TSDCMixture(K, Background)
    K: mixture component of TSDC
    Background: default 'uniform': including Uniform background model
                'none' : No background model is included.

    """
    def __init__(self, K=2, beta=0.0833, T_min=300, T_max=900, Ea_min=0.05, Ea_max=3.0,
                 background='none', k_ramp=0):
        super().__init__(K=K, x_min=0, x_max=10, background=background, k_ramp=k_ramp)
        self.beta = beta
        self.T_min = T_min
        self.T_max = T_max
        self.Ea_min = Ea_min
        self.Ea_max = Ea_max
        self.dT = 1.0
        self.model = [TSDC(beta, T_min, T_max, Ea_min, Ea_max) for k in range(self.K)]
        self.set_param_background()
        self.N_tot = 1.0
        self.N = self.pi * self.N_tot

    def set_single_params(self, **param):
        # setting parameters for each single TSDC model.
        param_set = {'Tp', 'Ea', 'tau0', 'is_not_converged'}
        single_params = self.extract_single_params(param_set, **param)
        self.model = [TSDC(self.beta, self.T_min, self.T_max, self.Ea_min, self.Ea_max) for k in range(self.K)]
        [self.model[k].set_param(**single_params[k]) for k in range(self.K)]
        return

    def adapted_em(self, x, intensity, max_iter, r_eps, stdout):
        from EMPeaks.EMCore._backend import get_backend
        if get_backend() == "rust" and self.background in {"none", "uniform", "squareroot", "linear"}:
            return self._adapted_em_rust_tsdc(x, intensity, max_iter, r_eps, stdout)
        return self._adapted_em_python(x, intensity, max_iter, r_eps, stdout)

    def _adapted_em_rust_tsdc(self, x, intensity, max_iter, r_eps, stdout):
        import empeaks_rust_core
        import time

        start = time.time()
        print("<< Start fitting via Adapted EM Algorithm. [Rust backend / TSDC Phase 3] >>")

        t_f64 = np.ascontiguousarray(x, dtype=np.float64)
        int_f64 = np.ascontiguousarray(intensity, dtype=np.float64)

        ea_arr = np.array([float(np.squeeze(self.model[k].Ea)) for k in range(self.K)], dtype=np.float64)
        tau0_arr = np.array([float(np.squeeze(self.model[k].tau0)) for k in range(self.K)], dtype=np.float64)
        tp_arr = np.array([float(np.squeeze(self.model[k].Tp)) for k in range(self.K)], dtype=np.float64)
        pi_arr = np.array(self.pi[:self.K_all], dtype=np.float64)
        da_arr = np.ascontiguousarray(self.Dirichlet_alpha[:self.K_all], dtype=np.float64)

        bg_type = {"none": 0, "uniform": 1, "squareroot": 2, "linear": 3}.get(self.background, 0)
        s_tri_in = float(self.model[-1].s_tri) if self.background == "linear" else 0.0

        total_iter, ll_hist_np, res_hist_np, s_tri_out = empeaks_rust_core.run_tsdc_em_loop(
            t_f64, int_f64,
            ea_arr, tau0_arr, tp_arr, pi_arr,
            da_arr,
            self.beta,
            self.Ea_min, self.Ea_max,
            self.T_min, self.T_max,
            max_iter, r_eps, bg_type, s_tri_in,
        )

        for k in range(self.K):
            self.model[k].Ea = float(ea_arr[k])
            self.model[k].tau0 = float(tau0_arr[k])
            self.model[k].Tp = float(tp_arr[k])
        self.pi[:self.K_all] = pi_arr
        if self.background == "linear":
            self.model[-1].s_tri = s_tri_out
            self.model[-1].s_uni = 1.0 - s_tri_out

        ll_hist = list(ll_hist_np)
        res_hist = list(res_hist_np)
        ll = ll_hist[-1]
        residual = res_hist[-1] if len(res_hist) > 1 else 0.0
        t_tot = time.time() - start

        if total_iter < max_iter:
            print('Convergence is achieved at iter. {:3d}, elapsed time {:5.2f} s'
                  .format(total_iter, t_tot))
            print('   LogLikelihood:     {:12.8e}\n        residual:      {:12.8e}'
                  .format(ll, residual))
        else:
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

    def set_param(self, **param):
        """
        """
        if not param:
            return self

        param_keys = set(param.keys())
        if param_keys >= {'K'}:
            self.K = param['K']

        self.set_single_params(**param)
        self.set_param_background(**param)

        param_keys = set(param.keys())
        if param_keys >= {'N'}:
            self.N = param['N']
            self.N_tot = np.sum(param['N'])
            self.pi = param['N'] / self.N_tot
            return

        # setting mixture ratio pi, N, and N_tot
        if param_keys >= {'N'}:
            self.N = param[('N')]
            self.N_tot = np.sum(param['N'])
            self.pi = param['N']/self.N_tot
            return

        if param_keys >= {'pi'}:
            if type(param['pi']) != list:
                print("Parameter \"pi\" is not list type. Then pi is uniformly set as default.")
                self.pi = np.ones(self.K_all) / self.K_all
                return
            if len(param["pi"]) == self.K_all:
                self.pi = np.array(param['pi']) / sum(param['pi'])
                if sum(param["pi"]) != 1.0:
                    print("Sum of \"pi\" must be unity. Then values are normalized as ", self.pi)
                return
            elif len(param["pi"]) == self.K:
                print("Parameter \"pi\" does not have background ratio. pi for background is set to be 0.1.")
                self.pi = np.array(param['pi']) / sum(param['pi']) * 0.9
                self.pi = np.append(self.pi, 0.1)
                if sum(param["pi"]) != 1.0:
                    print("Sum of \"pi\" must be unity. Then values are normalized as ", self.pi)
                return
            else:
                print("Parameter \"pi\" does not have relevant length. Then pi is uniformly set as default.")
                self.pi = np.ones(self.K_all) / self.K_all
                return

    def export_param(self):
        _tmp_param = self.__dict__
        _tmp_param, _tmp_index = self.export_single_params(_tmp_param)
        if self.background == 'linear':
            s_in_linear = [self.model[-1].s_uni, self.model[-1].s_tri]
        if self.background == 'linear':
            self.model[-1].s_uni = s_in_linear[0]
            self.model[-1].s_tri = s_in_linear[1]
        _tmp_param['pi'][0:self.K] = list(np.array(self.pi)[0:self.K][_tmp_index])
        _tmp_param['N'] = list(np.array(_tmp_param['pi']) * self.N_tot)

        self.set_param(**_tmp_param)

        return _tmp_param

    def export_single_params(self, _tmp_param):
        """ parameters are sorted in ascending order of Ea."""
        param_set = {"Ea", "tau0", "Tp", "is_not_converged"}
        _tmp = {}
        for param in list(param_set):
            _tmp[param] = [self.model[k].__dict__[param] for k in range(self.K)]

        # Always sort by scalar "Ea" to avoid non-deterministic set ordering
        # and 2-D argsort when is_not_converged (a 3-element array) comes first.
        _tmp_index = np.array(_tmp["Ea"]).argsort()

        for param in list(param_set):
            _tmp_param[param] = list(np.array(_tmp[param])[_tmp_index])

        return _tmp_param, _tmp_index

    def init_param_empirical(self, T, intensity):
        self.pi = np.random.rand(self.K_all)
        self.pi = self.pi/self.pi.sum()
        self.N = self.pi * self.N_tot
        [self.model[k].set_param_empirical(T, intensity) for k in range(self.K)]
        return

    def add_hist_model(self, info, hist_model, trial):
        info.update({'N_tot_hist': np.array([hist_model[i]['N_tot'] for i in range(trial)]),
                     'Ea_hist': np.array([hist_model[i]['Ea'] for i in range(trial)]),
                     'tau0_hist': np.array([hist_model[i]['tau0'] for i in range(trial)]),
                     'Tp_hist': np.array([hist_model[i]['Tp'] for i in range(trial)])})
        return

    def print_param_summary(self, param):
        print('   Ea:       ' + ('{:5.3f} eV        ' * len(param['Ea'])).format(*param['Ea']))
        print('   tau0:     ' + ('{:6.3e} s     ' * len(param['tau0'])).format(*param['tau0']))
        print('   Tp:       ' + ('{:6.3e} K     ' * len(param['Tp'])).format(*param['Tp']))
        print('   N_tot:   {:6.3e} '.format(self.N_tot))
        print('   N:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi'] * self.N_tot))
        print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
        return

    def leastsq(self, T, intensity, stdout):
        print("Starting TSDC fitting via least square method.")
        print("Fitting parameters are {Ea, Tp, pi, and N_tot.}")

        def TSDC(T, param):
            dict_param = {"K":self.K, "Tp": list(param[0:self.K]),
                    "Ea": list(param[self.K:2*self.K]),
                    'N': list(param[2*self.K:3*self.K+self.K_all])
                    }
            if self.background == 'linear':
                dict_param.update({'s_tri': param[-1]})
            self.set_param(**dict_param)
            return self.predict(T) * self.N_tot #* self.beta

        def residual(param, x, y):
            return y - TSDC(x, param)

        init_param = np.zeros(2*self.K + self.K_all)
        init_param[0:self.K] = [self.model[k].Tp[0] for k in range(self.K)]
        init_param[self.K:2*self.K] = [self.model[k].Ea for k in range(self.K)]
        init_param[2*self.K:2*self.K + self.K_all] = self.pi * integrate.trapezoid(intensity, T)/self.beta

        start = time.time()
        lb = [self.T_min for i in range(self.K)]
        ub = [self.T_max for i in range(self.K)]
        [lb.append(self.Ea_min) for i in range(self.K)]
        [ub.append(self.Ea_max) for i in range(self.K)]
        [lb.append(0.0) for i in range(self.K_all)]
        [ub.append(np.inf) for i in range(self.K_all)]

        if self.background == 'linear':
            init_param = np.append(init_param, np.random.rand() * 1000)
            lb.append(-1000)
            ub.append(1000)

        bnds= (lb, ub)

        try:
            ls = optimize.least_squares(residual, init_param,
                                        args=(T, intensity),
                                        bounds=bnds,
                                        loss='linear'
                                        )
            opt_param = ls['x']
            rmse = np.sqrt((ls['cost']) * 2.0 / T.size)

        except ValueError:
            print("ValueError: Least square method is failed.")
            ls = {'success':False}
            opt_param = init_param
            rmse = np.nan
        except RuntimeError:
            print("RuntimeError: Least square method is failed.")
            ls = {'success':False}
            opt_param = init_param
            rmse = np.nan

        param_dict = {"K":self.K, "Tp": list(opt_param[0:self.K]),
                      "Ea": list(opt_param[self.K:2*self.K]),
                      'N': list(opt_param[2*self.K:2*self.K + self.K_all])}

        #hessian = np.dot(ls['jac'].T, ls['jac'])
        #det = np.linalg.det(hessian)
        #dim = init_param.size
        #print(det)
        #print("Estimated log Marginal likelihood via Laplace method:",
        #       - T.size/2 * np.log(2*np.pi* ls['cost']*2/T.size) - T.size/2
        #       + dim/2.0 * np.log(2*np.pi) - 1/2.0 * np.log(det)
        #      )
        #print("BIC: ", T.size * np.log(ls['cost']*2/T.size) + dim * np.log(T.size))

        self.set_param(**param_dict)
        t_tot = time.time() - start

        run_info = {
            'total_iter': np.nan,
            'total_time': t_tot,
            'time/iter': np.nan,
            'LL': np.nan,
            'LL_hist': np.nan,
            'LL_residual': np.nan,
            'LL_residual_hist': np.nan,
            'RMSE': rmse
        }

        param = self.export_param()
        if ls['success'] == True:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(rmse, t_tot))
            print('Estimated model parameters and scores are following:')
            print('   Ea:       ' + ('{:5.3f} eV        ' * len(param['Ea'])).format(*param['Ea']))
            print('   tau0:     ' + ('{:6.3e} s     ' * len(param['tau0'])).format(*param['tau0']))
            print('   N_tot:   {:6.3e} '.format(self.N_tot))
            print('   N:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*np.array(param['pi']) * self.N_tot))
            print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
            print('   RMSE:     {:12.8e}\n'.format(rmse))
        else:
            print("   Warning: non-linear least-square optimization is failed.")
        return run_info

    def leastsq_tau0(self, T, intensity, stdout):
        print("Starting TSDC fitting via least square method.")
        print("Fitting parameters are {Ea, tau0, pi, and N_tot.}")

        def TSDC(T, param):
            dict = {"K": self.K, "tau0": list(param[0:self.K]),
                    "Ea": list(param[self.K:2 * self.K]),
                    'N': list(param[2 * self.K:3 * self.K + self.K_all])
                    }
            self.set_param(**dict)
            return self.predict(T) * self.N_tot * self.beta

        def residual(param, x, y):
            return y - TSDC(x, param)
        self.init_param_empirical(T, intensity)
        print([self.model[k].tau0 for k in range(self.K)])
        init_param = np.zeros(2 * self.K + self.K_all)
        init_param[0:self.K] = [self.model[k].tau0 for k in range(self.K)]
        init_param[self.K:2 * self.K] = [self.model[k].Ea for k in range(self.K)]
        init_param[2 * self.K:2 * self.K + self.K_all] = self.pi * integrate.trapezoid(intensity, T) / self.beta

        start = time.time()
        lb = [1.0e-20 for i in range(self.K)]
        ub = [np.inf for i in range(self.K)]
        [lb.append(self.Ea_min) for i in range(self.K)]
        [ub.append(self.Ea_max) for i in range(self.K)]
        [lb.append(0.0) for i in range(self.K_all)]
        [ub.append(np.inf) for i in range(self.K_all)]
        bnds = (lb, ub)
        try:
            ls = optimize.least_squares(residual, init_param,
                                        args=(T, intensity),
                                        bounds=bnds,
                                        loss='linear'
                                        )
            opt_param = ls.x
            rmse = np.sqrt((ls['cost']) * 2.0 / T.size)

        except ValueError:
            print("Least square method is failed.")
            opt_param = init_param
            rmse = np.nan
        except RuntimeError:
            print("Least square method is failed.")
            opt_param = init_param
            rmse = np.nan

        param_dict = {"K": self.K, "tau0": list(opt_param[0:self.K]),
                      "Ea": list(opt_param[self.K:2*self.K]),
                      'N': list(opt_param[2*self.K:2*self.K + self.K_all])}
        self.set_param(**param_dict)
        t_tot = time.time() - start

        run_info = {
            'total_iter': np.nan,
            'total_time': t_tot,
            'time/iter': np.nan,
            'LL': np.nan,
            'LL_hist': np.nan,
            'LL_residual': np.nan,
            'LL_residual_hist': np.nan,
            'RMSE': rmse
        }

        param = self.export_param()
        if rmse != np.nan:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(rmse, t_tot))
            print('Estimated model parameters and scores are following:')
            print('   Ea:       ' + ('{:5.3f} eV        ' * len(param['Ea'])).format(*param['Ea']))
            print('   tau0:     ' + ('{:6.3e} s     ' * len(param['tau0'])).format(*param['tau0']))
            print('   N_tot:   {:6.3e} '.format(self.N_tot))
            print('   N:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*np.array(param['pi']) * self.N_tot))
            print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
            print('   RMSE:     {:12.8e}\n'.format(rmse))

        else:
            print("   Warning: non-linear least-square optimization is failed.")

        return run_info

    def l2_div(self, T, intensity, stdout):
        print("Starting TSDC fitting via least square method.")
        print("L2 divergence based estimation.")
        print("Fitting parameters are Ea, Tp, pi, and N_tot.")

        T_bin = np.arange(self.T_min, self.T_max, self.dT)
        freq = np.zeros(T_bin.size)
        window = np.max([np.abs((T - T_bin[i])).min() for i in range(T_bin.size)]) * 2.0

        for i in range(T_bin.size):
            index = np.where((T >= T_bin[i] - window / 2.0) & (T <= T_bin[i] + window / 2.0))
            freq[i] = intensity[index[0]].mean()
        Z = freq.sum() * self.dT

        def TSDC(T, param):
            dict_param = {"K":self.K, "Tp": list(param[0:self.K]),
                    "Ea": list(param[self.K:2*self.K]),
                    'N': list(param[2*self.K:3*self.K+self.K_all])
                    }
            if self.background == 'linear':
                dict_param.update({'s_tri': param[-1]})
            self.set_param(**dict_param)
            return self.predict(T) * Z

        def residual(param, x, y):
            return y - TSDC(x, param)

        init_param = np.zeros(2*self.K + self.K_all)
        init_param[0:self.K] = [self.model[k].Tp for k in range(self.K)]
        init_param[self.K:2*self.K] = [self.model[k].Ea for k in range(self.K)]
        init_param[2*self.K:2*self.K + self.K_all] = self.pi * integrate.trapezoid(intensity, T)/self.beta

        start = time.time()
        lb = [self.T_min for i in range(self.K)]
        ub = [self.T_max for i in range(self.K)]
        [lb.append(self.Ea_min) for i in range(self.K)]
        [ub.append(self.Ea_max) for i in range(self.K)]
        [lb.append(0.0) for i in range(self.K_all)]
        [ub.append(np.inf) for i in range(self.K_all)]

        if self.background == 'linear':
            init_param = np.append(init_param, np.random.rand() * 1000)
            lb.append(-1000)
            ub.append(1000)

        bnds = (lb, ub)

        try:
            ls = optimize.least_squares(residual, init_param,
                                        args=(T_bin, freq),
                                        bounds=bnds,
                                        loss='linear'
                                        )
            opt_param = ls.x
            rmse = np.sqrt((ls['cost']) * 2.0 / T.size)

        except ValueError:
            print("ValueError: Least square method is failed.")
            opt_param = init_param
            rmse = np.nan
        except RuntimeError:
            print("RuntimeError: Least square method is failed.")
            opt_param = init_param
            rmse = np.nan

        N = np.array(opt_param[2 * self.K:2 * self.K + self.K_all])
        N = N/(N.sum()*self.beta) * Z
        param_dict = {"K":self.K, "Tp": list(opt_param[0:self.K]),
                      "Ea": list(opt_param[self.K:2*self.K]),
                      'N': list(N)}

        self.set_param(**param_dict)
        t_tot = time.time() - start

        run_info = {
            'total_iter': np.nan,
            'total_time': t_tot,
            'time/iter': np.nan,
            'LL': np.nan,
            'LL_hist': np.nan,
            'LL_residual': np.nan,
            'LL_residual_hist': np.nan,
            'RMSE': rmse
        }

        param = self.export_param()
        if ls['success'] == True:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(rmse, t_tot))
            print('Estimated model parameters and scores are following:')
            print('   Ea:       ' + ('{:5.3f} eV        ' * len(param['Ea'])).format(*param['Ea']))
            print('   tau0:     ' + ('{:6.3e} s     ' * len(param['tau0'])).format(*param['tau0']))
            print('   N_tot:   {:6.3e} '.format(self.N_tot))
            print('   N:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*np.array(param['pi']) * self.N_tot))
            print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
            print('   RMSE:     {:12.8e}\n'.format(rmse))
        else:
            print("   Warning: non-linear least-square optimization is failed.")

        return run_info

    def regularization(self, T, intensity):
        print("Starting TSDC fitting via regularized least square method.")
        print("L2 divergence based estimation.")
        print("Fitting parameters are Ea, Tp, pi, and N_tot.")

        T_bin = np.arange(self.T_min, self.T_max, self.dT)
        freq = np.zeros(T_bin.size)
        window = np.max([np.abs((T - T_bin[i])).min() for i in range(T_bin.size)]) * 2.0

        for i in range(T_bin.size):
            index = np.where((T >= T_bin[i] - window / 2.0) & (T <= T_bin[i] + window / 2.0))
            freq[i] = intensity[index[0]].mean()
        Z = freq.sum() * self.dT

        def TSDC(T, param):
            dict_param = {"K":self.K, "Tp": list(param[0:self.K]),
                    "Ea": list(param[self.K:2*self.K]),
                    'N': list(param[2*self.K:3*self.K+self.K_all])
                    }
            if self.background == 'linear':
                dict_param.update({'s_tri': param[-1]})
            self.set_param(**dict_param)
            return self.predict(T) * Z

        def loss(param, x, y):
            return np.average((y - TSDC(x, param))**2) + 1000 * (param[0] - 600) ** 2 + 1000 * (param[1] - 700) ** 2

        init_param = np.zeros(2*self.K + self.K_all)
        init_param[0:self.K] = [self.model[k].Tp for k in range(self.K)]
        init_param[self.K:2*self.K] = [self.model[k].Ea for k in range(self.K)]
        init_param[2*self.K:2*self.K + self.K_all] = self.pi * integrate.trapezoid(intensity, T)/self.beta

        start = time.time()
        lb = [self.T_min for i in range(self.K)]
        ub = [self.T_max for i in range(self.K)]
        [lb.append(self.Ea_min) for i in range(self.K)]
        [ub.append(self.Ea_max) for i in range(self.K)]
        [lb.append(0.0) for i in range(self.K_all)]
        [ub.append(np.inf) for i in range(self.K_all)]

        if self.background == 'linear':
            init_param = np.append(init_param, np.random.rand() * 1000)
            lb.append(-1000)
            ub.append(1000)

        bnds = (lb, ub)
        bnds = []
        for k in range(self.K):
            bnds.append([self.T_min, self.T_max])
        for k in range(self.K):
            bnds.append([self.Ea_min, self.Ea_max])
        for k in range(self.K):
            bnds.append([0.0, np.infty])

        try:
            ls = optimize.minimize(loss, init_param,
                                        args=(T_bin, freq),
                                        bounds=bnds,
                                        method='L-BFGS-B'
                                        )
            print(ls)
            opt_param = ls.x
            rmse = 0.0#np.sqrt((ls['cost']) * 2.0 / T.size)

        except ValueError:
            print("ValueError: Least square method is failed.")
            opt_param = init_param
            rmse = np.nan

        except RuntimeError:
            print("RuntimeError: Least square method is failed.")
            opt_param = init_param
            rmse = np.nan

        N = np.array(opt_param[2 * self.K:2 * self.K + self.K_all])
        N = N/(N.sum()*self.beta) * Z
        param_dict = {"K":self.K, "Tp": list(opt_param[0:self.K]),
                      "Ea": list(opt_param[self.K:2*self.K]),
                      'N': list(N)}

        self.set_param(**param_dict)
        t_tot = time.time() - start

        run_info = {
            'total_iter': np.nan,
            'total_time': t_tot,
            'time/iter': np.nan,
            'LL': np.nan,
            'LL_hist': np.nan,
            'LL_residual': np.nan,
            'LL_residual_hist': np.nan,
            'RMSE': rmse
        }

        param = self.export_param()
        if ls['success'] == True:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(rmse, t_tot))
            print('Estimated model parameters and scores are following:')
            print('   Ea:       ' + ('{:5.3f} eV        ' * len(param['Ea'])).format(*param['Ea']))
            print('   tau0:     ' + ('{:6.3e} s     ' * len(param['tau0'])).format(*param['tau0']))
            print('   Tp:       ' + ('{:6.3e} s     ' * len(param['Tp'])).format(*param['Tp']))
            print('   N_tot:   {:6.3e} '.format(self.N_tot))
            print('   N:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*np.array(param['pi']) * self.N_tot))
            print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
            print('   RMSE:     {:12.8e}\n'.format(rmse))
        else:
            print("   Warning: non-linear least-square optimization is failed.")

        return run_info

    def sampling(self, x, intensity, method='adapted_em', trial=10,
                 max_iter=1000, r_eps=1e-7, criteria='likelihood', stdout=False):
        import copy
        from EMPeaks.EMCore._backend import get_backend
        
        hist_model = []
        hist_run_info = []

        # for reviewers comments
        Ea_conv = np.array([])
        Tp_conv = np.array([])
        Ea_not_conv = np.array([])
        Tp_not_conv = np.array([])

        if get_backend() == "rust" and method in ['adapted_em', 'smart']:
            import concurrent.futures
            
            def _run_tsdc_trial(i):
                model_copy = copy.deepcopy(self)
                model_copy.init_param_empirical(x, intensity)
                init_param = model_copy.export_param()
                if stdout:
                    print(init_param['Ea'])
                
                for k in range(model_copy.K):
                    model_copy.model[k].is_not_converged[0] = float(np.squeeze(init_param['Ea'][k]))
                    model_copy.model[k].is_not_converged[1] = float(np.squeeze(init_param['Tp'][k]))
                
                if stdout:
                    print('* Starting Trial # {:3d}'.format(i))
                    print("parameters are initialized as follows.")
                    print("\t Ea: ", init_param['Ea'])
                    print("\t Tp: ", list(init_param['Tp']))
                    print("")

                run_info = model_copy.fit(x, intensity, method=method, max_iter=max_iter, r_eps=r_eps, stdout=stdout)                
                tmp_param = model_copy.export_param()
                if stdout:
                    print("<<<<<<fitting of trial {:d} is finished.<<<<<<<<<".format(i))
                    print("")
                return tmp_param, run_info

            from EMPeaks.EMCore._backend import get_backend
            if get_backend() == "rust":
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(trial, 8)) as executor:
                    results = list(executor.map(_run_tsdc_trial, range(trial)))
            else:
                # 純粋なPython実装の場合、GILの競合によりThreadPoolExecutorがハングアップするため直列実行する
                results = [_run_tsdc_trial(i) for i in range(trial)]
            
            for res in results:
                tmp_param = res[0]
                run_info = res[1]
                hist_model.append(tmp_param)
                hist_run_info.append(run_info)

                #>>> for reviewers comments
                flag = np.array([False for i in range(self.K)])
                flag = np.array(tmp_param["is_not_converged"]).T[2].astype(bool)
                Ea_not_conv = np.append(Ea_not_conv, np.array(tmp_param["is_not_converged"]).T[0][flag])
                Tp_not_conv = np.append(Tp_not_conv, np.array(tmp_param["is_not_converged"]).T[1][flag])

                flag= ~flag
                Ea_conv = np.append(Ea_conv, np.array(tmp_param["is_not_converged"]).T[0][flag])
                Tp_conv = np.append(Tp_conv, np.array(tmp_param["is_not_converged"]).T[1][flag])
                
        else:
            for i in range(trial):
                # first initialization
                self.init_param_empirical(x, intensity)
                init_param = self.export_param()
                print(init_param['Ea'])
                for k in range(self.K):
                    self.model[k].is_not_converged[0] = float(np.squeeze(init_param['Ea'][k]))
                    self.model[k].is_not_converged[1] = float(np.squeeze(init_param['Tp'][k]))
     
                print('* Starting Trial # {:3d}'.format(i))
                print("parameters are initialized as follows.")
                print("\t Ea: ", init_param['Ea'])
                print("\t Tp: ", list(init_param['Tp']))
    
                print("")
    
                run_info = self.fit(x, intensity, method=method, max_iter=max_iter, r_eps=r_eps, stdout=stdout)                
    
                tmp_param = copy.deepcopy(self.export_param())
                hist_model.append(tmp_param)
                hist_run_info.append(run_info)
    
                print("<<<<<<fitting of trial {:d} is finished.<<<<<<<<<".format(i))
                print("")
    
                #>>> for reviewers comments
                flag = np.array([False for i in range(self.K)])
                flag = np.array(tmp_param["is_not_converged"]).T[2].astype(bool)
                Ea_not_conv = np.append(Ea_not_conv, np.array(tmp_param["is_not_converged"]).T[0][flag])
                Tp_not_conv = np.append(Tp_not_conv, np.array(tmp_param["is_not_converged"]).T[1][flag])
    
                flag= ~flag
                Ea_conv = np.append(Ea_conv, np.array(tmp_param["is_not_converged"]).T[0][flag])
                Tp_conv = np.append(Tp_conv, np.array(tmp_param["is_not_converged"]).T[1][flag])

        np.savetxt("conv.csv", np.array([Ea_conv, Tp_conv]).T, delimiter=",")
        np.savetxt("not_conv.csv", np.array([Ea_not_conv, Tp_not_conv]).T, delimiter=",")

        print("total count: ",np.size(Tp_conv)+np.size(Tp_not_conv))
        print("error rate: ",np.size(Tp_not_conv)/(np.size(Tp_conv)+np.size(Tp_not_conv)))
        # for reviewers comments <<<
            
        hist_LL = np.array([hist_run_info[i]['LL'] for i in range(trial)])
        hist_RMSE = np.array([hist_run_info[i]['RMSE'] for i in range(trial)])

        print('Sampling the different initial guess with {:3d} trial is finished.'.format(trial))

        if criteria == 'likelihood':
            index_best = int(np.nanargmax(hist_LL))
            print('Maximum Log-Likelihood is obtained in trial {:3d}'.format(index_best))
        elif criteria == 'rmse':
            index_sucess = ~np.isnan(hist_RMSE)
            index_best = int(np.nanargmin(hist_RMSE[index_sucess]))
            print(index_best)
            print('Minimum RMSE is obtained in trial {:3d}'.format(index_best))
        else:
            index_best = 0

        info = {'index_best': index_best,
                'LL_hist': hist_LL,
                'RMSE_hist': np.array([hist_run_info[i]['RMSE'] for i in range(trial)]),
                'time_hist': np.array([hist_run_info[i]['total_time'] for i in range(trial)]),
                'iter_hist': np.array([hist_run_info[i]['total_iter'] for i in range(trial)])
                }

        self.add_hist_model(info, hist_model, trial)

        param = hist_model[index_best]
        self.set_param(**param)

        print('Best model parameters and scores of samples are following:')
        self.print_param_summary(param)
        print('   LL:      {:12.8e}\n'
              '   RMSE:     {:12.8e}\n'.format(info['LL_hist'][index_best], info['RMSE_hist'][index_best])
              )

        return info