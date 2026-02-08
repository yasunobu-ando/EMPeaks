# License: BSD-3-clause
# Copyright © 2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu ANDO

import logging
from typing import Dict, List, Optional, Union, Any

from EMPeaks.EMCore._gaussian import Gaussian
from EMPeaks.EMCore.exceptions import BackgroundTypeError, ParameterError
from EMPeaks.EMCore.background_factory import BackgroundFactory
from EMPeaks.EMCore.visualizer import Visualizer
from EMPeaks.EMCore.fitting_engine import FittingEngine
from EMPeaks.Background import UniformModel, SquareRootModel, LinearModel, TriangleModel, RampModel
from scipy import integrate
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

# Module logger
logger = logging.getLogger(__name__)


class EMCore:
    # Class constants
    INITIAL_BACKGROUND_WEIGHT = 1.0e-4
    EPSILON_LOG = 1e-200
    EPSILON_PREDICT = 1e-20
    DEFAULT_MAX_ITER = 3000
    DEFAULT_R_EPS = 1e-9

    def __init__(self, K=2, x_min=-300, x_max=300, sigma_min=0.1, sigma_max=50,
                 background='none', k_ramp=5):
        self.K = K
        self.x_min = x_min
        self.x_max = x_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.background = background
        self.dx = 1.0
        self.k_ramp = k_ramp  # Store k_ramp for ramp_sum
        
        # Initialize BackgroundFactory for delegation
        self._bg_factory = BackgroundFactory(x_min, x_max, k_ramp)

        self.model = [Gaussian(x_min, x_max, sigma_min, sigma_max) for k in range(self.K)]
        self.pi = np.ones(self.K) / self.K

        # Initialize background models using factory method
        if self.background == 'ramp_sum':
            print("RampSum Background is set.")
            self.ramp_node = np.linspace(self.x_min, self.x_max, self.k_ramp + 1, endpoint=False)
            background_models = self._bg_factory.create(self.background)
            self.K_all = K + len(background_models)
            self._update_mixture_weights(len(background_models), use_random=True)
        elif self.background == 'none':
            self.K_all = K
        elif self.background in ('uniform', 'squareroot', 'linear'):
            background_models = self._bg_factory.create(self.background)
            self.K_all = K + len(background_models)
            self._update_mixture_weights(len(background_models))
            self.model.extend(background_models)
        else:
            print("Setting Background is not implemented.")
            self.K_all = K

        # Add ramp_sum models after weight update
        if self.background == 'ramp_sum':
            self.model.extend(background_models)

        self.N_tot = 1.0
        self.N = self.pi * self.N_tot
        self.Dirichlet_alpha = np.ones(self.K_all)

    def set_param(self, **param):
        """
        """
        if not param:
            return self

        param_keys = set(param.keys())
        if param_keys >= {'K'}:
            self.K = param['K']
        else:
            param['K'] = self.K

        self.set_single_params(**param)
        self.set_param_background(**param)

        # setting mixture ratio pi, P0, and P0_tot
        if param_keys >= {'N'}:
            self.N = param['N']
            self.N_tot = np.sum(param['N'])
            self.pi = param['N'] / self.N_tot
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

    def set_param_background(self, **param):
        """
        # setting parameters for Background. (in _em_core.py)
        :param param:
        :return:
        """
        if self.background == 'none':
            self.K_all = self.K
        elif self.background == 'ramp_sum':
            self.K_all = self.K + self.k_ramp + 2
            self.ramp_node = np.linspace(self.x_min, self.x_max, self.k_ramp + 1, endpoint=False)
            background_models = self._bg_factory.create(self.background)
            self.model.extend(background_models)
        elif self.background in ('uniform', 'squareroot', 'linear'):
            self.K_all = self.K + 1
            # Pass s_tri if provided for linear background
            s_tri = param.get('s_tri', None)
            background_models = self._bg_factory.create(self.background, s_tri=s_tri)
            self.model.extend(background_models)

    def extract_single_params(self, param_set, **param):
        org_K = self.K
        param_keys = set(param.keys())
        single_params = []
        for k in range(self.K):
            dict = {}
            for key in param_keys & param_set:
                if type(param[key]) != list:
                    print("Error: SingleModel parameters must be list type.")
                    self.K = org_K
                    return
                if len(param[key]) != param['K']:
                    print("Error: length of parameters \"" + key + "\" is not consistent with value K.")
                    self.K = org_K
                    return
                dict[key] = param[key][k]
            single_params.append(dict)

            instance_keys = set(self.__dict__.keys())
            for key in (instance_keys & param_keys):
                self.__setattr__(key, param[key])

        return single_params

    def set_single_params(self, **param):
        # setting parameters for each single Gaussian model.
        param_set = {"mu", "sigma"}
        single_params = self.extract_single_params(param_set, **param)
        self.model = [Gaussian(self.x_min, self.x_max, self.sigma_min, self.sigma_max) for k in range(self.K)]
        [self.model[k].set_param(**single_params[k]) for k in range(self.K)]
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
        """ parameters are sorted in the order of the first element in param_set."""
        param_set = {"mu", "sigma"}
        _tmp = {}
        for param in list(param_set):
            _tmp[param] = [self.model[k].__dict__[param] for k in range(self.K)]

        _tmp_index = np.array(_tmp[list(param_set)[0]]).argsort()

        for param in list(param_set):
            _tmp_param[param] = list(np.array(_tmp[param])[_tmp_index])

        return _tmp_param, _tmp_index

    def _update_mixture_weights(self, n_background_models, use_random=False):
        """背景モデル追加時に混合重みを更新
        
        Args:
            n_background_models: 追加する背景モデルの数
            use_random: Trueならランダム初期化、Falseなら定数初期化
        """
        if n_background_models == 0:
            return
        if use_random:
            new_weights = np.random.rand(n_background_models)
        else:
            new_weights = np.full(n_background_models, self.INITIAL_BACKGROUND_WEIGHT)
        self.pi = np.append(self.pi, new_weights)
        self.pi = self.pi / np.sum(self.pi)

    def init_param_uniform(self):
        self.pi = np.random.rand(self.K_all)
        self.pi = self.pi / self.pi.sum()
        self.N = self.pi * self.N_tot
        return [self.model[k].init_model() for k in range(self.K)]

    def predict(self, x):
        return np.sum([self.pi[k] * self.model[k].predict(x) for k in range(self.K_all)], axis=0)

    def log_likelihood(self, x, intensity):
        return np.sum(intensity * np.log(self.predict(x) + self.EPSILON_LOG))

    def fit(self, x, intensity, method='adapted_em', max_iter=3000, r_eps=1e-9,
            stdout=True, trial=10, criteria='likelihood'):
        print("background: {}".format(self.background))

        self.x_min = np.min(x)
        self.x_max = np.max(x)
        
        # Recreate background model with updated x_min/x_max
        if self.background in ('uniform', 'squareroot', 'linear'):
            self._bg_factory.update_range(self.x_min, self.x_max)
            self.model[-1] = self._bg_factory.create(self.background)[0]
        elif self.background == 'ramp_sum':
            # For ramp_sum, replace the background models (last k_ramp+2 models)
            n_bg = self.k_ramp + 2
            self.model = self.model[:self.K]  # Keep only peak models
            self._bg_factory.update_range(self.x_min, self.x_max)
            background_models = self._bg_factory.create(self.background)
            self.model.extend(background_models)

        if method == 'leastsq':
            print("**** Start spectrum fitting via Least-Square algorithm ****")
            info = self.leastsq(x, intensity, stdout)
            return info

        if method == 'l2div':
            print("**** Start spectrum fitting via L2-divergence based algorithm ****")
            info = self.l2_div(x, intensity, stdout)
            return info

        if method == 'leastsq_tau0':
            print("**** Start spectrum fitting via L2-divergence based algorithm ****")
            print("**** leastsq_tau0 works only in TSDC model.")
            info = self.leastsq_tau0(x, intensity, stdout)
            return info

        elif method == 'adapted_em':
            print("**** Start spectrum fitting via EM algorithm ****")
            info = self.adapted_em(x, intensity, max_iter, r_eps, stdout)
            return info

        elif method == 'smart':
            print('Start smart fitting process.')
            print('>>> Step 1: Sampling {:3d} trials with low threshold of 1.0e-6.'.format(trial))
            info1 = self.sampling(x, intensity, method='adapted_em',
                                  trial=trial, max_iter=max_iter, r_eps=1.0e-6, criteria=criteria, stdout=False)
            self.plot(x, intensity)
            print('>>> Step 2: Adapted_EM estimation with high threshold of {:}.'.format(r_eps))
            info2 = self.adapted_em(x, intensity, max_iter, r_eps, stdout)
            self.plot(x, intensity)
            print('>>> Step 3: Non-linear Least Square Optimization from Step 2.')
            self.leastsq(x, intensity, stdout=True)
            self.plot(x, intensity)

            return info1, info2

        else:
            print("ERROR !!! setting method is not implemented. Please set the method appropriately.")
            return

    def sampling(self, x, intensity, method='adapted_em', trial=10,
                 max_iter=1000, r_eps=1e-7, criteria='likelihood', stdout=False):
        hist_model = []
        hist_run_info = []
        for i in range(trial):
            self.init_param_uniform()
            print('* Starting Trial # {:3d}'.format(i))
            run_info = self.fit(x, intensity, method=method, max_iter=max_iter, r_eps=r_eps, stdout=stdout)
            tmp_param = copy.deepcopy(self.export_param())
            hist_model.append(tmp_param)
            hist_run_info.append(run_info)

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

    def deterministic_annealing(self, x, intensity, method='adapted_em', temp0=3, n_temp=5, trial=10,
                                max_iter=3000, r_eps=1e-9, criteria='likelihood', stdout=False):
        """
        temp0: in reference, t_0
        n_temp: in reference, H
        """
        print("Start deterministic annealing.")
        print("Please check class variable Dirichlet_alpha: {}".format(self.Dirichlet_alpha))
        print("If Dirichlet_alpha is one, no sparsity is introduced.")

        # confirm the validity of temp0
        print("Average of data intensity: ", np.average(intensity))
        # creating temprerature profile
        temp_power = np.array([temp0 + (h-1)*(np.log10(np.sum(intensity))-temp0)/(n_temp-1) 
                                 for h in range(1, n_temp+1)])
        temp_profile = np.array([np.sum(intensity)/10**temp_power[h] for h in range(0,n_temp)])

        hist_model = []
        hist_run_info = []
        for temp in temp_profile:
            print("Stage Temperature: {}:".format(temp))
            run_info = self.fit(x, intensity/temp, method=method, max_iter=max_iter, r_eps=r_eps, stdout=stdout)
            tmp_param = copy.deepcopy(self.export_param())
            hist_model.append(tmp_param)
            hist_run_info.append(run_info)

        hist_LL = np.array([hist_run_info[i]['LL'] for i in range(n_temp)])

        print('Deterministic Annealing with {:3d} temperature stage is finished.'.format(n_temp))
        print()
        info = {'LL_hist': hist_LL,
                'RMSE_hist': np.array([hist_run_info[i]['RMSE'] for i in range(n_temp)]),
                'time_hist': np.array([hist_run_info[i]['total_time'] for i in range(n_temp)]),
                'iter_hist': np.array([hist_run_info[i]['total_iter'] for i in range(n_temp)])
                }
        tmp = self.pi[0:self.K]
        K_non_zero = tmp[tmp>0].size
        print("number of non zero components of spectrum is {} in {}".format(K_non_zero, self.K))
        print("index of non zero components of spectrum: ", np.nonzero(tmp)[0])
        print('Final model parameters and scores of samples are following:')
        self.print_param_summary(hist_model[-1])
        print('   LL:      {:12.8e}\n'
              '   RMSE:     {:12.8e}\n'.format(info['LL_hist'][-1], info['RMSE_hist'][-1])
              )
        return info

    def add_hist_model(self, info, hist_model, trial):
        info.update({'mu_hist': np.array([hist_model[i]['mu'] for i in range(trial)]),
                     'sigma_hist': np.array([hist_model[i]['sigma'] for i in range(trial)])})
        return

    def adapted_em(self, x, intensity, max_iter, r_eps, stdout):
        start = time.time()
        it_tot = 0
        t_tot = 0
        LL_hist = []
        res_hist = []
        ll_0 = self.log_likelihood(x, intensity)
        flag = True
        ll = 0.0
        residual = 0.0

        print("<< Start fitting via Adapted EM Algorithm. >>")
        tmp_it = range(max_iter)
        t1 = time.time()

        for it in tmp_it:
            self.e_step(x)
            self.m_step(x, intensity)

            ll = self.log_likelihood(x, intensity)
            residual = (ll - ll_0) / np.abs(ll_0)
            t2 = time.time()

            if stdout == True:
                if it % 10 == 0:
                    print("> iteration #{:3d}, LL={:10.8e}, residual={:4.3e}, elapsed time: {:5.2f} s"
                          .format(it, ll, residual, t2 - t1))
                    t1 = time.time()

            LL_hist.append(ll)
            res_hist.append(residual)

            if residual < 0.0:
                print("Warning!!!! residual is negative!!!  Parameters are initialized again.")
                self.init_param_uniform()
            elif residual < r_eps:
                t_tot = time.time() - start
                it_tot = it

                print('Convergence is achieved at iter. {:3d}, elapsed time {:5.2f} s'
                      .format(it, t_tot))
                print('   LogLikelihood:     {:12.8e}\n'
                      '        residual:      {:12.8e}'.format(ll, residual))
                flag = False
                break
            ll_0 = ll

        if flag:
            t_tot = time.time() - start
            it_tot = max_iter
            print('>>> Convergence is not achieved within the iteration of {:3d}, elapsed time: {:5.2f} s'
                  .format(max_iter, t_tot))
            print('   LogLikelihood:      {:12.8e}\n'
                  '        residual:       {:12.8e}'.format(ll, residual))

        rmse = self.leastsq_for_normalization_factor(x, intensity, stdout)
        # self.N_tot = np.sum(intensity)
        # rmse = np.sqrt(np.average((intensity - self.predict(x) * self.N_tot)**2))
        param = self.export_param()

        run_info = {
            'total_iter': (it_tot + 1),
            'total_time': t_tot,
            'time/iter': t_tot / (it_tot + 1),
            'LL': ll,
            'LL_hist': LL_hist,
            'LL_residual': residual,
            'LL_residual_hist': res_hist,
            'RMSE': rmse
        }
        if stdout == True:
            print('Estimated model parameters and scores are following:')
            self.print_param_summary(param)
            print('   LL:      {:12.8e}\n'
                  '   RMSE:     {:12.8e}\n'.format(run_info['LL'], run_info['RMSE'])
                  )

        return run_info

    def print_param_summary(self, param):
        """
        This function depends on a mixture model. Please define in a setting files of new mixture models.
        :param param:
        :return:
        """
        print('   mu:        ' + ('{:5.3f} eV       ' * len(param['mu'])).format(*param['mu']))
        print('   sigma:     ' + ('{:6.3e}          ' * len(param['sigma'])).format(*param['sigma']))
        print('   N_tot:   {:6.3e} '.format(self.N_tot))
        print('   N:         ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi'] * self.N_tot))
        print('   pi:        ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
        return

    def e_step(self, x):
        eps = self.EPSILON_PREDICT
        self._gamma = np.array([self.pi[k] * self.model[k].predict(x)
                                / (self.predict(x) + eps) for k in range(self.K_all)])
        return

    def m_step(self, x, intensity):        
        N_k = np.array([np.sum(intensity * self._gamma[k]) for k in range(self.K_all)])
        self.pi = (N_k + self.Dirichlet_alpha - 1) / np.sum(N_k + self.Dirichlet_alpha -1)
        # altering the negative mixing ratio to zero
        self.pi[self.pi < 0] = 0.0
        # renormalizing the pi (合ってる？ Is it correct????)
        self.pi = self.pi/np.sum(self.pi) 
        [self.model[k].maximum_likelihood_estimation(x, intensity * self._gamma[k]) for k in range(self.K_all)]
        return

    def leastsq_for_normalization_factor(self, x, intensity, stdout):
        print('<< Optimizing normalization factor by using least square method. >>')

        def model(x, param):
            return self.predict(x) * param

        def residual(param, x, y):
            return y - model(x, param)

        init_param = np.abs(integrate.trapz(intensity, x))
        print('init', init_param)

        start = time.time()
        ls = optimize.least_squares(residual, init_param,
                                    args=(x, intensity),
                                    loss='linear'
                                    )
        self.N_tot = ls.x[0]
        self.N = list(self.pi * self.N_tot)

        rmse = np.sqrt((ls['cost']) * 2.0 / x.size)
        if ls["success"] == True:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(rmse, time.time() - start))
        else:
            print("   Warning: non-linear least-square optimization is failed.")

        return rmse

    def l2_div(self, x, intensity, stdout):
        print("Starting TSDC fitting via least square method.")
        print("L2 divergence based estimation.")
        print("Fitting parameters are Ea, Tp, pi, and P0_tot.")

        flag=True
        x_bin = np.arange(self.x_min, self.x_max, self.dx)
        freq = np.zeros(x_bin.size)
        window = np.max([np.abs((x - x_bin[i])).min() for i in range(x_bin.size)]) * 2.0

        for i in range(x_bin.size):
            index = np.where((x >= x_bin[i] - window / 2.0) & (x <= x_bin[i] + window / 2.0))
            freq[i] = intensity[index[0]].mean()
        Z = freq.sum() * self.dx

        def model(x, param):
            dict_param = {"K": self.K, "x0": list(param[0:self.K]),
                          "gamma": list(param[self.K:2 * self.K]),
                          'P0': list(param[2 * self.K:3 * self.K + self.K_all])
                          }
            if self.background == 'linear':
                dict_param.update({'s_tri': param[-1]})
            self.set_param(**dict_param)
            return self.predict(x) * Z

        def residual(param, x, y):
            return y - model(x, param)

        init_param = np.zeros(2 * self.K + self.K_all)
        init_param[0:self.K] = [self.model[k].x0 for k in range(self.K)]
        init_param[self.K:2 * self.K] = [self.model[k].gamma for k in range(self.K)]
        init_param[2 * self.K:2 * self.K + self.K_all] = self.pi * integrate.trapz(intensity, x)

        start = time.time()
        lb = [self.x_min for i in range(self.K)]
        ub = [self.x_max for i in range(self.K)]
        [lb.append(self.gamma_min) for i in range(self.K)]
        [ub.append(self.gamma_max) for i in range(self.K)]
        [lb.append(0.0) for i in range(self.K_all)]
        [ub.append(np.inf) for i in range(self.K_all)]

        if self.background == 'linear':
            init_param = np.append(init_param, np.random.rand() * 1000)
            lb.append(-1000)
            ub.append(1000)

        bnds = (lb, ub)

        try:
            ls = optimize.least_squares(residual, init_param,
                                        args=(x_bin, freq),
                                        bounds=bnds,
                                        loss='linear'
                                        )
            opt_param = ls.x
            rmse = np.sqrt((ls['cost']) * 2.0 / x.size)

        except ValueError:
            print("ValueError: Least square method is failed.")
            opt_param = init_param
            rmse = np.nan
            ls = {'success': False}

        except RuntimeError:
            print("RuntimeError: Least square method is failed.")
            opt_param = init_param
            rmse = np.nan
            ls = {'success': False}

        P0 = np.array(opt_param[2 * self.K:2 * self.K + self.K_all])
        P0 = P0 / P0.sum() * Z
        param_dict = {"K": self.K, "x0": list(opt_param[0:self.K]),
                      "gamma": list(opt_param[self.K:2 * self.K]),
                      'P0': list(P0)}

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
        if flag == True:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(run_info['rmse'], run_info['t_tot']))
            print('Estimated model parameters and scores are following:')
            self.print_param_summary(param)
            print('   RMSE:     {:12.8e}\n'.format(run_info['rmse']))
        else:
            print("   Warning: non-linear least-square optimization is failed.")

        return run_info


# class Sharley():
#     def __init__(self, K, x_min, x_max):
#         self.K = K
#         self.x_min = x_min
#         self.x_max = x_max
#         self.dx = 1.0
#         self.pi = np.ones(K)/K
#         self.peak_model = GaussianMixtureModel(K, x_min, x_max)
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
