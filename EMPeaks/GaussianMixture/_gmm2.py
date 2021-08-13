from EMPeaks.GaussianMixture._gaussian import Gaussian
# from ..Background import UniformModel, SquareRootModel, LinearModel, TriangleModel, RampModel
import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import copy
import time


class GaussianMixtureModel:
    def __init__(self, K=2, x_min=-300, x_max=300, sigma_min=0.1, sigma_max=50,
                 background='none', k_ramp=0):
        self.K = K
        self.x_min = x_min
        self.x_max = x_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.background = background
        self.dx = 1.0

        self.model = [Gaussian(x_min, x_max, sigma_min, sigma_max) for k in range(self.K)]
        self.pi = np.ones(self.K) / self.K
        self.N_tot = 1.0
        self.N = self.pi * self.N_tot

        if self.background == 'none':
            self.K_all = K
        # elif self.background == 'uniform':
        #     self.K_all = K + 1
        #     self.pi = np.append(self.pi, 1.0e-4)
        #     self.pi = self.pi / np.sum(self.pi)
        #     self.model.append(UniformModel(self.x_min, self.x_max))
        # elif self.background == 'squareroot':
        #     self.K_all = K + 1
        #     self.pi = np.append(self.pi, 1.0e-4)
        #     self.pi = self.pi / np.sum(self.pi)
        #     self.model.append(SquareRootModel(self.x_min, self.x_max))
        # elif self.background == 'linear':
        #     self.K_all = K + 1
        #     self.pi = np.append(self.pi, 1.0e-4)
        #     self.pi = self.pi / np.sum(self.pi)
        #     self.model.append(LinearModel(self.x_min, self.x_max))
        # elif self.background == 'ramp_sum':
        #     print("RampSum Background is set.")
        #     self.k_ramp = k_ramp
        #     self.K_all = K + self.k_ramp + 2
        #     self.ramp_node = np.linspace(self.x_min, self.x_max, self.k_ramp + 1, endpoint=False)
        #     self.pi = np.append(self.pi, np.random.rand(self.k_ramp + 2))
        #     self.pi = self.pi / np.sum(self.pi)
        #     self.model.append(UniformModel(self.x_min, self.x_max))
        #     for k in range(k_ramp):
        #         self.model.append(RampModel(self.ramp_node[k], self.ramp_node[k + 1], self.x_max))
        #     self.model.append(TriangleModel(self.ramp_node[-1], self.x_max))
        # elif self.background == 'sharley':
        #     print("Sharley Background is set.")
        #     self.K_all = K + 2
        #     self.pi = np.append(self.pi, 1.0e-4)
        #     self.pi = np.append(self.pi, 1.0e-4)
        #     self.pi = self.pi / np.sum(self.pi)
        #     self.model.append(UniformModel(self.x_min, self.x_max))
        #     self.model.append(Sharley(self.K, self.x_min, self.x_max))
        #     self.model[-1].peak_model.model = self.model[0:K]
        else:
            print("Setting Background is not implemented.")
        self.N = self.pi * self.N_tot

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

        # setting mixture ratio pi, P0, and P0_tot
        if param_keys >= {'N'}:
            self.N = param['N']
            self.N_tot = np.sum(param['N'])
            self.pi = param['N'] / self.N_tot
            return

        if param_keys >= {'pi'}:
            if type(param['pi']) is not list:
                print("Parameter \"pi\" is not list type. Then pi is uniformly set as default.")
                self.pi = np.ones(self.K_all) / self.K_all
                return
            if len(param["pi"]) is self.K_all:
                self.pi = np.array(param['pi']) / sum(param['pi'])
                if sum(param["pi"]) is not 1.0:
                    print("Sum of \"pi\" must be unity. Then values are normalized as ", self.pi)
                return
            elif len(param["pi"]) is self.K:
                print("Parameter \"pi\" does not have background ratio. pi for background is set to be 0.1.")
                self.pi = np.array(param['pi']) / sum(param['pi']) * 0.9
                self.pi = np.append(self.pi, 0.1)
                if sum(param["pi"]) is not 1.0:
                    print("Sum of \"pi\" must be unity. Then values are normalized as ", self.pi)
                return
            else:
                print("Parameter \"pi\" does not have relevant length. Then pi is uniformly set as default.")
                self.pi = np.ones(self.K_all) / self.K_all
                return

    def set_param_background(self, **param):
        # setting parameters for Background.
        if self.background == 'none':
            self.K_all = self.K

        # elif self.background == 'uniform':
        #     self.K_all = self.K + 1
        #
        #     self.model.append(UniformModel(self.x_min, self.x_max))
        # elif self.background == 'squareroot':
        #     self.K_all = self.K + 1
        #     self.model.append(SquareRootModel(self.x_min, self.x_max))
        #
        # elif self.background == 'linear':
        #     self.K_all = self.K + 1
        #     if ('s_tri' in param) and (0 <= param['s_tri'] <= 1.0):
        #         self.model.append(LinearModel(self.x_min, self.x_max, s_tri=param['s_tri']))
        #     else:
        #         self.model.append(LinearModel(self.x_min, self.x_max))
        #
        # elif self.background == 'ramp_sum':
        #     self.K_all = self.K + self.k_ramp + 2
        #     self.ramp_node = np.linspace(self.x_min, self.x_max, self.k_ramp + 1, endpoint=False)
        #     # self.pi = np.append(self.pi, np.random.rand(self.k_ramp + 2))
        #     # self.pi = self.pi / np.sum(self.pi)
        #     self.model.append(UniformModel(self.x_min, self.x_max))
        #     for k in range(self.k_ramp):
        #         self.model.append(RampModel(self.ramp_node[k], self.ramp_node[k + 1], self.x_max))
        #     self.model.append(TriangleModel(self.ramp_node[-1], self.x_max))
        #
        # elif self.background == 'sharley':
        #     self.K_all = self.K + 2
        #     self.model.append(UniformModel(self.x_min, self.x_max))
        #     self.model.append(Sharley(self.K, self.x_min, self.x_max))
        #     self.model[-1].peak_model.model = self.model[0:self.K]

    def extract_single_params(self, param_set, **param):
        org_K = self.K
        param_keys = set(param.keys())
        single_params = []
        for k in range(self.K):
            dict = {}
            for key in param_keys & param_set:
                if type(param[key]) is not list:
                    print("Error: SingleModel parameters must be list type.")
                    self.K = org_K
                    return
                if len(param[key]) is not param['K']:
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
        # if self.background is 'linear':
        #     s_in_linear = [self.model[-1].s_uni, self.model[-1].s_tri]
        # if self.background is 'linear':
        #     self.model[-1].s_uni = s_in_linear[0]
        #     self.model[-1].s_tri = s_in_linear[1]
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

    def init_param_uniform(self):
        self.pi = np.random.rand(self.K_all)
        self.pi = self.pi / self.pi.sum()
        self.N = self.pi * self.N_tot
        return [self.model[k].init_model() for k in range(self.K)]

    def predict(self, x):
        return np.sum([self.pi[k] * self.model[k].predict(x) for k in range(self.K_all)], axis=0)

    def log_likelihood(self, x, intensity):
        return np.sum(intensity * np.log(self.predict(x) + 1e-200))

    def fit(self, x, intensity, method='adapted_em', max_iter=3000, r_eps=1e-9,
            stdout=True, trial=10, criteria='likelihood'):
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        # if self.background is "uniform":
        #     self.model[-1] = UniformModel(self.x_min, self.x_max)
        # if self.background is "squareroot":
        #     self.model[-1] = SquareRootModel(self.x_min, self.x_max)
        # if self.background is "linear":
        #     self.model[-1] = LinearModel(self.x_min, self.x_max)
        # elif self.background == 'ramp_sum':
        #     self.model.append(UniformModel(self.x_min, self.x_max))
        #     for k in range(self.k_ramp):
        #         self.model.append(RampModel(self.ramp_node[k], self.ramp_node[k + 1], self.x_max))
        #     self.model.append(TriangleModel(self.ramp_node[-1], self.x_max))
        # elif self.background == 'sharley':
        #     self.model.append(UniformModel(self.x_min, self.x_max))
        #     self.model.append(Sharley(self.K, self.x_min, self.x_max))

        if method is 'leastsq':
            info = self.leastsq(x, intensity, stdout)
            return info

        if method is 'l2div':
            info = self.l2_div(x, intensity, stdout)
            return info

        elif method is 'adapted_em':
            info = self.adapted_em(x, intensity, max_iter, r_eps, stdout)
            return info

        elif method is 'smart':
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

        if criteria is 'likelihood':
            index_best = int(np.nanargmax(hist_LL))
            print('Maximum Log-Likelihood is obtained in trial {:3d}'.format(index_best))
        elif criteria is 'rmse':
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

            if stdout is True:
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

        # rmse = self.leastsq_for_normalization_factor(x, intensity, stdout)
        self.N_tot = np.sum(intensity)
        rmse = np.sqrt(np.average((intensity - self.predict(x) * self.N_tot)**2))
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
        if stdout is True:
            print('Estimated model parameters and scores are following:')
            self.print_param_summary(param)
            print('   LL:      {:12.8e}\n'
                  '   RMSE:     {:12.8e}\n'.format(run_info['LL'], run_info['RMSE'])
                  )

        return run_info

    def print_param_summary(self, param):
        print('   mu:        ' + ('{:5.3f} eV       ' * len(param['mu'])).format(*param['mu']))
        print('   sigma:     ' + ('{:6.3e}          ' * len(param['sigma'])).format(*param['sigma']))
        print('   N_tot:   {:6.3e} '.format(self.N_tot))
        print('   N:         ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi'] * self.N_tot))
        print('   pi:        ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
        return

    def e_step(self, x):
        eps = 1e-20
        self._gamma = np.array([self.pi[k] * self.model[k].predict(x)
                                / (self.predict(x) + eps) for k in range(self.K_all)])
        return

    def m_step(self, x, intensity):
        self.pi = np.array([np.sum(intensity * self._gamma[k]) for k in range(self.K_all)])
        self.pi = self.pi / np.sum(self.pi)
        [self.model[k].maximum_likelihood_estimation(x, intensity * self._gamma[k]) for k in range(self.K_all)]
        return

    def leastsq_for_normalization_factor(self, x, intensity, stdout):
        print('<< Optimizing normalization factor by using least square method. >>')

        def model(x, param):
            return self.predict(x) * param

        def residual(param, x, y):
            return y - model(x, param)

        init_param = integrate.trapz(intensity, x)
        print('init', init_param)

        start = time.time()
        ls = optimize.least_squares(residual, init_param,
                                    args=(x, intensity),
                                    loss='linear'
                                    )
        self.N_tot = ls.x[0]
        self.N = list(self.pi * self.N_tot)

        rmse = np.sqrt((ls['cost']) * 2.0 / x.size)
        if ls["success"] is True:
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
            # if self.background is 'linear':
            #     dict_param.update({'s_tri': param[-1]})
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

        # if self.background is 'linear':
        #     init_param = np.append(init_param, np.random.rand() * 1000)
        #     lb.append(-1000)
        #     ub.append(1000)

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
        if flag is True:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(run_info['rmse'], run_info['t_tot']))
            print('Estimated model parameters and scores are following:')
            self.print_param_summary(param)
            print('   RMSE:     {:12.8e}\n'.format(run_info['rmse']))
        else:
            print("   Warning: non-linear least-square optimization is failed.")

        return run_info

    def plot(self, x_data, intensity):
        figsize = (8, 3)
        fig = plt.figure(figsize=figsize)
        x = np.arange(self.x_min, self.x_max)

        ax = fig.add_subplot(1, 1, 1)
        for k in range(self.K):
            ax.plot(x, self.model[k].predict(x) * self.N[k], label='model_' + str(k))
        if self.background is 'sharley':
            y = self.model[-1].predict(x) * self.N[-1] + self.model[-2].predict(x) * self.N[-2]
            ax.plot(x, y, label=self.background)
        else:
            ax.plot(x, self.model[-1].predict(x) * self.N[-1], label=self.background)

        ax.plot(x, self.predict(x) * self.N_tot, 'black', linewidth=3, ls='--', label='full_model')
        ax.scatter(x_data, intensity, label='data')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('Intensity')
        ax.legend()
        plt.show()
        return


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
