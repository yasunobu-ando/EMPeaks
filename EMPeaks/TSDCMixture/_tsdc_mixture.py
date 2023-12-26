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
        self.P0_tot = 1.0
        self.P0 = self.pi * self.P0_tot

    def set_single_params(self, **param):
        # setting parameters for each single Gaussian model.
        param_set = {'Tp', 'Ea', 'tau0'}
        single_params = self.extract_single_params(param_set, **param)
        self.model = [TSDC(self.beta, self.T_min, self.T_max, self.Ea_min, self.Ea_max) for k in range(self.K)]
        [self.model[k].set_param(**single_params[k]) for k in range(self.K)]
        return

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
        if param_keys >= {'P0'}:
            self.P0 = param['P0']
            self.P0_tot = np.sum(param['P0'])
            self.pi = param['P0'] / self.P0_tot
            return

        # setting mixture ratio pi, P0, and P0_tot
        if param_keys >= {'P0'}:
            self.P0 = param['P0']
            self.P0_tot = np.sum(param['P0'])
            self.pi = param['P0']/self.P0_tot
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

    def export_param(self):
        _tmp_param = self.__dict__
        _tmp_param, _tmp_index = self.export_single_params(_tmp_param)
        if self.background is 'linear':
            s_in_linear = [self.model[-1].s_uni, self.model[-1].s_tri]
        if self.background is 'linear':
            self.model[-1].s_uni = s_in_linear[0]
            self.model[-1].s_tri = s_in_linear[1]
        _tmp_param['pi'][0:self.K] = list(np.array(self.pi)[0:self.K][_tmp_index])
        _tmp_param['P0'] = list(np.array(_tmp_param['pi']) * self.P0_tot)
        self.set_param(**_tmp_param)
        return _tmp_param

    def export_single_params(self, _tmp_param):
        """ parameters are sorted in the order of the first element in param_set."""
        param_set = {"Ea", "tau0", "Tp"}
        _tmp = {}
        for param in list(param_set):
            _tmp[param] = [self.model[k].__dict__[param] for k in range(self.K)]

        _tmp_index = np.array(_tmp[list(param_set)[0]]).argsort()

        for param in list(param_set):
            _tmp_param[param] = list(np.array(_tmp[param])[_tmp_index])

        return _tmp_param, _tmp_index

    def init_param_empirical(self, T, intensity):
        self.pi = np.random.rand(self.K_all)
        self.pi = self.pi/self.pi.sum()
        self.P0 = self.pi * self.P0_tot
        [self.model[k].set_param_empirical(T, intensity) for k in range(self.K)]
        return

    def add_hist_model(self, info, hist_model, trial):
        info.update({'P0_tot_hist': np.array([hist_model[i]['P0_tot'] for i in range(trial)]),
                     'Ea_hist': np.array([hist_model[i]['Ea'] for i in range(trial)]),
                     'tau0_hist': np.array([hist_model[i]['tau0'] for i in range(trial)]),
                     'Tp_hist': np.array([hist_model[i]['Tp'] for i in range(trial)])})
        return

    def print_param_summary(self, param):
        print('   Ea:       ' + ('{:5.3f} eV        ' * len(param['Ea'])).format(*param['Ea']))
        print('   tau0:     ' + ('{:6.3e} s     ' * len(param['tau0'])).format(*param['tau0']))
        print('   Tp:       ' + ('{:6.3e} K     ' * len(param['Tp'])).format(*param['Tp']))
        print('   P0_tot:   {:6.3e} '.format(self.P0_tot))
        print('   P0:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi'] * self.P0_tot))
        print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
        return

    def leastsq(self, T, intensity, stdout):
        print("Starting TSDC fitting via least square method.")
        print("Fitting parameters are Ea, Tp, pi, and P0_tot.")
        self._gamma =np.array([[T] for k in range(self.K_all)])

        def TSDC(T, param):
            dict_param = {"K":self.K, "Tp": list(param[0:self.K]),
                    "Ea": list(param[self.K:2*self.K]),
                    'P0': list(param[2*self.K:3*self.K+self.K_all])
                    }
            if self.background is 'linear':
                dict_param.update({'s_tri': param[-1]})
            self.set_param(**dict_param)
            return self.predict(T) * self.P0_tot * self.beta

        def residual(param, x, y):
            return y - TSDC(x, param)

        init_param = np.zeros(2*self.K + self.K_all)
        init_param[0:self.K] = [self.model[k].Tp for k in range(self.K)]
        init_param[self.K:2*self.K] = [self.model[k].Ea for k in range(self.K)]
        init_param[2*self.K:2*self.K + self.K_all] = self.pi * integrate.trapz(intensity, T)/self.beta

        start = time.time()
        lb = [self.T_min for i in range(self.K)]
        ub = [self.T_max for i in range(self.K)]
        [lb.append(self.Ea_min) for i in range(self.K)]
        [ub.append(self.Ea_max) for i in range(self.K)]
        [lb.append(0.0) for i in range(self.K_all)]
        [ub.append(np.inf) for i in range(self.K_all)]

        if self.background is 'linear':
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
            opt_param = init_param
            rmse = np.nan
        except RuntimeError:
            print("RuntimeError: Least square method is failed.")
            opt_param = init_param
            rmse = np.nan

        param_dict = {"K":self.K, "Tp": list(opt_param[0:self.K]),
                      "Ea": list(opt_param[self.K:2*self.K]),
                      'P0': list(opt_param[2*self.K:2*self.K + self.K_all])}

        hessian = np.dot(ls['jac'].T, ls['jac'])
        det = np.linalg.det(hessian)
        dim = init_param.size
        print(det)
        print("Estimated log Marginal likelihood via Laplace method:",
               - T.size/2 * np.log(2*np.pi* ls['cost']*2/T.size) - T.size/2
               + dim/2.0 * np.log(2*np.pi) - 1/2.0 * np.log(det)
              )
        print("BIC: ", T.size * np.log(ls['cost']*2/T.size) + dim * np.log(T.size))

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
        if ls['success'] is True:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(rmse, t_tot))
            print('Estimated model parameters and scores are following:')
            print('   Ea:       ' + ('{:5.3f} eV        ' * len(param['Ea'])).format(*param['Ea']))
            print('   tau0:     ' + ('{:6.3e} s     ' * len(param['tau0'])).format(*param['tau0']))
            print('   P0_tot:   {:6.3e} '.format(self.P0_tot))
            print('   P0:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*np.array(param['pi']) * self.P0_tot))
            print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
            print('   RMSE:     {:12.8e}\n'.format(rmse))
        else:
            print("   Warning: non-linear least-square optimization is failed.")

        return run_info

    def leastsq_tau0(self, T, intensity, stdout):
        print("Starting TSDC fitting via least square method.")
        print("Fitting parameters are Ea, tau0, pi, and P0_tot.")

        def TSDC(T, param):
            dict = {"K": self.K, "tau0": list(param[0:self.K]),
                    "Ea": list(param[self.K:2 * self.K]),
                    'P0': list(param[2 * self.K:3 * self.K + self.K_all])
                    }
            self.set_param(**dict)
            return self.predict(T) * self.P0_tot * self.beta

        def residual(param, x, y):
            return y - TSDC(x, param)

        init_param = np.zeros(2 * self.K + self.K_all)
        init_param[0:self.K] = [self.model[k].tau0 for k in range(self.K)]
        init_param[self.K:2 * self.K] = [self.model[k].Ea for k in range(self.K)]
        init_param[2 * self.K:2 * self.K + self.K_all] = self.pi * integrate.trapz(intensity, T) / self.beta

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
                      'P0': list(opt_param[2*self.K:2*self.K + self.K_all])}
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
        if rmse is not np.nan:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(rmse, t_tot))
            print('Estimated model parameters and scores are following:')
            print('   Ea:       ' + ('{:5.3f} eV        ' * len(param['Ea'])).format(*param['Ea']))
            print('   tau0:     ' + ('{:6.3e} s     ' * len(param['tau0'])).format(*param['tau0']))
            print('   P0_tot:   {:6.3e} '.format(self.P0_tot))
            print('   P0:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*np.array(param['pi']) * self.P0_tot))
            print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
            print('   RMSE:     {:12.8e}\n'.format(rmse))

        else:
            print("   Warning: non-linear least-square optimization is failed.")

        return run_info

    def l2_div(self, T, intensity, stdout):
        print("Starting TSDC fitting via least square method.")
        print("L2 divergence based estimation.")
        print("Fitting parameters are Ea, Tp, pi, and P0_tot.")

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
                    'P0': list(param[2*self.K:3*self.K+self.K_all])
                    }
            if self.background is 'linear':
                dict_param.update({'s_tri': param[-1]})
            self.set_param(**dict_param)
            return self.predict(T) * Z

        def residual(param, x, y):
            return y - TSDC(x, param)

        init_param = np.zeros(2*self.K + self.K_all)
        init_param[0:self.K] = [self.model[k].Tp for k in range(self.K)]
        init_param[self.K:2*self.K] = [self.model[k].Ea for k in range(self.K)]
        init_param[2*self.K:2*self.K + self.K_all] = self.pi * integrate.trapz(intensity, T)/self.beta

        start = time.time()
        lb = [self.T_min for i in range(self.K)]
        ub = [self.T_max for i in range(self.K)]
        [lb.append(self.Ea_min) for i in range(self.K)]
        [ub.append(self.Ea_max) for i in range(self.K)]
        [lb.append(0.0) for i in range(self.K_all)]
        [ub.append(np.inf) for i in range(self.K_all)]

        if self.background is 'linear':
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

        P0 = np.array(opt_param[2 * self.K:2 * self.K + self.K_all])
        P0 = P0/(P0.sum()*self.beta) * Z
        param_dict = {"K":self.K, "Tp": list(opt_param[0:self.K]),
                      "Ea": list(opt_param[self.K:2*self.K]),
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
        if ls['success'] is True:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(rmse, t_tot))
            print('Estimated model parameters and scores are following:')
            print('   Ea:       ' + ('{:5.3f} eV        ' * len(param['Ea'])).format(*param['Ea']))
            print('   tau0:     ' + ('{:6.3e} s     ' * len(param['tau0'])).format(*param['tau0']))
            print('   P0_tot:   {:6.3e} '.format(self.P0_tot))
            print('   P0:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*np.array(param['pi']) * self.P0_tot))
            print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
            print('   RMSE:     {:12.8e}\n'.format(rmse))
        else:
            print("   Warning: non-linear least-square optimization is failed.")

        return run_info

    def regularization(self, T, intensity):
        print("Starting TSDC fitting via regularized least square method.")
        print("L2 divergence based estimation.")
        print("Fitting parameters are Ea, Tp, pi, and P0_tot.")

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
                    'P0': list(param[2*self.K:3*self.K+self.K_all])
                    }
            if self.background is 'linear':
                dict_param.update({'s_tri': param[-1]})
            self.set_param(**dict_param)
            return self.predict(T) * Z

        def loss(param, x, y):
            return np.average((y - TSDC(x, param))**2) + 1000 * (param[0] - 600) ** 2 + 1000 * (param[1] - 700) ** 2

        init_param = np.zeros(2*self.K + self.K_all)
        init_param[0:self.K] = [self.model[k].Tp for k in range(self.K)]
        init_param[self.K:2*self.K] = [self.model[k].Ea for k in range(self.K)]
        init_param[2*self.K:2*self.K + self.K_all] = self.pi * integrate.trapz(intensity, T)/self.beta

        start = time.time()
        lb = [self.T_min for i in range(self.K)]
        ub = [self.T_max for i in range(self.K)]
        [lb.append(self.Ea_min) for i in range(self.K)]
        [ub.append(self.Ea_max) for i in range(self.K)]
        [lb.append(0.0) for i in range(self.K_all)]
        [ub.append(np.inf) for i in range(self.K_all)]

        if self.background is 'linear':
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

        ls = optimize.minimize(loss, init_param,
                               args=(T_bin, freq),
                               bounds=bnds,
                               method='L-BFGS-B'
                               )
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

        P0 = np.array(opt_param[2 * self.K:2 * self.K + self.K_all])
        P0 = P0/(P0.sum()*self.beta) * Z
        param_dict = {"K":self.K, "Tp": list(opt_param[0:self.K]),
                      "Ea": list(opt_param[self.K:2*self.K]),
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
        if ls['success'] is True:
            print("   non-linear least-square optimization is successfully finished.")
            print("            RMSE:      {:12.6e}\n"
                  "    Elapsed time:      {:12.6e} s\n".format(rmse, t_tot))
            print('Estimated model parameters and scores are following:')
            print('   Ea:       ' + ('{:5.3f} eV        ' * len(param['Ea'])).format(*param['Ea']))
            print('   tau0:     ' + ('{:6.3e} s     ' * len(param['tau0'])).format(*param['tau0']))
            print('   Tp:       ' + ('{:6.3e} s     ' * len(param['Tp'])).format(*param['Tp']))
            print('   P0_tot:   {:6.3e} '.format(self.P0_tot))
            print('   P0:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*np.array(param['pi']) * self.P0_tot))
            print('   pi:       ' + ('{:6.3e}       ' * len(param['pi'])).format(*param['pi']))
            print('   RMSE:     {:12.8e}\n'.format(rmse))
        else:
            print("   Warning: non-linear least-square optimization is failed.")

        return run_info

