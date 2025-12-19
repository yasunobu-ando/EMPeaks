import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import stats
from scipy.special import expi, expn
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

kB=8.61733034e-5
inv_kB = 1.0/kB


def f(T, Ea):
    return np.exp(-Ea / (kB * T))


def g(T, Ea):
    return inv_kB / T * np.exp(-Ea * inv_kB / T)


class TSDC:
    def __init__(self, beta=0.0833, T_min=300, T_max=1000, Ea_min=0.1, Ea_max=3.0):
        self.beta = beta
        self.T_min = T_min
        self.T_max = T_max
        self.Ea_min = Ea_min
        self.Ea_max = Ea_max
        self.Tp = np.random.uniform(T_min, T_max)
        self.Ea = np.random.uniform(Ea_min, Ea_max)
        self.tau0 = self.get_tau0()
        self.P0 = 1.0
        self.dT = 1.0
        self.dE = 0.1
        self.is_not_converged = [0.0, 0.0, False] # detecting initial condition not converged.
        self.fix_Tp = False
        self.fix_Ea = False

    def set_param(self, **param):
        """
        Tp_min, Tp_maxの大小関係についてのチェックは未実装
        Ea_min, Ea_maxの大小関係についてのチェックは未実装
        """
        if not param:
            return self
        prohibited_set = {'Tp', 'tau0'}
        param_keys = set(param.keys())
        instance_keys = set(self.__dict__.keys())
        if (instance_keys & param_keys) >= prohibited_set:
            #print('Parameter Tp and tau0 cannot set simultaneously. Tp is set and tau0 is generated.')
            for key in (instance_keys & param_keys):
                self.__setattr__(key, param[key])
            self.tau0 = self.get_tau0()
            return
        if (instance_keys & param_keys) >= {'Tp'}:
            for key in (instance_keys & param_keys):
                self.__setattr__(key, param[key])
            self.tau0 = self.get_tau0()
            return
        if (instance_keys & param_keys) >= {'tau0'}:
            for key in (instance_keys & param_keys):
                self.__setattr__(key, param[key])
            self.Tp = self.get_Tp()
            return

    def init_model(self):
        if not self.fix_Tp:
            self.Tp = np.random.uniform(self.T_min, self.T_max)
        if not self.fix_Ea:    
            self.Ea = np.random.uniform(self.Ea_min, self.Ea_max)
        self.tau0 = self.get_tau0()
        return

    def set_param_empirical(self, T, intensity):
        Tp_mu = np.sum(intensity * T) / np.sum(intensity)
        Tp_sigma = np.sqrt(np.sum(intensity * (T ** 2))
                           / np.sum(intensity) - Tp_mu ** 2)

        self.Tp = np.random.normal(Tp_mu, Tp_sigma*0.5, 1)
        self.Ea = np.random.uniform(self.Ea_min, self.Ea_max)
        self.tau0 = self.get_tau0()[0]
        return

    def get_tau0(self):
        return (kB * self.Tp ** 2) / (self.beta * self.Ea) * np.exp(-self.Ea / (kB * self.Tp))

    def get_Tp(self):
        def f(Tp, Ea, tau0):
            return np.log((kB * Tp ** 2) / (self.beta * Ea)) - Ea / (kB * Tp) - np.log(tau0)
        Tp = optimize.newton(f, 100, args=(self.Ea, self.tau0))
        return Tp

    def predict(self, T):
        prob = 1.0 / (self.beta * self.tau0) * \
                       np.exp(- self.Ea / kB / T
                              - 1 / self.beta / self.tau0
                              * np.array(T * expn(2, self.Ea*inv_kB / T)))
        return prob / (integrate.trapezoid(prob, T) + 1.0e-100)

    def maximum_likelihood_estimation(self, T, intensity):
        # for sparse modeling with Dirichlet prior with alpha less than 0.
        if np.sum(intensity) == 0:
            return
        try:
            #self.direct_minimization(T, intensity)
            self.find_root(T, intensity)
            return

        except RuntimeError:
            print("RuntimeError: brentq failed. Switch to direct maximization of LL")
            try:
                self.gradient_minimization(T, intensity)
                #print("direct minimization of LL finished.")
                return

            except RuntimeError:
                print("RuntimeError_1: Log likelihood maximization failed. initialized Again.")
                self.is_not_converged[2] = True
                self.set_param_empirical(T, intensity)
                return

        except ValueError:
            print("ValueError: brentq failed. Switch to direct maximization of LL")
            # self.is_not_converged[2] = True
            try:
                self.gradient_minimization(T, intensity)
                #print("direct minimization of LL finished.")
                return

            except RuntimeError:
                print("RuntimeError_2: Log likelihood maximization failed. initialized Again.")
                self.is_not_converged[2] = True
                self.set_param_empirical(T, intensity)
                return

    def log_likelihood(self, T, intensity):
        return np.sum(intensity * np.log(self.predict(T) + 1e-200))

    def posterior(self, T, intensity):
        self.dT = 1.00
        self.dE = 0.01
        Tp = np.arange(self.T_min, self.T_max, self.dT)
        Ea = np.arange(self.Ea_min, self.Ea_max, self.dE)

        print(Tp.size*Ea.size)
        posterior = np.zeros([Tp.size, Ea.size])
        log_posterior = - np.ones([Tp.size, Ea.size]) * np.infty
        # prior = np.ones(T.size, )

        map_param = np.array([self.Tp, self.Ea])
        index = self.cutoff_index(Tp, Ea, map_param, 10, 0.05)
        print(index.size)
        _param_dict = self.__dict__
        template = TSDC()
        template.set_param(**_param_dict)

        for i in range(index.shape[0]):
            template.Tp = Tp[index[i, 0]]
            template.Ea = Ea[index[i, 1]]
            template.tau0 = template.get_tau0()
            tmp = template.log_likelihood(T, intensity)
            log_posterior[index[i, 0], index[i, 1]] = tmp

        C = np.max(log_posterior)
        Z = np.sum(np.exp(log_posterior - C)) * self.dE * self.dT
        posterior = np.exp(log_posterior - C) / Z

        return posterior, log_posterior

    def cutoff_index(self, Tp, Ea, map_param, Tp_cutoff=10, Ea_cutoff=0.1, flag=0):
        sigma_Tp = (Tp_cutoff / 3.0) ** 2
        sigma_Ea = (Ea_cutoff / 3.0) ** 2
        cov = np.array([sigma_Tp, sigma_Ea])

        xv, yv = np.meshgrid(Tp, Ea)
        grids = np.array([xv, yv]).T
        z = multivariate_normal.pdf(grids, mean=map_param, cov=cov)
        threshold = multivariate_normal.pdf(map_param + np.array([Tp_cutoff, 0]),
                                            mean=map_param, cov=[sigma_Tp, sigma_Ea]) * (flag != 0)

        index = np.array(np.where(z >= threshold)).T
        return index

    def plot(self, T, label=""):
        plt.plot(T, self.beta * self.P0 * self.predict(T), label=label)
        return

    def leastsq_tau0(self, T, intensity):
        print("Starting TSDC fitting via least square method.")
        print("Fitting parameters are Ea, tau0, pi, and P0_tot.")

        def TSDC(T, param):
            self.tau0 = param[0]
            self.Ea = param[1]
            self.P0 = param[2]
            return self.predict(T) * self.P0 * self.beta

        def residual(param, x, y):
            return y - TSDC(x, param)

        self.set_param_empirical(T, intensity)
        init_param = np.zeros(3)
        init_param[0] = self.tau0
        init_param[1] = self.Ea
        init_param[2] = integrate.trapezoid(intensity, T) / self.beta

        start = time.time()
        lb = [0]
        ub = [1.0e+1]
        lb.append(self.Ea_min)
        ub.append(self.Ea_max)
        lb.append(0.0)
        ub.append(np.inf)

        start = time.time()
        bnds = (lb, ub)
        try:
            ls = optimize.least_squares(residual, init_param,
                                        args=(T, intensity),
                                        bounds=bnds,
                                        loss='linear'
                                        )
            print(ls['x'])
            opt_param = ls['x']
            self.Tp = opt_param[0]
            self.Ea = opt_param[1]
            self.P0 = opt_param[2]
            rmse = np.sqrt((ls['cost']) * 2.0 / T.size)

        except ValueError:
            print("Least square method is failed.")
            opt_param = init_param
            rmse = np.nan
        except RuntimeError:
            print("Least square method is failed.")
            opt_param = init_param
            rmse = np.nan

        t_tot = time.time() - start
        print('elapse time: {:} '.format(t_tot))

        return opt_param, rmse

    def leastsq(self, T, intensity):
        print("Starting TSDC fitting via least square method.")
        print("L2 divergence based estimation.")
        print("Fitting parameters are Ea, Tp, pi, and P0_tot.")

        def TSDC(T, param):
            self.Tp = param[0]
            self.Ea = param[1]
            self.tau0 = self.get_tau0()
            self.P0 = param[2]
            return self.predict(T) * self.P0 * self.beta

        def residual(param, x, y):
            return y - TSDC(x, param)

        self.set_param_empirical(T, intensity)
        init_param = np.zeros(3)
        init_param[0] = self.Tp
        init_param[1] = self.Ea
        init_param[2] = integrate.trapezoid(intensity, T) / self.beta

        lb = [self.T_min]
        ub = [self.T_max]
        lb.append(self.Ea_min)
        ub.append(self.Ea_max)
        lb.append(0.0)
        ub.append(np.inf)

        start = time.time()
        bnds = (lb, ub)
        try:
            ls = optimize.least_squares(residual, init_param,
                                        args=(T, intensity),
                                        bounds=bnds,
                                        loss='linear'
                                        )
            opt_param = ls['x']
            self.Tp = opt_param[0]
            self.Ea = opt_param[1]
            self.P0 = opt_param[2]
            rmse = np.sqrt((ls['cost']) * 2.0 / T.size)

        except ValueError:
            print("ValueError: Least square method is failed.")
            opt_param = init_param
            rmse = np.nan

        except RuntimeError:
            print("RuntimeError: Least square method is failed.")
            opt_param = init_param
            rmse = np.nan

        t_tot = time.time() - start
        print('elapse time: {:} '.format(t_tot))

        return opt_param, rmse

    def gradient_minimization(self, T, intensity):
        d1 = np.sum(intensity)
        d2 = np.sum(intensity/T * inv_kB)

        def log_likelihood(param):
            self.Ea = param[0]
            self.Tp = param[1]
            self.tau0 = self.get_tau0()
            return - self.log_likelihood(T, intensity)

        def derivative(param):
            template = TSDC()
            template.Ea = param[0]
            template.Tp = param[1]
            template.tau0 = template.get_tau0()

            beta_tau = template.beta * template.tau0
            ktp = kB * template.Tp

            d3 = np.sum(intensity * np.array(T * expn(2, template.Ea * inv_kB / T)))
            d3_p = - np.sum(intensity * inv_kB * np.array(expn(1, template.Ea * inv_kB / T)))

            # LLの Ea微分
            LL_e = (d1 - d3 / beta_tau) * (1.0 / template.Ea + 1.0 / ktp) - d2 - d3_p / beta_tau
            # LLの Tp微分
            LL_t = (-d1 + d3 / beta_tau) * (2.0 / template.Tp + template.Ea * inv_kB / template.Tp ** 2)

            return np.array([-LL_e, -LL_t])

        init = np.zeros(2)
        init[0] = self.Ea
        init[1] = self.Tp
        info = optimize.minimize(log_likelihood, x0=init, jac=derivative,
                                 bounds=[(self.Ea_min, self.Ea_max), (self.T_min, self.T_max)],
                                 method='L-BFGS-B')
        #print(info)
        self.Ea = info['x'][0]
        self.Tp = info['x'][1]
        self.tau0 = self.get_tau0()
        self.P0 = integrate.trapezoid(intensity, T) / self.beta
        return

    def find_root(self, T, intensity):
        def f_sum(E, T, Y):
            return np.sum(Y * np.array(T * expn(2, E * inv_kB / T)))

        def f_g_diff(E, T, Y, beta):
            eps = 1e-10
            a = np.sum(Y / (kB * T)) + eps
            b = np.sum(Y) + eps
            g_sum = np.sum(Y * inv_kB * np.array(- expi(-E * inv_kB / T)))
            return np.log10(f_sum(E, T, Y)) - np.log10(beta * b) - np.log10(g_sum) + np.log10(beta * a)

        self.Ea = optimize.brentq(f_g_diff, self.Ea_min, self.Ea_max, args=(T, intensity, self.beta))
        self.tau0 = f_sum(self.Ea, T, intensity) / (self.beta * np.sum(intensity))
        self.Tp = self.get_Tp()
        self.P0 = integrate.trapezoid(intensity, T) / self.beta

        return

    def map_estimation(self, T, intensity):
        def log_likelihood(param):
            self.Ea = param[0]
            self.Tp = param[1]
            self.tau0 = self.get_tau0()
            return - self.log_likelihood(T, intensity) - np.log(stats.norm.pdf(param[1], 860, 1.0e-3) + 1.0e-200)\
                   - np.log(stats.norm.pdf(param[0], 1.0, 0.25) + 1.0e-200)

        init = np.zeros(2)
        init[0] = self.Ea
        init[1] = 860 #self.Tp
        info = optimize.minimize(log_likelihood, x0=init,
                                 bounds=[(self.Ea_min, self.Ea_max), (self.T_min, self.T_max)],
                                 method='L-BFGS-B')
        #print(info)
        self.Ea = info['x'][0]
        self.Tp = info['x'][1]
        self.tau0 = self.get_tau0()
        self.P0 = integrate.trapezoid(intensity, T) / self.beta
        return

    def fisher_matrix(self, T, intensity):
        d1 = np.sum(intensity)
        d2 = np.sum(intensity/T * inv_kB)
        d3 = np.sum(intensity * np.array(T * expn(2, self.Ea * inv_kB / T)))
        d3_p = - np.sum(intensity * inv_kB * np.array(expn(1, self.Ea * inv_kB / T)))
        d3_pp = inv_kB/self.Ea * np.sum(intensity * np.exp(- self.Ea * inv_kB / T))

        bt = self.beta * self.tau0
        ktp = kB * self.Tp

        LL_t = (-d1 + d3 / bt) * (2.0 / self.Tp + self.Ea * inv_kB / self.Tp ** 2)

        LL_tt = - d3/bt * (2.0/self.Tp + (self.Ea * inv_kB/self.Tp**2))**2 \
                - 2 * (-d1 + d3/bt) * (1.0/self.Tp**2 + self.Ea * inv_kB/self.Tp**3)

        LL_te = (d3_p/bt + d3/bt * (1.0/self.Ea + 1.0/ktp)) * (2.0/self.Tp + self.Ea/(self.Tp * ktp)) \
                + (d1 - d3/bt)/(ktp * self.Tp)

        LL_ee = - d1/self.Ea**2 - d3_pp/bt - 2 * d3_p/bt * (1.0/self.Ea + 1.0/ktp) \
                - d3*inv_kB/(self.beta*self.tau0*self.Tp) * (2.0/self.Ea + inv_kB/self.Tp)

        print("\nHessian of EE: ",LL_ee)
        print("Hessian of tt: ",LL_tt)

        #self.numerical_fisher_matrix(T, intensity)
        #print("Ea, Tp", self.Ea, self.Tp)
        #print('derivative of Tp',LL_t)
        #print(np.array([[LL_tt, LL_te], [LL_te, LL_ee]]))
        return np.array([[LL_tt, LL_te], [LL_te, LL_ee]])

    def numerical_fisher_matrix(self, T, intensity):
        def log_likelihood(param):
            template = TSDC()
            template.Ea = param[0]
            template.Tp = param[1]
            template.tau0 = template.get_tau0()
            return template.log_likelihood(T, intensity)

        dT = 1.0e-2
        dE = 1.0e-5
        b = []
        param = []
        for n in np.arange(-100, 100):
            param.append([self.Ea + n * dE, self.Tp])
            b.append(log_likelihood(param[-1]))

        a=[]
        for i in range(len(b) - 2):
            a.append((b[i + 2] + b[i] - b[i + 1] * 2) / dE ** 2)
        E = np.array(param).T[0]

        #plt.plot(b)
        #plt.plot(np.average(a)/2 * (E - self.Ea)**2 + self.log_likelihood(T, intensity))
        #plt.show()
        #print('average LL_ee: ', np.average(a))

        b = []
        param = []
        for n in np.arange(-300, 300):
            param.append([self.Ea, self.Tp + n * dT])
            b.append(log_likelihood(param[-1]))

        a=[]
        for i in range(len(b) - 2):
            a.append((b[i + 2] + b[i] - b[i + 1] * 2) / dT ** 2)
        TT = np.array(param).T[1]
        plt.plot(TT, np.average(a)/2 * (TT - self.Tp)**2 + self.log_likelihood(T, intensity))
        plt.plot(TT, b)
        #print('average: LL_tt', np.average(a))
        plt.show()

        a=[]
        for i in range(len(b) - 1):
            a.append((b[i + 1] - b[i])/ dT)
        TT = np.array(param).T[1]
        #plt.plot(TT, b)
        #plt.show()
        #plt.plot(TT[0:-1], a)
        #plt.show()

        #b = np.zeros([20, 20])
        #for n in np.arange(-10, 10):
        #    for m in np.arange(-10, 10):
        #        param = [self.Ea + n * dE, self.Tp + m * dT]
        #        b[n, m] = (log_likelihood(param))
        #print(((b[1, 1] - b[0, 1])/dT - (b[1, 0] - b[0, 0])/dT)/dE)

        return