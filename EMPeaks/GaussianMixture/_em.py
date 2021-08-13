# License: BSD-3-clause
# Copyright © 2020 National Institute of Advanced Industrial Science and Technology (AIST)

from EMPeaks.GaussianMixture._gmm import GaussianMixtureModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class EM(GaussianMixtureModel):
    def __init__(self, data1d, K=2):
        super().__init__(data1d, K)
        self.eps = 1.e-3
        self.max_iter = 500
        self.LL = []
        self.residual = []

    def fit(self, iter_log=False, sampling=1):
        LL_sample = []
        param_sample = []
        for i_sample in range(sampling):
            print('\n**** Sampling: {} ****'.format(i_sample))
            self.init_params()
            self.optimization(iter_log=iter_log)
            LL_sample.append(self.LL[-1])
            param_sample.append(self.param)

        best_index = np.nanargmax(LL_sample)
        self.param = param_sample[best_index]
        self.gmm_mu_ = [self.param[k]['mu'] for k in range(self.gmm_K_)]
        self.gmm_sigma_ = [self.param[k]['sigma'] for k in range(self.gmm_K_)]
        self.gmm_pi_ = [self.param[k]['pi'] for k in range(self.gmm_K_)]

        print('\n')
        print('*** Finishing {} sampling ***'.format(sampling))
        print('maximum-likelihood model is sampled with index {}'.format(best_index))
        print('log-likelihood: {}\n'.format(LL_sample[best_index]))
        print('Model parameters')
        print("mu(means):          {}".format(self.gmm_mu_))
        print("sigma^2(variance):  {}".format(self.gmm_sigma_))
        print("pi(mixing ratio):   {}".format(self.gmm_pi_))

    def optimization(self, iter_log=False):
        print("Start Fitting by conventional EM algorithm.")
        LL1 = self._LL_gmm()
        for i in range(self.max_iter):
            LL2 = LL1
            # E-step
            gamma = self._e_step()
            # M-step
            self._m_step(gamma)
            LL1 = self._LL_gmm()
            self.LL.append(LL1)
            self.residual.append(LL1 - LL2)

            if iter_log:
                print("iter. ", i, ":  log-likelihood: ", LL1, "residual: ", np.abs(LL2 - LL1))

            if np.abs(LL2 - LL1) < self.eps:
                print("\nLog Likelihood is converged with iteration ", i)
                print("Log Likelihood: ", self.LL[-1])
                print("\n-- Converged parameters --")
                sort_index = np.argsort(self.gmm_mu_)
                self.param = [{'mu': self.gmm_mu_[sort_index[k]],
                               'sigma': self.gmm_sigma_[sort_index[k]],
                               'pi': self.gmm_pi_[sort_index[k]]} for k in range(self.gmm_K_)]

                print("mu(means):          {}".format(self.gmm_mu_[sort_index]))
                print("sigma^2(variance):  {}".format(self.gmm_sigma_[sort_index]))
                print("pi(mixing ratio):   {}".format(self.gmm_pi_[sort_index]))
                break

        sort_index = np.argsort(self.gmm_mu_)
        self.param = [{'mu': self.gmm_mu_[sort_index[k]],
                       'sigma': self.gmm_sigma_[sort_index[k]],
                       'pi': self.gmm_pi_[sort_index[k]]} for k in range(self.gmm_K_)]

    def _LL_gmm(self):
        L = np.zeros(self.dat.size)
        for k in range(self.gmm_K_):
            L += self.gmm_pi_[k] * norm.pdf(self.dat, self.gmm_mu_[k], np.sqrt(self.gmm_sigma_[k]))

        return np.sum(np.log(L))

    def _e_step(self):
        x = self.dat
        marginal = np.zeros(x.size)

        gamma = np.zeros([x.size, self.gmm_K_])
        for k in range(self.gmm_K_):
            gamma[:, k] = self.gmm_pi_[k] * norm.pdf(x, self.gmm_mu_[k], np.sqrt(self.gmm_sigma_[k]))
            marginal += gamma[:, k]

        for k in range(self.gmm_K_):
            gamma[:, k] = gamma[:, k]/marginal

        return gamma

    def _m_step(self, gamma):
        for k in range(self.gmm_K_):
            n_k = gamma[:, k].sum()
            self.gmm_mu_[k] = (gamma[:, k] * self.dat).sum() / n_k
            self.gmm_sigma_[k] = np.sum(gamma[:, k] * np.square(self.dat - self.gmm_mu_[k])) / n_k
            self.gmm_pi_[k] = n_k / self.dat.size

    def plot_fitting_summary(self, dpi=100, save=False):
        fig = plt.figure(figsize=(9, 4), dpi=dpi)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0.5)
        ax1 = plt.subplot2grid((2, 3), (0, 0), fig=fig)
        ax2 = plt.subplot2grid((2, 3), (1, 0), fig=fig)
        ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2, fig=fig)

        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
        ax1.grid(axis='y', which='major')
        ax1.set_ylabel('log-likelihood')
        ax1.plot(self.LL)

        ax2.semilogy(self.residual)
        ax2.set_ylabel('residual')
        ax2.set_xlabel('iteration')
        ax2.grid(axis='y', which='major')
        ax2.plot(self.eps)
        ax2.plot([0, len(self.residual)], [self.eps, self.eps], "red", linestyle='dashed')

        ax3 = self.plot_gmm()

        if save:
            plt.savefig(fname='plt_fitting_summary.png')

    def plot_gmm(self, dx=0.05):
        X = np.arange(0, 10, dx)
        Y = np.zeros([self.gmm_K_, X.size])
        plt.ylabel('Intensity')
        plt.xlabel('Energy')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
        bins=100
        plt.hist(self.dat, bins=100, range=(0,10))

        for k in range(self.gmm_K_):
            Y[k, :] = self.gmm_pi_[k] * norm.pdf(X, self.gmm_mu_[k], np.sqrt(self.gmm_sigma_[k]))

        weights, bins = np.histogram(self.dat, bins=bins, range=(0, 10))
        data2D = np.array([bins[:-1], weights])
        #plt.scatter(data2D[0], data2D[1], c='gray', marker='o')

        dx_dat = data2D[0, 1] - data2D[0, 0]
        Z = (data2D[1].sum() * dx_dat) / (Y.sum() * dx)
        for k in range(self.gmm_K_):
            plt.plot(X, Z*Y[k, :], linewidth=2, linestyle='dashed', color='navy')

        return plt.plot(X, Z * Y.sum(axis=0), linewidth=3, color='navy')
