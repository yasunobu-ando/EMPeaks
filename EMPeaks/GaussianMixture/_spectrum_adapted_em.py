# License: BSD-3-clause
# Copyright © 2020 National Institute of Advanced Industrial Science and Technology (AIST)

from EMPeaks.GaussianMixture._gmm import GaussianMixtureModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm


class SpectrumAdaptedEM(GaussianMixtureModel):
    def __init__(self, data, K=2, max_iter=500):
        super().__init__(data, K)
        self.eps = 1.e-7
        self.max_iter = max_iter

        self.LL = []
        self.residual = []
        self.mu_hist = np.array([self.gmm_mu_])
        self.sigma_hist = [self.gmm_sigma_]
        self.pi_hist = [self.gmm_pi_]

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
        print("Start Fitting by Spectrum-adapted EM algorithm.")
        LL1 = self._LL_gmm()

        for i in range(self.max_iter):
            LL2 = LL1
            gamma = self._e_step()
            self._m_step(gamma)
            LL1 = self._LL_gmm()

            self.LL.append(LL1)
            self.residual.append(LL1-LL2)
            self.mu_hist = np.append(self.mu_hist, np.array([self.gmm_mu_]), axis=0)
            self.sigma_hist = np.append(self.sigma_hist, np.array([self.gmm_sigma_]), axis=0)
            self.pi_hist = np.append(self.pi_hist, np.array([self.gmm_pi_]), axis=0)

            if iter_log:
                print("iter. {0:4d}:  log-likelihood: {1:.5f},  residual: {2:.3e}".format(i, LL2, np.abs(LL2 - LL1)))

            if np.abs(LL2 - LL1) < self.eps:
                sort_index = np.argsort(self.gmm_mu_)
                self.param = [{'mu': self.gmm_mu_[sort_index[k]],
                               'sigma': self.gmm_sigma_[sort_index[k]],
                               'pi': self.gmm_pi_[sort_index[k]]} for k in range(self.gmm_K_)]

                print("\nLog Likelihood is converged with iteration {}".format(i))
                print("Log Likelihood: ", self.LL[-1])
                print("\n-- Converged parameters --")
                print("mu(means):          {}".format(self.gmm_mu_[sort_index]))
                print("sigma^2(variance):  {}".format(self.gmm_sigma_[sort_index]))
                print("pi(mixing ratio):   {}".format(self.gmm_pi_[sort_index]))
                break

        sort_index = np.argsort(self.gmm_mu_)
        self.param = [{'mu': self.gmm_mu_[sort_index[k]],
                       'sigma': self.gmm_sigma_[sort_index[k]],
                       'pi': self.gmm_pi_[sort_index[k]]} for k in range(self.gmm_K_)]

    def _LL_gmm(self):
        x = self.dat[0]
        w = self.dat[1]

        L = np.zeros(x.size)
        for k in range(self.gmm_K_):
            L += self.gmm_pi_[k] * norm.pdf(x, self.gmm_mu_[k], np.sqrt(self.gmm_sigma_[k]))
        return np.sum(w * np.log(L + 1.0e-100))

    def _e_step(self):
        x = self.dat[0]
        marginal = np.zeros(x.size)

        gamma = np.zeros([x.size, self.gmm_K_])
        for k in range(self.gmm_K_):
            gamma[:, k] = self.gmm_pi_[k] * norm.pdf(x, self.gmm_mu_[k], np.sqrt(self.gmm_sigma_[k]))
            marginal += gamma[:, k]

        for k in range(self.gmm_K_):
            gamma[:, k] = gamma[:, k]/(marginal + 1.0e-200)

        return gamma

    def _m_step(self, gamma):
        x = self.dat[0]
        w = self.dat[1]

        for k in range(self.gmm_K_):
            n_k = (w * gamma[:, k]).sum()
            self.gmm_mu_[k] = (w * gamma[:, k] * x).sum() / n_k
            self.gmm_sigma_[k] = np.sum(w * gamma[:, k] * np.square(x - self.gmm_mu_[k])) / n_k
            self.gmm_pi_[k] = n_k / w.sum()

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

        #return fig

    def plot_param_history(self, dpi=100, save=False):
        fig, ax = plt.subplots(3, 1, dpi=dpi, sharex=True)
        fig.subplots_adjust(hspace=0)
        ax[0].set_xlim(0, len(self.mu_hist) - 1)
        ax[2].set_xlabel('iteration')
        ax[0].set_ylabel('mu')
        ax[1].set_ylabel('sigma^2')
        ax[2].set_ylabel('pi')

        for k in range(self.gmm_K_):
            ax[0].plot(self.mu_hist.T[k], linewidth=5)
            ax[0].plot([0, len(self.mu_hist)], [self.gmm_mu_, self.gmm_mu_], "red", linestyle='dashed')
            ax[1].plot(self.sigma_hist.T[k], linewidth=5)
            ax[1].plot([0, len(self.mu_hist)], [self.gmm_sigma_, self.gmm_sigma_], "red", linestyle='dashed')
            ax[2].plot(self.pi_hist.T[k], linewidth=5)
            ax[2].plot([0, len(self.mu_hist)], [self.gmm_pi_, self.gmm_pi_], "red", linestyle='dashed')

        if save:
            plt.savefig(fname='plt_param_history.png')

        #return fig

    def ani_gmm_history(self, dx=0.05, dpi=100, save=False, interval=100, repeat_delay=1500):
        fig = plt.figure(dpi=dpi)
        ims = []

        plt.scatter(self.dat[0], self.dat[1], c='gray', marker='o')

        X = np.arange(self.dat[0].min(), self.dat[0].max(), dx)
        Y = np.zeros([self.gmm_K_, X.size])

        for it in range(len(self.mu_hist)):
            line = []  # initialize the list of line2D objects
            for k in range(self.gmm_K_):
                Y[k, :] = self.pi_hist[it, k] * norm.pdf(X, self.mu_hist[it, k], np.sqrt(self.sigma_hist[it, k]))

            dx_dat = self.dat[0, 1] - self.dat[0, 0]
            Z = (self.dat[1].sum() * dx_dat) / (Y.sum() * dx)

            for k in range(self.gmm_K_):
                line.extend(plt.plot(X, Z * Y[k, :], linewidth=2, linestyle='dashed', color='navy', animated=True))

            line.extend(plt.plot(X, Z * Y.sum(axis=0), linewidth=3, color='navy', animated=True))

            # Line2D object is list! Then, we do not need to make line object list again.
            ims.append(line)

        ani = animation.ArtistAnimation(fig, ims,
                                        interval=interval, blit=True, repeat=True, repeat_delay=repeat_delay)
        if save:
            ani.save('ani_gmm_history.gif', writer='imagemagick', fps=30)

        return ani

    def plot_gmm(self, dx=0.05):
        X = np.arange(self.dat[0].min(), self.dat[0].max(), dx)
        Y = np.zeros([self.gmm_K_, X.size])
        plt.ylabel('Intensity')
        plt.xlabel('Energy')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
        plt.scatter(self.dat[0], self.dat[1], c='gray', marker='o')

        for k in range(self.gmm_K_):
            Y[k, :] = self.gmm_pi_[k] * norm.pdf(X, self.gmm_mu_[k], np.sqrt(self.gmm_sigma_[k]))

        dx_dat = self.dat[0, 1] - self.dat[0, 0]
        Z = (self.dat[1].sum() * dx_dat) / (Y.sum() * dx)
        for k in range(self.gmm_K_):
            plt.plot(X, Z * Y[k, :], linewidth=2, linestyle='dashed', color='navy')

        return plt.plot(X, Z * Y.sum(axis=0), linewidth=3, color='navy')