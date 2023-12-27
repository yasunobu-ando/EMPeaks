from .._tsdc_mixture import TSDCMixtureModel
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
kB = 8.61733034e-5

class TestData:
    data_param = {'K': 5,
                  'N': [4.0e+7, 5.0e+7, 10.0e+7, 1.0e+7, 8.0e+6],
                  'Tp': [600, 685, 750, 530, 873],
                  'Ea': [1.2, 1.6, 1.0, 0.74, 2.4],
                  'T_min':  300,
                  'T_max': 1100,
                  }

    def __init__(self, K=3, sigma=0, lambda_current=0, dtime=10):
        if (K > 5) or (K <= 0) :
            print("error: K must be the number between 1 to 5.")
            return
        param = {'K': K,
                 'N': TestData.data_param['N'][0:K],
                 'Tp': TestData.data_param['Tp'][0:K],
                 'Ea': TestData.data_param['Ea'][0:K],
                 'beta': 0.0833,
                 'T_min': TestData.data_param['T_min'],
                 'T_max': TestData.data_param['T_max'],
                 'background': 'none'
                 }
        self.K = K
        self.beta = param['beta']
        self.T_min = param['T_min']
        self.T_max = param['T_max']
        self.background = param['background']

        self.N_tot = sum(param['N'])
        self.model = TSDCMixtureModel(background='none')
        self.model.set_param(**param)
        self.param = param

        self.sigma = sigma
        self.lambda_current = lambda_current
        self.dtime = dtime

        self.T, self.intensity = self.generate_data_with_noise()

    def _temperature_sampling(self):
        time = np.arange(0, (self.T_max - self.T_min) / self.beta, self.dtime)
        temperature = self.T_min + np.random.normal(self.beta * time, self.sigma, time.size)
        return temperature

    def generate_data_with_noise(self):
        temp = self._temperature_sampling()
        current = self.beta * self.N_tot * self.model.predict(temp)
        current += np.random.poisson(self.lambda_current, temp.size)
        return temp, current


class Data:
    """ [class Data]
    Data(f_name=None, beta=0.0833, param=None)
    f_name: name of input csv format file. Data is generated if f_name =None.
    beta: temperature change per second. It is determined in experiment.
    param: type dictionary {'K':positive integer, 'N':list[K], 'tau0':list[K], 'Ea':list[K]}
        K: number of components
        N: initial polarization for each process
        tau0: relaxation times for each process
        Ea: activation energy for each process
    """
    def __init__(self, f_name=None, beta=0.0833, param=None):
        if f_name is None:
            if param is None:
                #print("Test data is generated automatically because no csv file is imported.")
                self.beta = 0.0833
                self.temp0 = 300
                self.temp1 = 850
                self.dT = 0.5
                self.T = np.arange(self.temp0, self.temp1, self.dT)
                self.N = self.T.size

                N = np.array([6.0e+7, 4.0e+7])
                tau0 = np.array([0.8e-6, 500.0e-6])
                Ea = np.array([1.0, 0.6])
                self.intensity = generate_data(self.T, self.N, 2,
                                               N, tau0, Ea, self.beta, self.temp0)
            else:
                print("Data is generated according to setting parameters")
                self.beta = 0.0833
                self.temp0 = 300
                self.temp1 = 850
                self.dT = 1
                self.T = np.arange(self.temp0, self.temp1, self.dT)
                self.N = self.T.size

                N = param['N']
                tau0 = param['tau0']
                Ea = param['Ea']
                self.intensity = generate_data(self.T, self.N, param['K'],
                                               N, tau0, Ea, self.beta, self.temp0)
        else:
            self.beta = beta
            self.read_csv(f_name)
        for i in range(len(self.intensity)):
            if self.intensity[i] < 0.0:
                self.intensity[i] = 0.0
        #print(self.intensity)

    def import_data(self, data):
        self.beta = data.beta
        self.temp0 = data.temp0
        self.temp1 = data.temp1
        self.dT = data.dT
        self.T = data.T
        self.N = data.N

    def export_data(self):
        data = Data()
        data.beta = self.beta
        data.temp0 = self.temp0
        data.temp1 = self.temp1
        data.dT = self.dT
        data.T = self.T
        data.N = self.N

        return data

    def read_csv(self, f_name):
        """
        This method updates data of intensity and temperature from csv file.
        Assuming uniform mesh grid for temperature T.
        """
        T, I = np.loadtxt(f_name, delimiter=',', unpack=True)
        self.intensity = I / self.beta
        self.T = T + 273.15
        self.temp0 = T[0] + 273.15
        self.temp1 = T[-1] + 273.15
        self.N = T.size
        self.dT = T[1] - T[0]

    def test_info(self):
        print("[Information summary of contained data] ")
        print("Mixing number K: ", 2)
        print("Heating ratio beta: ", 0.0833)
        print("Temperature range: ", "from ", 300, "to", 850,"[K]")
        print("Initial Polarization N: ", 6.2, 4.0)
        print("Total Initial Polarization N: ", 10.0)
        print("Exponential pre-factor tau0[us]: ", 6.2, 500)
        print("Activation Energy Ea[eV]: ", 1, 0.6)
        print("Temperature interval dT[K]: ", 1)

    def plot_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.T, self.intensity, label='generated data')
        ax.set_xlabel('Temperature [K]')
        ax.set_ylim(0.0, self.intensity.max()*1.1)
        ax.set_ylabel('Spectrum Intensity/beta')
        ax.legend()
        plt.show()


def generate_data(T, N_size, K, N, tau0, Ea, beta, temp0):
    y = np.zeros(N_size)
    for i in range(N_size):
        for k in range(K):
            y[i] += N[k] / (tau0[k] * 1e-6) * np.exp(-Ea[k] / kB / T[i]
                                                      - 1 / beta / (tau0[k] * 1e-6)
                                                      * integrate.quad(f, temp0, T[i], Ea[k])[0])
    return y/beta

def f(T, E):
    return np.exp(-E / (kB * T))

