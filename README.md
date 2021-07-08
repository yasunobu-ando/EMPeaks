EMPeaks
========================

This package is for high-throughput peak analysis by using Spectrum Adapted EM algorithm.
Please refer the following paper when using this package:
[Sci. Tech. Adv. Mater. 20, 733-735 (2019).](https://www.tandfonline.com/doi/full/10.1080/14686996.2019.1620123)

## version 1.0.0
In version 1.0.0, Gaussian　Mixture　Model (GMM) is only available.

### Brief Explanation
**class: GaussianMixture(data2d, K=2):**

    class for GMM object.
    parameters:
    _______________________________________
    numpy_array data2d: [energy, intensity]
    integer          K: number of Gaussians
    _______________________________________

**class SpectrumAdaptedEM(GaussianMixture)(data2d, K=2, max_iter=500):**
    
    Class for Spectrum Adapted EM algorithm opject.
    parameters:
    ______________________________________________________
    integer max_iter: max iteration for E-step and M-step.
    ______________________________________________________

    def fit(iter_log=False, sampling=1):
        fitting GMM to the data2d via EM algorithm.
        parameters:
            boolean iter_log: switch to write the iteration log.
            integer sampling: sampling number for initial parameters. 
                              The largest likelihood　model in samples is selected.

    def plot_fitting_summary(self, dpi=100, save=False):
    
    def plot_param_history(self, dpi=100, save=False):
        
    def ani_gmm_history(self, dx=0.05, dpi=100, save=False, interval=100, repeat_delay=1500):

## Examples
We prepared three examples to test this package.
1. `test_em(N: int)`
```python 
    from EMPeaks import GaussianMixture
    test = GaussianMixture.Tests
    test.test_em(N=10000,sampling=5)
```
2. `test_spectrum_adapted_em(N: int)`
```python
from EMPeaks import GaussianMixture
test = GaussianMixture.Tests
test.test_spectrum_adapted_em(N=10000,sampling=5)
```
3. `test_exp_data()`
```python
from EMPeaks import GaussianMixture
test = GaussianMixture.Tests
test.test_exp_data('GFET0126_25V_C1s.txt', sampling=5)
```
---------------
&copy; 2020-2021 National Institute of Advanced Industrial Science and Technology (AIST)