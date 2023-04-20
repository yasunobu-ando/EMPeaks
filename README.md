EMPeaks
========================

This package is for high-throughput peak analysis by using Spectrum Adapted EM algorithm.
Please refer the following paper when using this package:
[Sci. Tech. Adv. Mater. 20, 733-735 (2019).](https://www.tandfonline.com/doi/full/10.1080/14686996.2019.1620123);
[Sci. Tech. Adv. Mater. method 1, 45 (2021).](https://www.tandfonline.com/doi/abs/10.1080/27660400.2021.1899449)

## version 2.1.x (Background Subtraction)
In version 2.1.x, Automated background subtraction is implemented. Following background models are available now.
* uniform: uniform background model
* linear: linear background model (positive gradient)
* ramp_sum: Ramp-Sum background model

The details of these models are explained in this paper,
[Sci. Tech. Adv. Mater. Method 3,  2159753 (2023).](https://www.tandfonline.com/doi/abs/10.1080/27660400.2022.2159753)

Background models are easily implemented into model instance as follows;
```python
from EMPeaks import GaussianMixture
gmm = GaussianMixture.GaussianMixtureModel(K=3, background='uniform')
```
Keywords to set the background models are ```uniform```, ```linear```, ```ramp_sum```.

## version 2.0.x
In version 2.0.x, Gaussian Mixture Model (GMM), Lorentzian Mixture Model (LMM), 
Pseudo Voigt Mixture model (PVMM), and Doniach-Sunijic Miture model (DSMM).
In principle, these combination models are also available but not implemented yet.

From this version, each model has the same functions but differ from version 1, 
though functions and classes in version 1 still work. Sample codes to import 
these models are followings for instance:
```python
from EMPeaks import GaussianMixture
gmm = GaussianMixture.GaussianMixtureModel(K=3)
```
```python
from EMPeaks import LorentzianMixture
lmm = LorentzianMixture.LorentzianMixtureModel(K=2)
```
Mixture model object includes a single model object.
These packages also have a class 
for single Gaussian, Lorentizan, pseudo Voigt, and DS models.
For example,
```python
from EMPeaks import GaussianMixture
gm = GaussianMixture.Gaussian(x_min=-100, x_max=100, sigma_min=0.1, sigma_max=10)
```

In version 2, we do not implement the class for optimization. 
Instead,all model classes has functions to optimize the parameters 
to fit the target data.
```python
from EMPeaks import GaussianMixture
import numpy as np

x = np.load("energy.dat")
y = np.load("intensity.dat")

gmm = GaussianMixture.GaussianMixtureModel(K=3)
gmm.fit(x, y)
# if you want to sample some initial guess and choose the highest likelihood model,
gmm.sampling(x, y, trial=10)
```
After fitting, you can plot both raw data and fitted model as follows:
```python
gmm.plot(x, y)
```

---------------
&copy; 2020-2021 National Institute of Advanced Industrial Science and Technology (AIST)