# License: BSD-3-clause
# Copyright © 2020-2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Copyright © 2024 University of Science Tokyo
# Author: Yasunobu ANDO

import numpy as np
from scipy import stats, optimize #, integrate


class UniformModel:
    """ Class UniformModel(x0, x1)

    Class for uniform background model

    Attributes
    ----------
    x0: float
        Lower bound of the domain
    x1: float
        Upper bound of the domain

    Methods
    -------
    predict(x):
        Return value of probability density of the uniform model at 'x'.

    Notes
    --------
    .. math::
        p(x)=\frac{1}{x_1 - x_0}

    """
    def __init__(self, x0, x1):
        """ Initialize an instance of UniformModel

        Parameters
        ----------
        x0: float
             Lower bound of the domain
        x1: float
            Upper bound of the domain
        """
        self.x0 = x0
        self.width = x1 - x0

    def predict(self, x):
        """ Return probability distribution function of uniform background model

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]
        Returns
        -------
        float
            Return value of probability density of the uniform model at 'x'.

        Notes
        -----
        .. math::
            p(x)=\frac{1}{x_1 - x_0}
        """
        return stats.uniform.pdf(x, loc=self.x0, scale=self.width)

    def _LL(self, x, weight):
        """ method _LL(x, weight)

        Parameters
        ----------
        x: list of float
            List of data points or data bins in the domain [x0, x1]
        weight: list of float
            List of weights, intensity, or counts at each 'x' in the domain [x0, x1]

        Returns
        -------
        float
            Return logarithmic likelihood of uniform background model
        """
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        """ maximum likelihood estimation of the uniform background model

        Return nothing because the uniform model does not have any parameters.

        Parameters
        ----------
        x: list of float
            List of sampling points or data bins in the domain [x0, x1]
        weight: list of float
            List of weights, intensity, or counts at each 'x' in the domain [x0, x1]

        Returns
        -------
        Nothing
        """
        return


class SquareRootModel:
    """
    Class SquareRootModel(x0, x1)

    Attributes
    ----------
    x0: float
        Lower bound of the domain
    x1: float
        Upper bound of the domain

    Methods
    -------
    predict(x): Return value of probability density of the SquareRoot model at 'x'.

    Notes
    -----
    .. math::
        p(x)=\frac{3.0}{2.0(x_1^{1.5} - x_0^{1.5})} \sqrt{x}

    """
    #x: sampling point or data bins
    #w: weight or intensity of the data
    def __init__(self, x0, x1):
        """
        Initialize an instance of SquareRootModel

        Parameters
        ----------
        x0: float
            Lower bound of the domain

        x1: float
            Upper bound of the domain
        """
        self.x0 = x0
        self.x1 = x1

    def predict(self, x):
        """
        Return probability density of the SquareRoot model at 'x'.

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]

        Returns
        -------
        float
            Return value of probability density of the SquareRoot model at 'x'.
        """
        return np.sqrt(x) /(2.0/3.0 * (self.x1**(1.5) - self.x0**(1.5)))

    def _LL(self, x, weight):
        """method _LL(x, weight)

        Parameters
        ----------
        x: list of float
            List of data points or data bins in the domain [x0, x1]
        weight: list of float
            List of weights, intensity, or counts at each 'x' in the domain [x0, x1]

        Returns
        -------
        float
            Return logarithmic likelihood of SquareRoot background model

        """
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        """ method maximum_likelihood_estimation(x, weight)
        maximum likelihood estimation of the SquareRoot background model
        Return nothing because the uniform model does not have any parameters.


        Parameters
        ----------
        x: list of float
            List of sampling points or data bins in the domain [x0, x1]
        weight: list of float
            List of weights, intensity, or counts at each 'x' in the domain [x0, x1]

        Returns
        -------
        Nothing
        """
        return


class TriangleModel:
    """
    Class TriangleModel(x0, x1)

    Attributes
    ----------
    x0: float
        Lower bound of the domain
    x1: float
        Upper bound of the domain

    Methods
    -------
    predict(x): Return value of probability density of the Triangle model at 'x'.
    maximum_likelihood_estimation(x, weight): maximum likelihood estimation of the triangular background model.

    """
    def __init__(self, x0, x1):
        """
        Initialize an instance of TriangleModel

        Parameters
        ----------
        x0: float
            Lower bound of the domain
        x1: float
            Upper bound of the domain
        """
        self.x0 = x0
        self.x1 = x1
        self.LL = 0.0

    def predict(self, x):
        """ method predict(x)
        Return probability density of the Triangle model at 'x'.

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]

        Returns
        -------
        float
            Return value of probability density of the Triangular background model at 'x'.
        """
        y1 = 2.0 / (self.x1 - self.x0)
        a = y1 / (self.x1 - self.x0)
        return (a * (x - self.x1) + y1) * (self.x0 < x) * (x <= self.x1)

    def _LL(self, x, weight):
        """
        logarithmic likelihood estimation of the triangular background model.

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]
        weight: float
            List of weights, intensity, or counts at each 'x' in the domain [x0, x1]

        Returns
        -------
        float
            return value of logarithmic likelihood estimation of the triangular background model.
        """
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        """ method maximum_likelihood_estimation(x, weight)

        return nothing because the uniform model does not have any parameters.

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]
        weight: float
            List of weights, intensity, or counts at each 'x' in the domain [x0, x1]

        Returns
        -------
        Nothing

        """
        return


class LinearModel:
    """ Class LinearModel(x, w):
    
    A mixture model of Uniform and Triangular distribution.

    Attributes
    ----------
    x: float
        sampling point or data bins
    w: float
        weight or intensity of the data

    Methods
    -------
    predict(x): Return value of probability density of the Linear model at 'x'.
    maximum_likelihood_estimation(x, weight): maximum likelihood estimation of the linear background model.
    """
    def __init__(self, x0, x1, s_tri=np.random.rand()):
        """
        Initialize an instance of LinearModel

        Parameters
        ----------
        x0: float
            Lower bound of the domain
        x1: float
            Upper bound of the domain
        s_tri: float = numpy.random.rand()
            mixture ratio of triangular distribution. It takes a value within [0, 1]

        """
        self.x0 = x0
        self.x1 = x1
        self.s_tri = s_tri
        self.s_uni = 1.0 - self.s_tri
        self.LL = 0.0

    def predict(self, x):
        """
        Return probability density of the Linear model at 'x'.

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]

        Returns
        -------
        float
            Return value of probability density of the Linear model at 'x'.
        """
        s1 = self.s_tri
        s2 = 1 - s1
        a = (2.0 * s1) / (self.x1 - self.x0) ** 2
        b = s2 / (self.x1 - self.x0) - a * self.x0
        return a*x + b

    def _LL(self, x, weight):
        """
        logarithmic likelihood estimation of the linear background model.

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]
        weight: float
            List of weights, intensity, or counts at each 'x' in the domain [x0, x1]

        Returns
        -------
        float
            return value of logarithmic likelihood estimation of the linear background model.
        """
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        """
        method maximum_likelihood_estimation(x, weight)

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]
        weight: float
            List of weights, intensity, or counts at each 'x' in the domain [x0, x1]

        Returns
        -------
            Optimize attribute of tri_s, mixture ratio of triangular distribution.
        """
        def score(s_tri, x, weight):
            eps = 1e-5
            s_uni = 1 - s_tri
            a = (2.0 * s_tri) / (self.x1 - self.x0) ** 2
            b = s_uni / (self.x1 - self.x0) - a * self.x0

            a_prime = 2.0 / (self.x1 - self.x0) ** 2
            b_prime = -1 / (self.x1 - self.x0) - a_prime * self.x0

            return np.sum(weight * (a_prime * x + b_prime) / (a * x + b + eps))

        try:
            self.s_tri = optimize.brentq(score, 0, 1.0 - 1.0e-5, args=(x, weight))
            self.s_uni = 1.0 - self.s_tri
            return

        except ValueError:
            self.s_tri = (score(0, x, weight) >= score(1, x, weight))
            self.s_uni = (score(0, x, weight) < score(1, x, weight))
            return


class RampModel:
    """ Class RampModel(x0, x1)

    Parameters
    ----------
    x0: float
        Lower bound of the domain
    x1: float
        corner point of the ramp where connecting triangular and uniform distribution.
    x2: float
        Upper bound of the domain

    Methods
    -------
    predict(x): Return value of probability density of the Ramp model at 'x'.
    maximum_likelihood_estimation(x, weight): maximum likelihood estimation of the ramp background model.

    """
    def __init__(self, x0, x1, x2):
        """
        Initialize an instance of RampModel

        Parameters
        ----------
        x0: float
            Lower bound of the domain
        x1: float
            corner point of the ramp where connecting triangular and uniform distribution.
        x2: float
            Upper bound of the domain
        """
        if not x0 < x1 < x2:
            print("Error !!! x1 must between x0 an x2.")
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.LL = 0.0

    def predict(self, x):
        """
        Return probability density of the Ramp model at 'x'.

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]

        Returns
        -------
        float
            Return value of probability density of the Ramp model at 'x'.
        """
        y1 = 2.0 / (2.0 * self.x2 - self.x1 - self.x0)
        a = y1/(self.x1 - self.x0)
        return np.minimum( a * (x - self.x1) + y1, y1 * np.ones(x.size)) * ( x <= self.x2) * (self.x0 < x)

    def _LL(self, x, weight):
        """
        logarithmic likelihood estimation of the Ramp background model.

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]
        weight: float
            List of weights, intensity, or counts at each 'x' in the domain [x0, x1]

        Returns
        -------
        float
            return value of logarithmic likelihood estimation of the Ramp background model.
        """
        t = np.log(self.predict(x))
        self.LL = (t * weight).sum()
        return self.LL

    def maximum_likelihood_estimation(self, x, weight):
        """method maximum_likelihood_estimation(x, weight)

        return nothing because the ramp model does not have any parameters.

        Parameters
        ----------
        x: float
            List of data points or data bins in the domain [x0, x1]
        weight: float
            List of weights, intensity, or counts at each 'x' in the domain [x0, x1]

        Returns
        -------
        Nothing

        """
        return


