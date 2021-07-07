Sample Module Repository
========================

This package is for high-throughput peak analysis by using Spectrum Adapted EM algorithm.
We prepared three examples to test this package.

1. test_em(N: int) ::

    from EMPeaks import GaussianMixture
    test = GaussianMixture.Tests
    test.test_em(N=10000,sampling=5)

2. test_spectrum_adapted_em(N: int) ::

    from EMPeaks import GaussianMixture
    test = GaussianMixture.Tests
    test.test_spectrum_adapted_em(N=10000,sampling=5)

3. test_exp_data() ::

    from EMPeaks import GaussianMixture
    test = GaussianMixture.Tests
    test.test_exp_data(file, sampling=5)

---------------

This project is formatted by an example repo for Python projects.
`Learn more <http://www.kennethreitz.org/essays/repository-structure-and-python>`_.
If you want to learn more about ``setup.py`` files, check out `this repository <https://github.com/kennethreitz/setup.py>`_.

