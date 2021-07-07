# License: BSD-3-clause
# Copyright © 2020 National Institute of Advanced Industrial Science and Technology (AIST)


from EMPeaks.GaussianMixture.Tests._test_data import TestData
from EMPeaks.GaussianMixture._spectrum_adapted_em import SpectrumAdaptedEM


def test_spectrum_adapted_em(N=10000, sampling=1):
    dat = TestData(N)
    dat.show_info()
    print("---data2D---")
    print("shape:  {}".format(dat.data2D.shape))

    opt = SpectrumAdaptedEM(dat.data2D)
    opt.eps = 1e-10
    print('initial param:', opt.param)
    opt.fit(sampling=sampling)
    opt.plot_fitting_summary()

    return print("finished.")


if __name__ =="__main__":
    test_spectrum_adapted_em(sampling=10)