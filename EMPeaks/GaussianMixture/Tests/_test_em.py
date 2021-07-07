# License: BSD-3-clause
# Copyright © 2020 National Institute of Advanced Industrial Science and Technology (AIST)

from ._test_data import TestData
from .._em import EM


def test_em(N=10000, sampling=1):
    dat = TestData(N)
    dat.show_info()
    print("---data1D---")
    print("shape:  {}".format(dat.data1D.shape))

    opt = EM(dat.data1D)
    opt.eps = 1e-10
    print('\ninitial param:', opt.param)
    opt.fit(sampling=sampling)
    opt.plot_fitting_summary()

    return print("\nfinished.")
