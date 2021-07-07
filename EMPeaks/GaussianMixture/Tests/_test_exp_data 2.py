# License: BSD-3-clause
# Copyright © 2020 National Institute of Advanced Industrial Science and Technology (AIST)


from EMPeaks.GaussianMixture import SpectrumAdaptedEM
from EMPeaks import file_io


def test_exp_data(file, sampling=1):
    #path = file_io.__file__.rstrip("file_io.py")
    #fname = path+'GaussianMixture/Tests/GFET0126_25V_C1s.txt'
    fname = file
    dat = file_io.FileIO(fname, 'GFET')
    opt = SpectrumAdaptedEM(data=dat.data2d, K=3)
    opt.eps = 1e-10
    print('initial param:', opt.param)
    opt.fit(sampling=sampling)
    opt.plot_fitting_summary()
    print('\n A data-file example is located at \n\t', fname)

    return
