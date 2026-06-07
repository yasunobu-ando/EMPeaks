import numpy as np
from scipy.integrate import quad
from scipy.special import voigt_profile
from numpy.polynomial.legendre import leggauss

def _voigt_fwhm(sigma, gamma):
    f_G = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    f_L = 2.0 * gamma
    return 0.5346 * f_L + np.sqrt(0.2166 * f_L**2 + f_G**2)

def test(peak_deg, tail_deg, half_width_mult):
    gx_p, gw_p = leggauss(peak_deg)
    gx_t, gw_t = leggauss(tail_deg)

    def seg(a, b, gx, gw, x0, sigma, gamma):
        if b <= a: return 0.0
        t = 0.5*(b-a)*gx + 0.5*(a+b)
        y = voigt_profile(t - x0, sigma, gamma)
        return 0.5*(b-a)*np.sum(gw*y)
    
    def composite(x_min, x_max, x0, sigma, gamma):
        fwhm = _voigt_fwhm(sigma, gamma)
        hw = half_width_mult * fwhm
        plo = max(x0-hw, x_min); phi = min(x0+hw, x_max)
        return (seg(x_min, plo, gx_t, gw_t, x0, sigma, gamma) +
                seg(plo, phi, gx_p, gw_p, x0, sigma, gamma) +
                seg(phi, x_max, gx_t, gw_t, x0, sigma, gamma))

    cases = [
        ("広区間・鋭1",  -300, 300,  0,   1.0,  1.0),
        ("広区間・鋭2",  -300, 300,  0,   0.1,  0.1),
        ("極端に鋭い",   -300, 300, 50, 0.05, 0.05),
    ]
    
    total_pts = peak_deg + 2*tail_deg
    print(f"peak={peak_deg}, tail={tail_deg}, hw={half_width_mult}*FWHM, total={total_pts}pts")
    for name, xmin, xmax, x0, s, g in cases:
        true = quad(lambda x: voigt_profile(x-x0, s, g), xmin, xmax, epsabs=1e-12, epsrel=1e-12)[0]
        val = composite(xmin, xmax, x0, s, g)
        print(f"  {name:16s}: err={abs(val-true):.4e}")
    print()

for pd in [100, 150, 200]:
    for td in [30, 50]:
        for hw in [5.0, 10.0]:
            test(pd, td, hw)
