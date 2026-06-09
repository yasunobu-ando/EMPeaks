import time
import numpy as np
from scipy.integrate import quad
from scipy.special import voigt_profile
from numpy.polynomial.legendre import leggauss

# 複合ガウス求積の実装（_voigt.py からコピー）
_GAUSS_X_PEAK, _GAUSS_W_PEAK = leggauss(100)
_GAUSS_X_TAIL, _GAUSS_W_TAIL = leggauss(30)

def _voigt_fwhm(sigma, gamma):
    f_G = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    f_L = 2.0 * gamma
    return 0.5346 * f_L + np.sqrt(0.2166 * f_L**2 + f_G**2)

def _gauss_legendre_segment(a, b, gauss_x, gauss_w, x0, sigma, gamma):
    if b <= a:
        return 0.0
    t = 0.5 * (b - a) * gauss_x + 0.5 * (a + b)
    y = voigt_profile(t - x0, sigma, gamma)
    return 0.5 * (b - a) * np.sum(gauss_w * y)

def _composite_gauss_quad(x_min, x_max, x0, sigma, gamma):
    fwhm = _voigt_fwhm(sigma, gamma)
    half_width = 5.0 * fwhm
    peak_lo = max(x0 - half_width, x_min)
    peak_hi = min(x0 + half_width, x_max)
    z  = _gauss_legendre_segment(x_min,  peak_lo, _GAUSS_X_TAIL, _GAUSS_W_TAIL, x0, sigma, gamma)
    z += _gauss_legendre_segment(peak_lo, peak_hi, _GAUSS_X_PEAK, _GAUSS_W_PEAK, x0, sigma, gamma)
    z += _gauss_legendre_segment(peak_hi, x_max,   _GAUSS_X_TAIL, _GAUSS_W_TAIL, x0, sigma, gamma)
    return z

# 旧実装（単一区間200点）
_GAUSS_X_OLD, _GAUSS_W_OLD = leggauss(200)
def _old_gauss_quad(x_min, x_max, x0, sigma, gamma):
    t = 0.5 * (x_max - x_min) * _GAUSS_X_OLD + 0.5 * (x_min + x_max)
    y = voigt_profile(t - x0, sigma, gamma)
    return 0.5 * (x_max - x_min) * np.sum(_GAUSS_W_OLD * y)

# scipy.integrate.quad（真値）
def _true_quad(x_min, x_max, x0, sigma, gamma):
    def integrand(x):
        return voigt_profile(x - x0, sigma, gamma)
    return quad(integrand, x_min, x_max, epsabs=1e-12, epsrel=1e-12)[0]

# テストケース
cases = [
    ("広区間・鋭ピーク",   -300, 300,   0,    1.0, 1.0),
    ("広区間・鋭ピーク2",  -300, 300,   0,    0.1, 0.1),
    ("中区間・普通ピーク", -100, 100, 10.5,  2.0, 1.5),
    ("狭区間・広ピーク",   -10,   10,   0,    0.1, 0.1),
    ("極端に鋭い",        -300, 300,  50,    0.05, 0.05),
]

print("=" * 100)
print(f"{'ケース':20s} {'真値(quad)':>18s} {'旧(200点)':>18s} {'旧誤差':>12s} {'新(複合)':>18s} {'新誤差':>12s}")
print("=" * 100)

for name, x_min, x_max, x0, sigma, gamma in cases:
    true = _true_quad(x_min, x_max, x0, sigma, gamma)
    old  = _old_gauss_quad(x_min, x_max, x0, sigma, gamma)
    new  = _composite_gauss_quad(x_min, x_max, x0, sigma, gamma)
    err_old = abs(old - true)
    err_new = abs(new - true)
    print(f"{name:20s} {true:18.14f} {old:18.14f} {err_old:12.4e} {new:18.14f} {err_new:12.4e}")

print()
# 速度比較
N = 1000
x0, sigma, gamma = 10.5, 2.0, 1.5
x_min, x_max = -100, 100

t0 = time.perf_counter()
for _ in range(N):
    _true_quad(x_min, x_max, x0, sigma, gamma)
t1 = time.perf_counter()
print(f"scipy.quad  ({N}回): {(t1-t0)/N*1e6:.1f} µs/回")

t0 = time.perf_counter()
for _ in range(N):
    _old_gauss_quad(x_min, x_max, x0, sigma, gamma)
t1 = time.perf_counter()
print(f"旧(200点)   ({N}回): {(t1-t0)/N*1e6:.1f} µs/回")

t0 = time.perf_counter()
for _ in range(N):
    _composite_gauss_quad(x_min, x_max, x0, sigma, gamma)
t1 = time.perf_counter()
print(f"新(複合)    ({N}回): {(t1-t0)/N*1e6:.1f} µs/回")
