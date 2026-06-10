import time
import numpy as np
from scipy.integrate import quad
from scipy.special import voigt_profile
from numpy.polynomial.legendre import leggauss

# 新実装を直接インポート
from _voigt import _composite_gauss_quad

# 旧実装（単一区間200点）
_GAUSS_X_OLD, _GAUSS_W_OLD = leggauss(200)
def _old_gauss_quad(x_min, x_max, x0, sigma, gamma):
    t = 0.5 * (x_max - x_min) * _GAUSS_X_OLD + 0.5 * (x_min + x_max)
    y = voigt_profile(t - x0, sigma, gamma)
    return 0.5 * (x_max - x_min) * np.sum(_GAUSS_W_OLD * y)

def _true_quad(x_min, x_max, x0, sigma, gamma):
    def integrand(x):
        return voigt_profile(x - x0, sigma, gamma)
    return quad(integrand, x_min, x_max, epsabs=1e-12, epsrel=1e-12)[0]

cases = [
    ("広区間・鋭ピーク",   -300, 300,   0,    1.0, 1.0),
    ("広区間・鋭ピーク2",  -300, 300,   0,    0.1, 0.1),
    ("中区間・普通ピーク", -100, 100, 10.5,  2.0, 1.5),
    ("狭区間・広ピーク",   -10,   10,   0,    0.1, 0.1),
    ("極端に鋭い",        -300, 300,  50,  0.05, 0.05),
]

print("=" * 110)
print(f"{'ケース':20s} {'真値(quad)':>18s} {'旧(200pt)誤差':>14s} {'新(複合)誤差':>14s} {'改善倍率':>10s}")
print("=" * 110)

for name, x_min, x_max, x0, sigma, gamma in cases:
    true = _true_quad(x_min, x_max, x0, sigma, gamma)
    old  = _old_gauss_quad(x_min, x_max, x0, sigma, gamma)
    new  = _composite_gauss_quad(x_min, x_max, x0, sigma, gamma)
    err_old = abs(old - true)
    err_new = abs(new - true)
    ratio = err_old / err_new if err_new > 0 else float('inf')
    print(f"{name:20s} {true:18.14f} {err_old:14.4e} {err_new:14.4e} {ratio:10.0f}x")

print()
N = 1000
x0, sigma, gamma = 10.5, 2.0, 1.5

t0 = time.perf_counter()
for _ in range(N):
    _true_quad(-100, 100, x0, sigma, gamma)
t1 = time.perf_counter()
print(f"scipy.quad  ({N}回): {(t1-t0)/N*1e6:.1f} µs/回")

t0 = time.perf_counter()
for _ in range(N):
    _old_gauss_quad(-100, 100, x0, sigma, gamma)
t1 = time.perf_counter()
print(f"旧(200点)   ({N}回): {(t1-t0)/N*1e6:.1f} µs/回")

t0 = time.perf_counter()
for _ in range(N):
    _composite_gauss_quad(-100, 100, x0, sigma, gamma)
t1 = time.perf_counter()
print(f"新(複合)    ({N}回): {(t1-t0)/N*1e6:.1f} µs/回")
