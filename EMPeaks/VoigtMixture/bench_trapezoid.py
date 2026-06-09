import time
import numpy as np
from scipy.integrate import quad
from scipy.special import voigt_profile
import sys

# Import the current composite gauss quadrature
sys.path.insert(0, '.')
from _voigt import _composite_gauss_quad

def _true_quad(x_min, x_max, x0, sigma, gamma):
    def integrand(x):
        return voigt_profile(x - x0, sigma, gamma)
    return quad(integrand, x_min, x_max, epsabs=1e-12, epsrel=1e-12)[0]

def _trapezoid_rule(x_min, x_max, x0, sigma, gamma, n_points):
    x = np.linspace(x_min, x_max, n_points)
    y = voigt_profile(x - x0, sigma, gamma)
    return np.trapz(y, x)

cases = [
    ("広区間・鋭ピーク",   -300, 300,   0,    1.0, 1.0),
    ("広区間・鋭ピーク2",  -300, 300,   0,    0.1, 0.1),
    ("中区間・普通ピーク", -100, 100, 10.5,  2.0, 1.5),
    ("狭区間・広ピーク",   -10,   10,   0,    0.1, 0.1),
    ("極端に鋭い",        -300, 300,  50,  0.05, 0.05),
]

n_points_list = [200, 500, 1000, 2000, 5000, 10000]

print("=" * 120)
print(f"{'ケース':20s} {'真値(quad)':>18s} {'複合ガウス(200pt)':>18s} " + " ".join([f"台形({n})" for n in n_points_list]))
print("=" * 120)

for name, x_min, x_max, x0, sigma, gamma in cases:
    true = _true_quad(x_min, x_max, x0, sigma, gamma)
    comp = _composite_gauss_quad(x_min, x_max, x0, sigma, gamma)
    
    trap_vals = []
    for n in n_points_list:
        trap_vals.append(_trapezoid_rule(x_min, x_max, x0, sigma, gamma, n))
        
    err_comp = abs(comp - true)
    err_traps = [abs(v - true) for v in trap_vals]
    
    print(f"{name:20s} {true:18.14f} {err_comp:18.4e} " + " ".join([f"{err:8.1e}" for err in err_traps]))

print("\n--- 速度比較 ---")
N_RUNS = 1000
x0, sigma, gamma = 10.5, 2.0, 1.5

t0 = time.perf_counter()
for _ in range(N_RUNS):
    _true_quad(-100, 100, x0, sigma, gamma)
t1 = time.perf_counter()
print(f"scipy.quad         ({N_RUNS}回): {(t1-t0)/N_RUNS*1e6:.1f} µs/回")

t0 = time.perf_counter()
for _ in range(N_RUNS):
    _composite_gauss_quad(-100, 100, x0, sigma, gamma)
t1 = time.perf_counter()
print(f"新複合ガウス(200点)({N_RUNS}回): {(t1-t0)/N_RUNS*1e6:.1f} µs/回")

for n in [200, 500, 1000, 2000, 5000]:
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _trapezoid_rule(-100, 100, x0, sigma, gamma, n)
    t1 = time.perf_counter()
    print(f"台形則({n:>5d}点)    ({N_RUNS}回): {(t1-t0)/N_RUNS*1e6:.1f} µs/回")

