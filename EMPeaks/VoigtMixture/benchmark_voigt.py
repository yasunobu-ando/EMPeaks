import time
import numpy as np
from scipy.integrate import quad
from scipy.special import voigt_profile
from numpy.polynomial.legendre import leggauss

class BenchVoigt:
    def __init__(self, x_min=-100, x_max=100, x0=0, sigma=1.0, gamma=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.x0 = x0
        self.sigma = sigma
        self.gamma = gamma
        self._gauss_x, self._gauss_w = leggauss(200)

    def _Z_quad(self):
        def integrand(x):
            return voigt_profile(x-self.x0, self.sigma, self.gamma)
        return quad(integrand, self.x_min, self.x_max, epsabs=1.49e-08, epsrel=1.49e-08)[0]

    def _Z_gauss(self):
        a = self.x_min
        b = self.x_max
        t = 0.5 * (b - a) * self._gauss_x + 0.5 * (a + b)
        y = voigt_profile(t - self.x0, self.sigma, self.gamma)
        return 0.5 * (b - a) * np.sum(self._gauss_w * y)

v = BenchVoigt(x_min=-100, x_max=100, x0=10.5, sigma=2.0, gamma=1.5)

# Accuracy Check
z_quad = v._Z_quad()
z_gauss = v._Z_gauss()
print("=== 精度 (Accuracy) ===")
print(f"旧手法 (scipy.integrate.quad): {z_quad:.14f}")
print(f"新手法 (Gauss-Legendre 200次): {z_gauss:.14f}")
print(f"絶対誤差 (Absolute Error): {abs(z_quad - z_gauss):.14e}")
print("")

# Speed Check
N_ITER = 1000

print("=== 速度 (Speed) ===")
t0 = time.perf_counter()
for _ in range(N_ITER):
    v._Z_quad()
t1 = time.perf_counter()
time_quad = (t1 - t0) / N_ITER

t0 = time.perf_counter()
for _ in range(N_ITER):
    v._Z_gauss()
t1 = time.perf_counter()
time_gauss = (t1 - t0) / N_ITER

print(f"旧手法 ({N_ITER}回平均): {time_quad*1e6:.2f} µs / 回")
print(f"新手法 ({N_ITER}回平均): {time_gauss*1e6:.2f} µs / 回")
print(f"速度向上率: 約 {time_quad / time_gauss:.1f} 倍高速")
