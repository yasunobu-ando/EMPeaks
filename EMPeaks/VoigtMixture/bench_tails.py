import time
import numpy as np
from scipy.integrate import quad
from scipy.special import voigt_profile
from numpy.polynomial.legendre import leggauss

# ==========================================
# 手法1: 裾野引き算方式（提案手法）
# ==========================================
# [0, 1] 区間のガウス・ルジャンドル求積点
_GAUSS_X_TAIL, _GAUSS_W_TAIL = leggauss(50)
_U = 0.5 * _GAUSS_X_TAIL + 0.5
_W = 0.5 * _GAUSS_W_TAIL

def _tail_integral(limit, direction, x0, sigma, gamma):
    """
    direction=1: [limit, ∞) の積分
    direction=-1: (-∞, limit] の積分
    変数変換: x = limit + direction * (u / (1 - u))  (u in [0, 1])
    dx = direction * (1 / (1 - u)^2) du
    """
    u = _U
    w = _W
    
    # 変換後のx
    x = limit + direction * (u / (1.0 - u))
    
    # ヤコビアン
    jac = 1.0 / ((1.0 - u)**2)
    
    y = voigt_profile(x - x0, sigma, gamma)
    
    # u=1 (x=±∞) の極限での特異性を回避しつつ積分
    # Voigtの裾野は 1/x^2 で落ちるため、jac と打ち消し合って被積分関数は有限値になる
    return np.sum(w * y * jac)

def _subtract_tails_quad(x_min, x_max, x0, sigma, gamma):
    """ 1.0 から両裾の積分を引く """
    # x_min, x_max がピークを跨いでいない（片側のみ）の場合などの分岐も考慮可能ですが、
    # 基本的にVoigtの全区間積分は1.0なので、引けば良い。
    # ただし x_min > x_max などのエラー処理は省く。
    
    left_tail = _tail_integral(x_min, -1, x0, sigma, gamma)
    right_tail = _tail_integral(x_max, 1, x0, sigma, gamma)
    
    return 1.0 - left_tail - right_tail

# ==========================================
# 手法2: 複合ガウス求積（現在の実装）
# ==========================================
import sys
sys.path.insert(0, '.')
from _voigt import _composite_gauss_quad

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
    ("片側のみ(ピーク含)", -300,  10,  50,  2.0,  1.5), # a=-300, b=10 にピーク50は含まれない
    ("片側のみ(裾野のみ)",  100, 300,   0,  1.0,  1.0),
]

print("=" * 110)
print(f"{'ケース':20s} {'真値(quad)':>18s} {'裾野引算(50x2)':>18s} {'引算の誤差':>14s} {'複合ガウスの誤差':>14s}")
print("=" * 110)

for name, x_min, x_max, x0, sigma, gamma in cases:
    true = _true_quad(x_min, x_max, x0, sigma, gamma)
    comp = _composite_gauss_quad(x_min, x_max, x0, sigma, gamma)
    tails = _subtract_tails_quad(x_min, x_max, x0, sigma, gamma)
    
    err_comp = abs(comp - true)
    err_tails = abs(tails - true)
    
    print(f"{name:20s} {true:18.14f} {tails:18.14f} {err_tails:14.4e} {err_comp:14.4e}")

print("\n--- 速度比較 ---")
N_RUNS = 1000
x0, sigma, gamma = 10.5, 2.0, 1.5

t0 = time.perf_counter()
for _ in range(N_RUNS):
    _composite_gauss_quad(-100, 100, x0, sigma, gamma)
t1 = time.perf_counter()
print(f"複合ガウス(200点) ({N_RUNS}回): {(t1-t0)/N_RUNS*1e6:.1f} µs/回")

t0 = time.perf_counter()
for _ in range(N_RUNS):
    _subtract_tails_quad(-100, 100, x0, sigma, gamma)
t1 = time.perf_counter()
print(f"裾野引算(100点)   ({N_RUNS}回): {(t1-t0)/N_RUNS*1e6:.1f} µs/回")
