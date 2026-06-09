"""
検証1: 4段ロケット戦略の妥当性
検証2: full_optimization_fix_sigma_gamma の init/bounds 不一致バグ
"""
import numpy as np
from scipy.special import voigt_profile
from scipy import optimize
import sys
sys.path.insert(0, '.')
from _voigt import Voigt, _composite_gauss_quad

# ============================================================
# 検証1: 4段ロケット戦略 vs 直接フル最適化
# ============================================================
print("=" * 80)
print("検証1: 4段ロケット戦略の効果")
print("  全パラメータ自由時に、部分最適化3回→フル最適化 vs フル最適化のみ")
print("=" * 80)

# テストデータ生成（既知のVoigtピーク）
x = np.linspace(-50, 50, 500)
true_x0, true_sigma, true_gamma = 5.0, 2.0, 1.5
true_profile = voigt_profile(x - true_x0, true_sigma, true_gamma)
Z = _composite_gauss_quad(-50, 50, true_x0, true_sigma, true_gamma)
intensity = true_profile / Z * 10000  # 正規化されたデータ

np.random.seed(42)
n_trials = 20
results_4stage = []
results_direct = []

for trial in range(n_trials):
    # 4段ロケット戦略
    v4 = Voigt(x_min=-50, x_max=50, sigma_min=0.01, sigma_max=10, gamma_min=0.01, gamma_max=10)
    v4.x0 = np.random.uniform(-50, 50)
    v4.sigma = np.random.uniform(0.01, 10)
    v4.gamma = np.random.uniform(0.01, 10)
    
    # 同一の初期値をコピー
    vd = Voigt(x_min=-50, x_max=50, sigma_min=0.01, sigma_max=10, gamma_min=0.01, gamma_max=10)
    vd.x0 = v4.x0
    vd.sigma = v4.sigma
    vd.gamma = v4.gamma
    
    # 4段ロケット
    v4.maximum_likelihood_estimation(x, intensity)
    ll_4 = v4._LL(x, intensity)
    
    # 直接フル最適化のみ
    vd.full_optimization(x, intensity)
    ll_d = vd._LL(x, intensity)
    
    results_4stage.append((ll_4, v4.x0, v4.sigma, v4.gamma))
    results_direct.append((ll_d, vd.x0, vd.sigma, vd.gamma))

print(f"\n{'Trial':>5s} {'4段ロケットLL':>16s} {'直接フルLL':>16s} {'差(4段-直接)':>14s} {'4段が優位?':>10s}")
print("-" * 70)
wins_4stage = 0
for i in range(n_trials):
    ll4 = results_4stage[i][0]
    lld = results_direct[i][0]
    diff = ll4 - lld
    better = "✓" if diff > 1e-6 else ("≈" if abs(diff) < 1e-6 else "✗")
    if diff > 1e-6:
        wins_4stage += 1
    print(f"{i:5d} {ll4:16.6f} {lld:16.6f} {diff:14.6e} {better:>10s}")

print(f"\n4段ロケットが優位だった回数: {wins_4stage}/{n_trials}")

# ============================================================
# 検証2: full_optimization_fix_sigma_gamma のバグ
# ============================================================
print("\n" + "=" * 80)
print("検証2: full_optimization_fix_sigma_gamma の init/bounds 不一致バグ")
print("  init = [x0, self.sigma, self.gamma] (3要素)")
print("  bounds = [(self.x_min, self.x_max)] (1要素)")
print("=" * 80)

# scipy.optimize.minimize の挙動を確認
def test_bounds_mismatch():
    """init が3要素、bounds が1要素の場合の scipy の挙動をテスト"""
    def f(param):
        return (param[0] - 3.0)**2
    
    # 正常ケース: init=1要素, bounds=1要素
    r1 = optimize.minimize(f, x0=[1.0], bounds=[(-10, 10)], method='L-BFGS-B')
    print(f"  正常ケース: init=[1.0], bounds=[(-10,10)]")
    print(f"    結果: x={r1.x}, success={r1.success}")
    
    # バグケース: init=3要素, bounds=1要素
    try:
        r2 = optimize.minimize(f, x0=[1.0, 2.0, 3.0], bounds=[(-10, 10)], method='L-BFGS-B')
        print(f"  バグケース: init=[1.0,2.0,3.0], bounds=[(-10,10)]")
        print(f"    結果: x={r2.x}, success={r2.success}")
        print(f"    ⚠️  scipy は init の余分な要素を無視して最適化を実行！")
    except Exception as e:
        print(f"  バグケース: エラー発生: {e}")

test_bounds_mismatch()

# 実際の Voigt で確認
print("\n実際の Voigt での検証:")
v = Voigt(x_min=-50, x_max=50, sigma_min=0.01, sigma_max=10, gamma_min=0.01, gamma_max=10)
v.x0 = 3.0
v.sigma = 2.0
v.gamma = 1.5

print(f"  Before: x0={v.x0:.4f}, sigma={v.sigma:.4f}, gamma={v.gamma:.4f}")
v.full_optimization_fix_sigma_gamma(x, intensity)
print(f"  After:  x0={v.x0:.4f}, sigma={v.sigma:.4f}, gamma={v.gamma:.4f}")
print(f"  → sigma と gamma は変更されていないか?: sigma unchanged={v.sigma==2.0}, gamma unchanged={v.gamma==1.5}")
