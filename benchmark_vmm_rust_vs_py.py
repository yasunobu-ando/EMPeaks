import time
import numpy as np
import matplotlib.pyplot as plt

# バックエンド切り替え用のモジュール
from EMPeaks.EMCore import _backend
from EMPeaks.VoigtMixture._vmm import VoigtMixtureModel

def make_test_data():
    x = np.arange(-200, 200, 0.5)
    
    # 2つのVoigtピークを生成
    from scipy.special import voigt_profile
    
    # Peak 1: x0=-50, sigma=10, gamma=5
    y1 = 1000 * voigt_profile(x - (-50), 10, 5)
    # Peak 2: x0=30, sigma=5, gamma=15
    y2 = 800 * voigt_profile(x - 30, 5, 15)
    
    y = y1 + y2
    
    # ノイズを追加
    np.random.seed(42)
    y += np.random.normal(0, np.sqrt(y + 1))
    
    # 負の値をゼロクリップ
    y = np.clip(y, 1e-10, None)
    return x, y

def run_benchmark():
    x, y = make_test_data()
    
    print("=" * 60)
    print("VoigtMixtureModel Benchmark: Python vs Rust (trial=50)")
    print("=" * 60)
    
    # 共通のパラメータ
    K = 2
    trial = 50
    x_min, x_max = -200, 200
    
    # ----------------------------------------------------
    # Python Backend
    # ----------------------------------------------------
    print("\n[Running Python Backend...]")
    _backend.set_backend("python")
    
    vmm_py = VoigtMixtureModel(K=K, x_min=x_min, x_max=x_max)
    
    start_py_samp = time.time()
    info_py_samp = vmm_py.sampling(x, y, trial=trial, stdout=False)
    time_py_samp = time.time() - start_py_samp
    
    # sampling のベストパラメータをリセットして fit でフル最適化
    start_py_fit = time.time()
    info_py_fit = vmm_py.fit(x, y, stdout=False)
    time_py_fit = time.time() - start_py_fit
    
    # ----------------------------------------------------
    # Rust Backend
    # ----------------------------------------------------
    print("[Running Rust Backend...]")
    _backend.set_backend("rust")
    
    vmm_rs = VoigtMixtureModel(K=K, x_min=x_min, x_max=x_max)
    
    start_rs_samp = time.time()
    info_rs_samp = vmm_rs.sampling(x, y, trial=trial, stdout=False)
    time_rs_samp = time.time() - start_rs_samp
    
    start_rs_fit = time.time()
    info_rs_fit = vmm_rs.fit(x, y, stdout=False)
    time_rs_fit = time.time() - start_rs_fit
    
    # ----------------------------------------------------
    # 結果の表示
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("Benchmark Results Summary")
    print("=" * 60)
    
    print(f"{'Metric':<25} | {'Python':<15} | {'Rust':<15} | {'Speedup'}")
    print("-" * 60)
    
    # 時間の比較
    speedup_samp = time_py_samp / time_rs_samp if time_rs_samp > 0 else 0
    print(f"{'Sampling (trial=50) Time':<25} | {time_py_samp:12.3f}s | {time_rs_samp:12.3f}s | {speedup_samp:5.1f}x")
    
    speedup_fit = time_py_fit / time_rs_fit if time_rs_fit > 0 else 0
    print(f"{'Fit (EM) Time':<25} | {time_py_fit:12.3f}s | {time_rs_fit:12.3f}s | {speedup_fit:5.1f}x")
    
    total_py = time_py_samp + time_py_fit
    total_rs = time_rs_samp + time_rs_fit
    speedup_tot = total_py / total_rs if total_rs > 0 else 0
    print(f"{'Total Time':<25} | {total_py:12.3f}s | {total_rs:12.3f}s | {speedup_tot:5.1f}x")
    print("-" * 60)
    
    # 精度の比較
    ll_py = info_py_fit['LL']
    ll_rs = info_rs_fit['LL']
    print(f"{'Final LogLikelihood':<25} | {ll_py:12.6e}  | {ll_rs:12.6e}  | {'-'}")
    
    rmse_py = info_py_fit['RMSE']
    rmse_rs = info_rs_fit['RMSE']
    print(f"{'Final RMSE':<25} | {rmse_py:12.6e}  | {rmse_rs:12.6e}  | {'-'}")
    
    iter_py = info_py_fit['total_iter']
    iter_rs = info_rs_fit['total_iter']
    print(f"{'Total Iterations':<25} | {iter_py:12d}    | {iter_rs:12d}    | {'-'}")

if __name__ == '__main__':
    run_benchmark()
