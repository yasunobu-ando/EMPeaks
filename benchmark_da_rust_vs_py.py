import time
import numpy as np

from EMPeaks.EMCore import _backend
from EMPeaks.VoigtMixture._vmm import VoigtMixtureModel

def make_10_peaks_data():
    x = np.arange(-500, 500, 0.5)
    
    from scipy.special import voigt_profile
    np.random.seed(42)
    
    y = np.zeros_like(x)
    
    # 10個のピークをランダムに配置
    for _ in range(10):
        x0 = np.random.uniform(-400, 400)
        sigma = np.random.uniform(5, 20)
        gamma = np.random.uniform(5, 20)
        intensity = np.random.uniform(500, 2000)
        y += intensity * voigt_profile(x - x0, sigma, gamma)
        
    # ノイズを追加
    y += np.random.normal(0, np.sqrt(y + 1))
    y = np.clip(y, 1e-10, None)
    
    return x, y

def run_benchmark():
    x, y = make_10_peaks_data()
    
    print("=" * 65)
    print("Deterministic Annealing Benchmark: Python vs Rust")
    print("VoigtMixtureModel, K=10, trial=10")
    print("=" * 65)
    
    K = 10
    trial = 10
    x_min, x_max = -500, 500
    
    # ----------------------------------------------------
    # Python Backend
    # ----------------------------------------------------
    print("\n[Running Python Backend...]")
    _backend.set_backend("python")
    
    vmm_py = VoigtMixtureModel(K=K, x_min=x_min, x_max=x_max)
    
    start_py = time.time()
    info_py = vmm_py.sampling(x, y, method='deterministic_annealing', trial=trial, stdout=False)
    time_py = time.time() - start_py
    
    # ----------------------------------------------------
    # Rust Backend
    # ----------------------------------------------------
    print("[Running Rust Backend...]")
    _backend.set_backend("rust")
    
    vmm_rs = VoigtMixtureModel(K=K, x_min=x_min, x_max=x_max)
    
    start_rs = time.time()
    info_rs = vmm_rs.sampling(x, y, method='deterministic_annealing', trial=trial, stdout=False)
    time_rs = time.time() - start_rs
    
    # ----------------------------------------------------
    # 結果の表示
    # ----------------------------------------------------
    print("\n" + "=" * 65)
    print("Benchmark Results Summary")
    print("=" * 65)
    
    print(f"{'Metric':<25} | {'Python':<15} | {'Rust':<15} | {'Speedup'}")
    print("-" * 65)
    
    speedup = time_py / time_rs if time_rs > 0 else 0
    print(f"{'DA Sampling (trial=10)':<25} | {time_py:12.3f}s | {time_rs:12.3f}s | {speedup:5.1f}x")
    print("-" * 65)
    
    # 精度の比較
    idx_py = info_py['index_best']
    idx_rs = info_rs['index_best']
    
    ll_py = info_py['LL_hist'][idx_py]
    ll_rs = info_rs['LL_hist'][idx_rs]
    print(f"{'Best LogLikelihood':<25} | {ll_py:12.6e}  | {ll_rs:12.6e}  | {'-'}")
    
    rmse_py = info_py['RMSE_hist'][idx_py]
    rmse_rs = info_rs['RMSE_hist'][idx_rs]
    print(f"{'Best RMSE':<25} | {rmse_py:12.6e}  | {rmse_rs:12.6e}  | {'-'}")

if __name__ == '__main__':
    run_benchmark()
