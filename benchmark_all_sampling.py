import time
import numpy as np
import pandas as pd
from EMPeaks.EMCore._backend import set_backend
from EMPeaks.GaussianMixture import GaussianMixtureModel
from EMPeaks.PseudoVoigtMixture import PseudoVoigtMixtureModel
from EMPeaks.LorentzianMixture import LorentzianMixtureModel
from EMPeaks.DoniachSunjicMixture import DoniachSunjicMixtureModel
from EMPeaks.TSDCMixture import TSDCMixtureModel

def make_data():
    x = np.linspace(-10, 10, 500)
    intensity = np.exp(-x**2) + 0.5 * np.exp(-(x-3)**2) + 0.3 * np.exp(-(x+4)**2)
    intensity /= intensity.sum()
    return x, intensity

def make_tsdc_data():
    # TSDC is typically defined for positive T
    T = np.linspace(300, 600, 500)
    # create synthetic data with 2 peaks roughly
    intensity = np.exp(-((T-400)/20)**2) + 0.5 * np.exp(-((T-500)/30)**2)
    intensity /= intensity.sum()
    return T, intensity

def run_benchmark():
    x, intensity = make_data()
    t_x, t_int = make_tsdc_data()
    
    models = {
        "Gaussian": (GaussianMixtureModel(K=2), x, intensity),
        "PseudoVoigt": (PseudoVoigtMixtureModel(K=2), x, intensity),
        "Lorentzian": (LorentzianMixtureModel(K=2, x_min=-10, x_max=10), x, intensity),
        "DoniachSunjic": (DoniachSunjicMixtureModel(K=2, x_min=-10, x_max=10), x, intensity),
        "TSDC": (TSDCMixtureModel(K=2, T_min=300, T_max=600), t_x, t_int),
    }

    results = []

    trial = 50
    max_iter = 100 # limit max iter so it doesn't take too long for python

    for name, (model, data_x, data_y) in models.items():
        print(f"Benchmarking {name}...")
        
        # Python
        set_backend("python")
        start = time.time()
        try:
            model.sampling(data_x, data_y, method='adapted_em', trial=trial, max_iter=max_iter, stdout=False)
            t_py = time.time() - start
            py_str = f"{t_py:.3f}"
        except Exception as e:
            t_py = -1
            py_str = "Error"

        # Rust
        set_backend("rust")
        start = time.time()
        try:
            model.sampling(data_x, data_y, method='adapted_em', trial=trial, max_iter=max_iter, stdout=False)
            t_rs = time.time() - start
            rs_str = f"{t_rs:.3f}"
        except Exception as e:
            t_rs = -1
            rs_str = "Error"

        if t_py > 0 and t_rs > 0:
            speedup = t_py / t_rs
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup_str = "N/A"
        
        results.append({
            "Model": name,
            "Python (sec)": py_str,
            "Rust (sec)": rs_str,
            "Speedup": speedup_str
        })

    df = pd.DataFrame(results)
    df.to_csv('benchmark_results.csv', index=False)
    print("\n=== Sampling Benchmark Results (trial=50, max_iter=100) ===")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_benchmark()
