import numpy as np
import time
from EMPeaks.EMCore._backend import set_backend
from EMPeaks.GaussianMixture import GaussianMixtureModel
from EMPeaks.PseudoVoigtMixture import PseudoVoigtMixtureModel
from EMPeaks.LorentzianMixture import LorentzianMixtureModel
from EMPeaks.DoniachSunjicMixture import DoniachSunjicMixtureModel
from EMPeaks.TSDCMixture import TSDCMixtureModel
import pandas as pd

models = [
    ("Gaussian", GaussianMixtureModel(K=2)),
    ("PseudoVoigt", PseudoVoigtMixtureModel(K=2)),
    ("Lorentzian", LorentzianMixtureModel(K=2)),
    ("DoniachSunjic", DoniachSunjicMixtureModel(K=2)),
    ("TSDC", TSDCMixtureModel(K=2))
]

data_x = np.linspace(-10, 10, 500)
data_t = np.linspace(300, 600, 500)

results = []

for name, model in models:
    model.init_param_uniform()
    if name == "TSDC":
        dx = data_t
    else:
        dx = data_x
        
    dy = model.predict(dx) * 10000 + np.random.normal(0, 10, 500)
    dy = np.clip(dy, 0, None)
    
    # -------------------------------------------------
    # Python (Incremental: 10 + 10 + 10 + 10 = 40)
    # -------------------------------------------------
    set_backend("python")
    py_times = []
    current_py_time = 0.0
    for i in range(4):
        start = time.time()
        try:
            model.sampling(dx, dy, method='adapted_em', trial=10, max_iter=100, stdout=False)
            dt = time.time() - start
            current_py_time += dt
            py_times.append(current_py_time)
        except Exception:
            py_times.append(-1)
            current_py_time += 0

    # -------------------------------------------------
    # Rust (Native: 10, 20, 30, 40) 
    # Rust is so fast that we can just measure directly
    # without doing incremental summation, to see thread scaling.
    # -------------------------------------------------
    set_backend("rust")
    rs_times = []
    for t in [10, 20, 30, 40]:
        start = time.time()
        model.sampling(dx, dy, method='adapted_em', trial=t, max_iter=100, stdout=False)
        rs_times.append(time.time() - start)
        
    for idx, t in enumerate([10, 20, 30, 40]):
        pt = py_times[idx]
        rt = rs_times[idx]
        sp = pt/rt if pt > 0 and rt > 0 else 0
        results.append({
            "Model": name,
            "Trial": t,
            "Python (s)": f"{pt:.3f}" if pt > 0 else "Error",
            "Rust (s)": f"{rt:.3f}",
            "Speedup": f"{sp:.1f}x" if sp > 0 else "N/A"
        })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
