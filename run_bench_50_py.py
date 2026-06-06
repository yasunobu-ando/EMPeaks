import numpy as np
import time
from EMPeaks.EMCore._backend import set_backend
from EMPeaks.GaussianMixture import GaussianMixtureModel
from EMPeaks.PseudoVoigtMixture import PseudoVoigtMixtureModel
from EMPeaks.LorentzianMixture import LorentzianMixtureModel
from EMPeaks.DoniachSunjicMixture import DoniachSunjicMixtureModel
from EMPeaks.TSDCMixture import TSDCMixtureModel

models = [
    ("Gaussian", GaussianMixtureModel(K=2)),
    ("PseudoVoigt", PseudoVoigtMixtureModel(K=2)),
    ("Lorentzian", LorentzianMixtureModel(K=2)),
    ("DoniachSunjic", DoniachSunjicMixtureModel(K=2))
]

data_x = np.linspace(-10, 10, 500)

print("Model,Python (sec)")
for name, model in models:
    model.init_param_uniform()
    data_y = model.predict(data_x) * 10000 + np.random.normal(0, 10, 500)
    data_y = np.clip(data_y, 0, None)
    
    set_backend("python")
    start = time.time()
    try:
        model.sampling(data_x, data_y, method='adapted_em', trial=50, max_iter=3000, stdout=False)
        t_py = time.time() - start
        print(f"{name},{t_py:.3f}")
    except Exception as e:
        print(f"{name},ERROR {e}")

tsdc_model = TSDCMixtureModel(K=2)
data_t = np.linspace(300, 600, 500)
tsdc_model.init_param_uniform()
data_i = tsdc_model.predict(data_t) * 10000 + np.random.normal(0, 10, 500)
data_i = np.clip(data_i, 0, None)

set_backend("python")
start = time.time()
try:
    tsdc_model.sampling(data_t, data_i, method='adapted_em', trial=50, max_iter=3000, stdout=False)
    t_py = time.time() - start
    print(f"TSDC,{t_py:.3f}")
except Exception as e:
    print(f"TSDC,ERROR {e}")
