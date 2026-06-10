import numpy as np
import matplotlib.pyplot as plt
from EMPeaks.GaussianMixture._gmm import GaussianMixtureModel

def test_bspline():
    np.random.seed(42)
    x = np.linspace(-10, 10, 200)
    # create some gaussian peaks
    intensity = 100 * np.exp(-(x - 2)**2 / 2) + 50 * np.exp(-(x + 3)**2 / 8)
    # add some background (linear + noise)
    intensity += 10 + 0.5 * x + np.random.normal(0, 1, x.shape)
    
    model = GaussianMixtureModel(K=2, background='b_spline', degree_spline=3, n_section=5)
    print("model initialized")
    model.fit(x, intensity, method='adapted_em', max_iter=10, stdout=True)
    print("fit complete")
    
    fig = model.plot(x, intensity, show=False)
    fig.savefig('bspline_test.png')
    
if __name__ == "__main__":
    test_bspline()
