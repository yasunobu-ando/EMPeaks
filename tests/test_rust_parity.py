"""Phase 1: Python backend vs Rust backend parity tests."""
import numpy as np
import pytest
from EMPeaks.GaussianMixture import GaussianMixtureModel
from EMPeaks.EMCore._backend import set_backend, get_backend


@pytest.fixture(autouse=True)
def reset_backend():
    """Restore backend to auto after each test."""
    yield
    set_backend("python")


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    x = np.linspace(-5, 5, 200)
    intensity = np.exp(-x ** 2) + 0.5 * np.exp(-(x - 2) ** 2)
    intensity += 0.01 * rng.random(len(x))
    intensity /= intensity.sum()
    return x, intensity


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def test_rust_backend_detected():
    set_backend("python")
    set_backend("rust")          # should not raise if installed
    assert get_backend() == "rust"


def test_python_backend_forced():
    set_backend("python")
    assert get_backend() == "python"


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        set_backend("invalid")


# ---------------------------------------------------------------------------
# Numerical parity: same initial params → LL must agree to atol=1e-5
# ---------------------------------------------------------------------------

def _fit_with_seed(x, intensity, backend, seed=0):
    set_backend(backend)
    m = GaussianMixtureModel(K=2, x_min=-5, x_max=5)
    np.random.seed(seed)
    m.init_param_uniform()
    # Capture initial params to ensure both paths start identically
    mu0    = [m.model[k].mu    for k in range(m.K)]
    sigma0 = [m.model[k].sigma for k in range(m.K)]
    pi0    = m.pi.copy()
    return m, mu0, sigma0, pi0


@pytest.mark.parametrize("K", [1, 2, 3])
def test_ll_parity(sample_data, K):
    x, intensity = sample_data

    # Python run
    set_backend("python")
    m_py = GaussianMixtureModel(K=K, x_min=-5, x_max=5)
    np.random.seed(7)
    m_py.init_param_uniform()
    mu0    = [m_py.model[k].mu    for k in range(K)]
    sigma0 = [m_py.model[k].sigma for k in range(K)]
    pi0    = m_py.pi.copy()
    r_py = m_py.fit(x, intensity, method="adapted_em", max_iter=2000, stdout=False)

    # Rust run — identical initial params
    set_backend("rust")
    m_rs = GaussianMixtureModel(K=K, x_min=-5, x_max=5)
    for k in range(K):
        m_rs.model[k].mu    = mu0[k]
        m_rs.model[k].sigma = sigma0[k]
    m_rs.pi = pi0.copy()
    r_rs = m_rs.fit(x, intensity, method="adapted_em", max_iter=2000, stdout=False)

    assert np.isclose(r_py["LL"], r_rs["LL"], atol=1e-5), (
        f"K={K}: LL mismatch  python={r_py['LL']:.8f}  rust={r_rs['LL']:.8f}"
    )


# ---------------------------------------------------------------------------
# Rust backend skips correctly for non-Gaussian (background != none)
# ---------------------------------------------------------------------------

def test_rust_falls_back_for_background(sample_data):
    x, intensity = sample_data
    set_backend("rust")
    m = GaussianMixtureModel(K=2, x_min=-5, x_max=5, background="uniform")
    # Should complete without error (falls back to Python path)
    r = m.fit(x, intensity, method="adapted_em", max_iter=200, stdout=False)
    assert "LL" in r


# ---------------------------------------------------------------------------
# Fallback: empeaks_rust_core absent → auto selects Python
# ---------------------------------------------------------------------------

def test_auto_fallback_to_python(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "empeaks_rust_core":
            raise ImportError("mocked absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from EMPeaks.EMCore import _backend
    _backend._BACKEND = None          # reset cache
    result = _backend.get_backend()
    assert result == "python"
    _backend._BACKEND = None          # restore for other tests


# ---------------------------------------------------------------------------
# Benchmark (informational, not a pass/fail test)
# ---------------------------------------------------------------------------

def test_benchmark(sample_data, capsys):
    import time
    x, intensity = sample_data
    times = {}

    for backend in ("python", "rust"):
        set_backend(backend)
        m = GaussianMixtureModel(K=2, x_min=-5, x_max=5)
        np.random.seed(0)
        m.init_param_uniform()
        t0 = time.perf_counter()
        m.fit(x, intensity, method="adapted_em", max_iter=3000, stdout=False)
        times[backend] = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n  python: {times['python']:.3f}s")
        print(f"    rust: {times['rust']:.3f}s")
        if times["python"] > 0:
            print(f"  speedup: {times['python'] / times['rust']:.1f}x")
