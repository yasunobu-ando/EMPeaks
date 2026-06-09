"""Phase 1, 2-①, 2-②, 2-③: Python backend vs Rust backend parity tests."""
import numpy as np
import pytest
from EMPeaks.GaussianMixture import GaussianMixtureModel
from EMPeaks.PseudoVoigtMixture import PseudoVoigtMixtureModel
from EMPeaks.TSDCMixture import TSDCMixtureModel
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


# ===========================================================================
# Phase 2-①: PseudoVoigt parity tests
# ===========================================================================

@pytest.fixture
def pv_sample_data():
    """Two overlapping PseudoVoigt-like peaks."""
    from scipy.stats import norm, cauchy
    rng = np.random.default_rng(99)
    x = np.linspace(-5, 5, 100)
    sigma = 0.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    hg = 0.25
    peak1 = 0.8 * norm.pdf(x, 0.0, sigma) + 0.2 * cauchy.pdf(x, 0.0, hg)
    peak2 = 0.7 * norm.pdf(x, 2.0, sigma) + 0.3 * cauchy.pdf(x, 2.0, hg)
    intensity = peak1 + 0.5 * peak2 + 0.02 * rng.random(len(x))
    intensity /= intensity.sum()
    return x, intensity


def _fit_pv(x, intensity, backend, x0_init, gamma_init, eta_init, pi_init, max_iter=100):
    set_backend(backend)
    m = PseudoVoigtMixtureModel(K=2, x_min=-5, x_max=5)
    for k in range(2):
        m.model[k].x0    = x0_init[k]
        m.model[k].gamma = gamma_init[k]
        m.model[k].eta   = eta_init[k]
    m.pi = pi_init.copy()
    return m.fit(x, intensity, method="adapted_em", max_iter=max_iter, stdout=False)


def test_parity_pseudo_voigt(pv_sample_data):
    """Python conditional_max と Rust mle_conditional_max の LL が atol=1e-5 で一致。"""
    x, intensity = pv_sample_data

    # Generate initial params once
    rng = np.random.default_rng(7)
    m_init = PseudoVoigtMixtureModel(K=2, x_min=-5, x_max=5)
    np.random.seed(7)
    m_init.init_param_uniform()
    x0_0    = [m_init.model[k].x0    for k in range(2)]
    gamma_0 = [m_init.model[k].gamma for k in range(2)]
    eta_0   = [m_init.model[k].eta   for k in range(2)]
    pi_0    = m_init.pi.copy()

    r_py = _fit_pv(x, intensity, "python", x0_0, gamma_0, eta_0, pi_0)
    r_rs = _fit_pv(x, intensity, "rust",   x0_0, gamma_0, eta_0, pi_0)

    assert np.isclose(r_py["LL"], r_rs["LL"], atol=1e-5), (
        f"PV LL mismatch: python={r_py['LL']:.8f}, rust={r_rs['LL']:.8f}"
    )


def test_pseudo_voigt_rust_falls_back_for_background(pv_sample_data):
    """background != 'none' のとき Python パスにフォールバックする。"""
    x, intensity = pv_sample_data
    set_backend("rust")
    m = PseudoVoigtMixtureModel(K=2, x_min=-5, x_max=5, background="uniform")
    r = m.fit(x, intensity, method="adapted_em", max_iter=50, stdout=False)
    assert "LL" in r


def test_benchmark_pseudo_voigt(pv_sample_data, capsys):
    """PseudoVoigt Python vs Rust ベンチマーク（情報のみ）。"""
    import time
    x, intensity = pv_sample_data
    times = {}

    np.random.seed(0)
    m_ref = PseudoVoigtMixtureModel(K=2, x_min=-5, x_max=5)
    m_ref.init_param_uniform()
    x0_0    = [m_ref.model[k].x0    for k in range(2)]
    gamma_0 = [m_ref.model[k].gamma for k in range(2)]
    eta_0   = [m_ref.model[k].eta   for k in range(2)]
    pi_0    = m_ref.pi.copy()

    for backend in ("python", "rust"):
        t0 = time.perf_counter()
        _fit_pv(x, intensity, backend, x0_0, gamma_0, eta_0, pi_0, max_iter=200)
        times[backend] = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n  PV python: {times['python']:.3f}s")
        print(f"  PV rust:   {times['rust']:.3f}s")
        if times["rust"] > 0:
            print(f"  PV speedup: {times['python'] / times['rust']:.1f}x")


# ===========================================================================
# Phase 2-②: Lorentzian parity tests
# ===========================================================================

from EMPeaks.LorentzianMixture import LorentzianMixtureModel


@pytest.fixture
def lo_sample_data():
    """Two overlapping Lorentzian peaks."""
    from scipy.stats import cauchy
    rng = np.random.default_rng(77)
    x = np.linspace(-5, 5, 120)
    peak1 = cauchy.pdf(x, loc=0.0,  scale=0.4)
    peak2 = cauchy.pdf(x, loc=1.8,  scale=0.6)
    intensity = peak1 + 0.6 * peak2 + 0.01 * rng.random(len(x))
    intensity /= intensity.sum()
    return x, intensity


def _fit_lo(x, intensity, backend, x0_init, gamma_init, pi_init, max_iter=100):
    set_backend(backend)
    m = LorentzianMixtureModel(K=2, x_min=-5, x_max=5)
    for k in range(2):
        m.model[k].x0    = x0_init[k]
        m.model[k].gamma = gamma_init[k]
    m.pi = pi_init.copy()
    return m.fit(x, intensity, method="adapted_em", max_iter=max_iter, stdout=False)


def test_parity_lorentzian(lo_sample_data):
    """Lorentzian Python と Rust EM の LL が atol=1e-2 で一致。"""
    x, intensity = lo_sample_data

    np.random.seed(13)
    m_init = LorentzianMixtureModel(K=2, x_min=-5, x_max=5)
    m_init.init_param_uniform()
    x0_0    = [m_init.model[k].x0    for k in range(2)]
    gamma_0 = [m_init.model[k].gamma for k in range(2)]
    pi_0    = m_init.pi.copy()

    r_py = _fit_lo(x, intensity, "python", x0_0, gamma_0, pi_0)
    r_rs = _fit_lo(x, intensity, "rust",   x0_0, gamma_0, pi_0)

    assert np.isclose(r_py["LL"], r_rs["LL"], atol=1e-2), (
        f"Lorentzian LL mismatch: python={r_py['LL']:.8f}, rust={r_rs['LL']:.8f}"
    )


def test_lorentzian_rust_falls_back_for_background(lo_sample_data):
    """background != 'none' のとき Python パスにフォールバックする。"""
    x, intensity = lo_sample_data
    set_backend("rust")
    m = LorentzianMixtureModel(K=2, x_min=-5, x_max=5, background="uniform")
    r = m.fit(x, intensity, method="adapted_em", max_iter=50, stdout=False)
    assert "LL" in r


def test_benchmark_lorentzian(lo_sample_data, capsys):
    """Lorentzian Python vs Rust ベンチマーク（情報のみ）。"""
    import time
    x, intensity = lo_sample_data
    times = {}

    np.random.seed(0)
    m_ref = LorentzianMixtureModel(K=2, x_min=-5, x_max=5)
    m_ref.init_param_uniform()
    x0_0    = [m_ref.model[k].x0    for k in range(2)]
    gamma_0 = [m_ref.model[k].gamma for k in range(2)]
    pi_0    = m_ref.pi.copy()

    for backend in ("python", "rust"):
        t0 = time.perf_counter()
        _fit_lo(x, intensity, backend, x0_0, gamma_0, pi_0, max_iter=200)
        times[backend] = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n  LO python: {times['python']:.3f}s")
        print(f"  LO rust:   {times['rust']:.3f}s")
        if times["rust"] > 0:
            print(f"  LO speedup: {times['python'] / times['rust']:.1f}x")


# ===========================================================================
# Phase 2-③: DoniachSunjic parity tests
# ===========================================================================

from EMPeaks.DoniachSunjicMixture import DoniachSunjicMixtureModel


@pytest.fixture
def ds_sample_data():
    """Two overlapping DoniachSunjic-like peaks (Cauchy data)."""
    from scipy.stats import cauchy
    rng = np.random.default_rng(55)
    x = np.linspace(-5, 5, 120)
    peak1 = cauchy.pdf(x, loc=0.0, scale=0.5)
    peak2 = cauchy.pdf(x, loc=2.0, scale=0.7)
    intensity = peak1 + 0.5 * peak2 + 0.01 * rng.random(len(x))
    intensity /= intensity.sum()
    return x, intensity


def _fit_ds(x, intensity, backend, x0_init, gamma_init, alpha_init, pi_init, max_iter=100):
    set_backend(backend)
    m = DoniachSunjicMixtureModel(K=2, x_min=-5, x_max=5)
    for k in range(2):
        m.model[k].x0    = x0_init[k]
        m.model[k].gamma = gamma_init[k]
        m.model[k].alpha = alpha_init[k]
    m.pi = pi_init.copy()
    return m.fit(x, intensity, method="adapted_em", max_iter=max_iter, stdout=False)


def test_parity_doniach_sunjic(ds_sample_data):
    """DoniachSunjic Python と Rust EM の LL が atol=1e-2 で一致。"""
    x, intensity = ds_sample_data

    np.random.seed(17)
    m_init = DoniachSunjicMixtureModel(K=2, x_min=-5, x_max=5)
    m_init.init_param_uniform()
    x0_0    = [m_init.model[k].x0    for k in range(2)]
    gamma_0 = [m_init.model[k].gamma for k in range(2)]
    alpha_0 = [m_init.model[k].alpha for k in range(2)]
    pi_0    = m_init.pi.copy()

    r_py = _fit_ds(x, intensity, "python", x0_0, gamma_0, alpha_0, pi_0)
    r_rs = _fit_ds(x, intensity, "rust",   x0_0, gamma_0, alpha_0, pi_0)

    assert np.isclose(r_py["LL"], r_rs["LL"], atol=1e-2), (
        f"DS LL mismatch: python={r_py['LL']:.8f}, rust={r_rs['LL']:.8f}"
    )


def test_doniach_sunjic_rust_falls_back_for_background(ds_sample_data):
    """background != 'none' のとき Python パスにフォールバックする。"""
    x, intensity = ds_sample_data
    set_backend("rust")
    m = DoniachSunjicMixtureModel(K=2, x_min=-5, x_max=5, background="uniform")
    r = m.fit(x, intensity, method="adapted_em", max_iter=50, stdout=False)
    assert "LL" in r


def test_benchmark_doniach_sunjic(ds_sample_data, capsys):
    """DoniachSunjic Python vs Rust ベンチマーク（情報のみ）。"""
    import time
    x, intensity = ds_sample_data
    times = {}

    np.random.seed(0)
    m_ref = DoniachSunjicMixtureModel(K=2, x_min=-5, x_max=5)
    m_ref.init_param_uniform()
    x0_0    = [m_ref.model[k].x0    for k in range(2)]
    gamma_0 = [m_ref.model[k].gamma for k in range(2)]
    alpha_0 = [m_ref.model[k].alpha for k in range(2)]
    pi_0    = m_ref.pi.copy()

    for backend in ("python", "rust"):
        t0 = time.perf_counter()
        _fit_ds(x, intensity, backend, x0_0, gamma_0, alpha_0, pi_0, max_iter=200)
        times[backend] = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n  DS python: {times['python']:.3f}s")
        print(f"  DS rust:   {times['rust']:.3f}s")
        if times["rust"] > 0:
            print(f"  DS speedup: {times['python'] / times['rust']:.1f}x")


# ---------------------------------------------------------------------------
# Phase 2-④: TSDCMixture parity tests
# ---------------------------------------------------------------------------

from EMPeaks.TSDCMixture import TSDCMixtureModel

@pytest.fixture
def tsdc_sample_data():
    """Two overlapping TSDC-like peaks."""
    rng = np.random.default_rng(42)
    T = np.linspace(300, 800, 200)
    # Generate some artificial TSDC data using the model itself
    m = TSDCMixtureModel(K=2, T_min=300, T_max=800)
    m.set_param(K=2, Tp=[400.0, 600.0], Ea=[0.5, 1.0], N=[0.6, 0.4])
    intensity = m.predict(T)
    intensity += 0.001 * rng.random(len(T))
    intensity /= intensity.sum()
    return T, intensity

def test_parity_tsdc(tsdc_sample_data):
    """TSDC Python と Rust EM の LL が atol=1e-2 で一致。"""
    T, intensity = tsdc_sample_data
    
    m_init = TSDCMixtureModel(K=2, T_min=300, T_max=800)
    m_init.set_param(K=2, Tp=[350.0, 650.0], Ea=[0.4, 0.9], N=[0.5, 0.5])
    
    set_backend("python")
    m_py = TSDCMixtureModel(K=2, T_min=300, T_max=800)
    m_py.set_param(**m_init.export_param())
    info_py = m_py.fit(T, intensity, max_iter=20)
    
    set_backend("rust")
    m_rs = TSDCMixtureModel(K=2, T_min=300, T_max=800)
    m_rs.set_param(**m_init.export_param())
    info_rs = m_rs.fit(T, intensity, max_iter=20)
    
    assert np.isclose(info_py["LL"], info_rs["LL"], atol=1e-2)

def test_benchmark_tsdc(capsys):
    """TSDC Python vs Rust ベンチマーク（1000点, max_iter=3000）。"""
    rng = np.random.default_rng(42)
    T = np.linspace(300, 800, 1000)
    m = TSDCMixtureModel(K=2, T_min=300, T_max=800)
    m.set_param(K=2, Tp=[400.0, 600.0], Ea=[0.5, 1.0], N=[0.6, 0.4])
    intensity = m.predict(T)
    intensity += 0.001 * rng.random(len(T))
    intensity /= intensity.sum()
    
    m_ref = TSDCMixtureModel(K=2, T_min=300, T_max=800)
    m_ref.set_param(K=2, Tp=[350.0, 650.0], Ea=[0.4, 0.9], N=[0.5, 0.5])
    
    import time
    
    set_backend("python")
    m_py = TSDCMixtureModel(K=2, T_min=300, T_max=800)
    m_py.set_param(**m_ref.export_param())
    t0 = time.perf_counter()
    m_py.fit(T, intensity, max_iter=3000)
    t_py = time.perf_counter() - t0
    
    set_backend("rust")
    m_rs = TSDCMixtureModel(K=2, T_min=300, T_max=800)
    m_rs.set_param(**m_ref.export_param())
    t0 = time.perf_counter()
    m_rs.fit(T, intensity, max_iter=3000)
    t_rs = time.perf_counter() - t0
    
    with capsys.disabled():
        print(f"\n  TSDC python: {t_py:.3f}s")
        print(f"  TSDC rust:   {t_rs:.3f}s")
        if t_rs > 0:
            print(f"  TSDC speedup: {t_py/t_rs:.1f}x")
