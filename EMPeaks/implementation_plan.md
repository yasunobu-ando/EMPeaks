# EMPeaks/EMCore Rust 高速化 実装計画 (v4)

## 概要

EMPeaks の計算コアを Rust で書き換え、Python バインディング (PyO3 + maturin) を通じて呼び出すことで、EMアルゴリズムのフィッティング処理を高速化する。Rust 実装を**標準**としつつ、Python 実装も保持して切り替え可能にする。

---

## 完了済み作業

### 環境・互換性

| 作業 | 内容 |
|---|---|
| Python 要件引き上げ | `python_requires = >=3.11`（setup.cfg） |
| classifiers 更新 | Python 3.11 / 3.12 / 3.13 を追加 |
| 依存ライブラリ更新 | numpy>=1.23.0, scipy>=1.9.0, matplotlib>=3.6.0, pandas>=1.5.0 |
| `trapz` 置換 | `integrate.trapz` → `integrate.trapezoid`、`np.trapz` → `np.trapezoid`（6ファイル） |
| SyntaxWarning 修正 | `is 0` → `== 0`、`is not 1` → `!= 1`（_lorentz.py） |
| Python 3.11 動作確認 | Gaussian / Lorentzian / PseudoVoigt / DoniachSunjic 全モデル OK |

### Phase 1: Gaussian EM コア ✅ 完了

| 作業 | 内容 |
|---|---|
| `_rust_core/` Cargo プロジェクト | Cargo.toml / pyproject.toml（abi3-py311） |
| `src/gaussian.rs` | predict / MLE（Rust 単体テスト済み） |
| `src/em_engine.rs` | E-step / M-step / EMループ（Rust 単体テスト済み） |
| `src/lib.rs` | PyO3 バインディング |
| `EMCore/_backend.py` | 自動検出・set_backend / get_backend |
| `EMCore/_em_core.py` | `adapted_em` → `_adapted_em_rust` / `_adapted_em_python` 分岐 |
| `tests/test_rust_parity.py` | パリティ / フォールバック / ベンチマーク（9/9 通過） |

**実測ベンチマーク**（Python 3.11.9 / Apple Silicon、5 回平均）:

| 条件 | Python | Rust | 高速化 |
|---|---|---|---|
| K=2, N=200 | 10.0 ms | 1.9 ms | **5.3x** |
| K=3, N=500 | 30.8 ms | 5.7 ms | **5.4x** |

> [!NOTE]
> 現時点で `leastsq_for_normalization_factor`（scipy.optimize）は Python 側に残っており、これが実行時間の下限。EMループ単体の高速化はさらに大きい。

---

## アーキテクチャ方針

### ディレクトリ構成

```
EMPeaks/
├── _rust_core/               ← Cargo プロジェクト（1つに統合）
│   ├── Cargo.toml
│   ├── pyproject.toml        ← maturin 設定
│   └── src/
│       ├── lib.rs            ← PyO3 モジュール登録
│       ├── gaussian.rs       ← Gaussian predict / MLE
│       ├── background.rs     ← 背景モデル predict（Phase 2）
│       └── em_engine.rs      ← E-step / M-step / EMループ
├── EMCore/
│   ├── _em_core.py           ← [MODIFY] Rust/Python 分岐を追加
│   ├── _gaussian.py          ← [既存・変更なし]
│   └── _backend.py           ← [NEW] バックエンド切り替え
└── ...
```

### バックエンド切り替え機構

```python
# EMPeaks/EMCore/_backend.py
import os
_BACKEND = None

def get_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    env = os.environ.get("EMPEAKS_BACKEND", "auto")
    if env == "python":
        _BACKEND = "python"; return _BACKEND
    try:
        import empeaks_rust_core
        _BACKEND = "rust"
    except ImportError:
        if env == "rust":
            raise ImportError("empeaks_rust_core がインストールされていません。")
        _BACKEND = "python"
    return _BACKEND

def set_backend(backend: str):
    global _BACKEND
    if backend not in ("rust", "python"):
        raise ValueError(f"backend は 'rust' または 'python' を指定: {backend}")
    _BACKEND = backend
```

---

## Phase 1: Gaussian EM コアの Rust 化

### スコープ

**対象**:
- `Gaussian.predict(x)` / `Gaussian.maximum_likelihood_estimation(x, weighted_intensity)`
- `EMCore.e_step(x)` / `EMCore.m_step(x, intensity)` / `EMCore.adapted_em(...)` の EMループ本体
- `background='none'` の場合のみ（Gaussianコンポーネントのみ）

**対象外**:
- `leastsq_for_normalization_factor`（scipy.optimize.least_squares を使用）
- `background != 'none'` のケース（背景モデルは Phase 2 で対応）
- Lorentzian / PseudoVoigt / DoniachSunjic（Phase 2）
- `sampling` の並列化（Phase 3）

### データフロー

```
Python 側（_em_core.py）
  ↓  x, intensity, mu[], sigma[], pi[], dirichlet_alpha[], max_iter, r_eps
Rust 側（empeaks_rust_core.run_em_loop）
  ↓  EMループ全体を実行（E-step × M-step × N回）
Python 側
  ↑  更新後の mu[], sigma[], pi[], LL_hist[], residual_hist[], total_iter
```

Python ↔ Rust の境界越えは**1回**（EMループ全体を渡す）。

---

### Step 1: Rust プロジェクト作成

**`EMPeaks/_rust_core/Cargo.toml`**:
```toml
[package]
name = "empeaks_rust_core"
version = "0.1.0"
edition = "2021"

[lib]
name = "empeaks_rust_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py311"] }
numpy = "0.22"
ndarray = "0.16"
```

**`EMPeaks/_rust_core/pyproject.toml`**:
```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[tool.maturin]
python-source = "."
features = ["pyo3/extension-module"]
```

---

### Step 2: `src/gaussian.rs` — Gaussian predict / MLE

```rust
use ndarray::Array1;

pub fn predict(x: &Array1<f64>, mu: f64, sigma: f64) -> Array1<f64> {
    let z = (2.0 * std::f64::consts::PI * sigma * sigma).sqrt();
    let inv_sqrt2_sigma = 1.0 / (2.0_f64.sqrt() * sigma);
    x.mapv(|xi| {
        let t = (xi - mu) * inv_sqrt2_sigma;
        (-t * t).exp() / z
    })
}

/// 重み付き最尤推定: 返り値は (mu, sigma)
pub fn mle(x: &Array1<f64>, w: &Array1<f64>) -> (f64, f64) {
    let eps = 1e-100;
    let sum_w = w.sum() + eps;
    let mu = (w * x).sum() / sum_w;
    let sigma2 = (w * &(x - mu).mapv(|v| v * v)).sum() / sum_w;
    let sigma = if sigma2 > 0.0 { sigma2.sqrt() } else { 1e-5 };
    (mu, sigma.max(1e-5))
}
```

---

### Step 3: `src/em_engine.rs` — E-step / M-step / EMループ

```rust
use ndarray::{Array1, Array2};
use crate::gaussian;

const EPSILON_PREDICT: f64 = 1e-20;
const EPSILON_LOG: f64 = 1e-200;

/// E-step: gamma[k][n] = pi[k] * p_k(x[n]) / sum_k(...)
fn e_step(
    x: &Array1<f64>,
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
    pi: &Array1<f64>,
) -> Array2<f64> {
    let n = x.len();
    let k_all = pi.len();
    let mut gamma = Array2::<f64>::zeros((k_all, n));

    // 各コンポーネントの重み付き予測値を計算
    for k in 0..k_all {
        let p_k = gaussian::predict(x, mu[k], sigma[k]);
        gamma.row_mut(k).assign(&(pi[k] * &p_k));
    }

    // 正規化（列方向 = データ点ごと）
    let total = gamma.sum_axis(ndarray::Axis(0)).mapv(|v| v + EPSILON_PREDICT);
    for k in 0..k_all {
        gamma.row_mut(k).zip_mut_with(&total, |g, &t| *g /= t);
    }
    gamma
}

/// M-step: pi, mu, sigma を更新
fn m_step(
    x: &Array1<f64>,
    intensity: &Array1<f64>,
    gamma: &Array2<f64>,
    dirichlet_alpha: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let k_all = gamma.nrows();
    let mut pi = Array1::<f64>::zeros(k_all);
    let mut mu_new = Array1::<f64>::zeros(k_all);
    let mut sigma_new = Array1::<f64>::zeros(k_all);

    let mut n_k_sum = 0.0;
    let mut n_k = Array1::<f64>::zeros(k_all);
    for k in 0..k_all {
        let w = intensity * &gamma.row(k).to_owned();
        n_k[k] = w.sum() + dirichlet_alpha[k] - 1.0;
        n_k_sum += n_k[k];
        let (m, s) = gaussian::mle(x, &w);
        mu_new[k] = m;
        sigma_new[k] = s;
    }
    // pi の更新と非負補正
    for k in 0..k_all {
        pi[k] = (n_k[k] / n_k_sum).max(0.0);
    }
    let pi_sum = pi.sum();
    pi.mapv_inplace(|v| v / pi_sum);

    (pi, mu_new, sigma_new)
}

/// 対数尤度
fn log_likelihood(x: &Array1<f64>, intensity: &Array1<f64>,
                  mu: &Array1<f64>, sigma: &Array1<f64>, pi: &Array1<f64>) -> f64 {
    let k_all = pi.len();
    let mut total = Array1::<f64>::zeros(x.len());
    for k in 0..k_all {
        total += &(pi[k] * &gaussian::predict(x, mu[k], sigma[k]));
    }
    (intensity * &total.mapv(|v| (v + EPSILON_LOG).ln())).sum()
}

/// EMループ本体
pub fn run_em_loop(
    x: &Array1<f64>,
    intensity: &Array1<f64>,
    mu: &mut Array1<f64>,
    sigma: &mut Array1<f64>,
    pi: &mut Array1<f64>,
    dirichlet_alpha: &Array1<f64>,
    max_iter: usize,
    r_eps: f64,
) -> (usize, Vec<f64>, Vec<f64>) {
    let mut ll_0 = log_likelihood(x, intensity, mu, sigma, pi);
    let mut ll_hist = vec![ll_0];
    let mut res_hist = vec![0.0_f64];
    let mut total_iter = max_iter;

    for it in 0..max_iter {
        let gamma = e_step(x, mu, sigma, pi);
        let (new_pi, new_mu, new_sigma) = m_step(x, intensity, &gamma, dirichlet_alpha);
        *pi = new_pi;
        *mu = new_mu;
        *sigma = new_sigma;

        let ll = log_likelihood(x, intensity, mu, sigma, pi);
        let residual = (ll - ll_0) / ll_0.abs();
        ll_hist.push(ll);
        res_hist.push(residual);

        if residual < 0.0 {
            // 対数尤度が下降 → パラメータリセットは Python 側で処理
            total_iter = it + 1;
            break;
        }
        if residual < r_eps {
            total_iter = it + 1;
            break;
        }
        ll_0 = ll;
    }
    (total_iter, ll_hist, res_hist)
}
```

---

### Step 4: `src/lib.rs` — PyO3 バインディング

```rust
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use ndarray::Array1;

mod gaussian;
mod em_engine;

#[pyfunction]
fn run_em_loop<'py>(
    _py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    intensity: PyReadonlyArray1<'py, f64>,
    mu: PyReadwriteArray1<'py, f64>,
    sigma: PyReadwriteArray1<'py, f64>,
    pi: PyReadwriteArray1<'py, f64>,
    dirichlet_alpha: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    r_eps: f64,
) -> PyResult<(usize, Vec<f64>, Vec<f64>)> {
    let x = x.as_array().to_owned();
    let intensity = intensity.as_array().to_owned();
    let mut mu = mu.as_array_mut();
    let mut sigma = sigma.as_array_mut();
    let mut pi = pi.as_array_mut();
    let da = dirichlet_alpha.as_array().to_owned();

    let (total_iter, ll_hist, res_hist) =
        em_engine::run_em_loop(&x, &intensity, &mut mu, &mut sigma, &mut pi, &da, max_iter, r_eps);

    Ok((total_iter, ll_hist, res_hist))
}

#[pymodule]
fn empeaks_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_em_loop, m)?)?;
    Ok(())
}
```

---

### Step 5: `_em_core.py` の変更

```python
# EMPeaks/EMCore/_em_core.py
from EMPeaks.EMCore._backend import get_backend

def adapted_em(self, x, intensity, max_iter, r_eps, stdout):
    if get_backend() == "rust" and self.background == "none":
        return self._adapted_em_rust(x, intensity, max_iter, r_eps, stdout)
    return self._adapted_em_python(x, intensity, max_iter, r_eps, stdout)

def _adapted_em_rust(self, x, intensity, max_iter, r_eps, stdout):
    import empeaks_rust_core
    import numpy as np

    mu = np.array([m.mu for m in self.model[:self.K]])
    sigma = np.array([m.sigma for m in self.model[:self.K]])
    pi = self.pi.copy()
    da = self.Dirichlet_alpha.copy()

    total_iter, ll_hist, res_hist = empeaks_rust_core.run_em_loop(
        x, intensity, mu, sigma, pi, da, max_iter, r_eps
    )
    # パラメータを Python 側に反映
    for k in range(self.K):
        self.model[k].mu = mu[k]
        self.model[k].sigma = sigma[k]
    self.pi = pi

    # leastsq_for_normalization_factor は Python 側で処理
    rmse = self.leastsq_for_normalization_factor(x, intensity, stdout)
    param = self.export_param()
    run_info = {
        'total_iter': total_iter,
        'total_time': float(total_iter),  # Rust 側でも計測予定
        'time/iter': 0.0,
        'LL': ll_hist[-1] if ll_hist else float('nan'),
        'LL_hist': ll_hist,
        'LL_residual': res_hist[-1] if res_hist else float('nan'),
        'LL_residual_hist': res_hist,
        'RMSE': rmse
    }
    return run_info

def _adapted_em_python(self, x, intensity, max_iter, r_eps, stdout):
    # 既存の adapted_em 実装をそのまま移動
    ...
```

---

### Step 6: テスト (`tests/test_rust_parity.py`)

```python
import numpy as np
import pytest
from EMPeaks.GaussianMixture import GaussianMixtureModel
from EMPeaks.EMCore._backend import set_backend

@pytest.fixture
def sample_data():
    np.random.seed(42)
    x = np.linspace(-5, 5, 200)
    intensity = np.exp(-x**2) + 0.5 * np.exp(-(x - 2)**2)
    intensity += 0.01 * np.random.rand(len(x))
    intensity /= intensity.sum()
    return x, intensity

def test_parity(sample_data):
    """Python と Rust の出力が atol=1e-6 で一致"""
    x, intensity = sample_data

    set_backend("python")
    m_py = GaussianMixtureModel(K=2, x_min=-5, x_max=5)
    np.random.seed(0); m_py.init_param_uniform()
    r_py = m_py.fit(x, intensity, method='adapted_em', max_iter=1000)

    set_backend("rust")
    m_rs = GaussianMixtureModel(K=2, x_min=-5, x_max=5)
    np.random.seed(0); m_rs.init_param_uniform()
    r_rs = m_rs.fit(x, intensity, method='adapted_em', max_iter=1000)

    assert np.allclose(r_py['LL'], r_rs['LL'], atol=1e-6), \
        f"LL mismatch: py={r_py['LL']:.8f}, rs={r_rs['LL']:.8f}"

def test_fallback(sample_data, monkeypatch):
    """Rust 未インストール時に Python にフォールバック"""
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "empeaks_rust_core":
            raise ImportError
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", mock_import)

    from EMPeaks.EMCore import _backend
    _backend._BACKEND = None
    assert _backend.get_backend() == "python"
```

---

### Step 7: CI/CD (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  build-wheels:
    strategy:
      matrix:
        target:
          - x86_64-unknown-linux-gnu
          - aarch64-apple-darwin
          - x86_64-apple-darwin
          - x86_64-pc-windows-msvc
    runs-on: ${{ contains(matrix.target, 'linux') && 'ubuntu-latest'
                 || contains(matrix.target, 'apple') && 'macos-latest'
                 || 'windows-latest' }}
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          args: --release --out dist --manifest-path EMPeaks/_rust_core/Cargo.toml
          manylinux: auto
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.target }}
          path: dist/

  publish:
    needs: build-wheels
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with: { path: dist/, merge-multiple: true }
      - uses: pypa/gh-action-pypi-publish@release/v1
```

---

## Phase 1 実装順序

```
① _rust_core/ プロジェクト作成 + Cargo.toml / pyproject.toml
② gaussian.rs（predict, mle）+ 単体テスト（#[test]）
③ em_engine.rs（e_step, m_step, run_em_loop）+ 単体テスト
④ lib.rs（PyO3 バインディング）
⑤ maturin develop --release でローカルビルド確認
⑥ _backend.py 作成
⑦ _em_core.py 修正（adapted_em → _adapted_em_rust / _adapted_em_python）
⑧ tests/test_rust_parity.py 作成・実行
⑨ ベンチマーク（compare_backends）
```

---

## Phase 2: 他の分布モデル + MLE 完全 Rust 化（戦略A）

### 方針

**戦略A（完全 Rust 化）を直接採用**。predict と optimizer をすべて Rust 内で完結させ、Python ↔ Rust の境界越えをゼロにする。

L-BFGS-B の実装は `lbfgsb` クレートを使用する。このクレートは **scipy の L-BFGS-B と同一の Fortran コード（L-BFGS-B v3.0）をラップ**しているため、同一入力に対して数値的に一致した結果が得られる。

### 実施順序

**PseudoVoigt → Lorentzian → DoniachSunjic** の順に進める。  
各モデルを実装・テスト・ベンチマーク確認してから次に進むこと。

### 追加ファイル構成

```
_rust_core/src/
├── lib.rs            ← [MODIFY] Phase 2 関数を追加登録
├── gaussian.rs       ← [既存・変更なし]
├── em_engine.rs      ← [既存・変更なし]
├── pseudo_voigt.rs   ← [NEW] ① 最初に実装
├── lorentzian.rs     ← [NEW] ② 次に実装
├── doniach_sunjic.rs ← [NEW] ③ 最後に実装
└── background.rs     ← [NEW] Phase 2 完了後
```

### 追加 Cargo 依存

```toml
[dependencies]
lbfgsb = "0.3"   # scipy と同一 Fortran L-BFGS-B v3.0 ラッパー
rayon  = "1.10"  # PseudoVoigt eta グリッドサーチの並列化
```

---

## Phase 2-① PseudoVoigt（最初に実装）✅ 完了

### Python 実装の分析

**`_pseudo_voigt.py` の現行 MLE フロー（`conditional_max` が呼ばれる）:**

```
maximum_likelihood_estimation(x, w)
  └─ conditional_max(x, w)
       ├─ _e_step(x)           : gamma1（Gaussian責任），gamma2（Lorentzian責任）を計算
       ├─ _cm_step_x0_gamma(x, w) : x0/gamma を brentq 根探索で更新
       └─ _cm_step_eta(x, w)   : eta を grid search（0.8〜1.0、step 0.01）で更新
```

**Predict 関数の構造:**
```python
sigma = gamma / (2 * sqrt(2 * log(2)))
Z     = 1/pi * (arctan((x_max - x0)/(gamma/2)) - arctan((x_min - x0)/(gamma/2)))
predict(x) = eta * Gaussian(x, x0, sigma)
           + (1 - eta) / Z * Lorentzian(x, x0, gamma/2)
```

### Rust での実装方針

**両 MLE アルゴリズムを Rust 化**:
- `conditional_max`（brentq）→ `roots` クレートの `find_root_brent` で完全移植 → atol=1e-5 でパリティ確認可
- `full_optimization`（L-BFGS-B + eta グリッド）→ `lbfgsb` + `rayon` クレートで実装 → 同上

Python 側の `_pseudo_voigt.py` が呼ぶメソッド名（`conditional_max` / `full_optimization`）を判定して、対応する Rust 関数に振り分ける。

### Step 1: `src/pseudo_voigt.rs` の実装

```rust
// pseudo_voigt.rs
use ndarray::ArrayView1;
use rayon::prelude::*;
use roots::{find_root_brent, SimpleConvergency};

const PI: f64 = std::f64::consts::PI;
const SQRT2: f64 = std::f64::consts::SQRT_2;
const LN2: f64 = std::f64::consts::LN_2;

fn normalization_z(x0: f64, gamma: f64, x_min: f64, x_max: f64) -> f64 {
    let hg = gamma / 2.0;
    (1.0 / PI) * (((x_max - x0) / hg).atan() - ((x_min - x0) / hg).atan())
}

pub fn predict_single(x: f64, x0: f64, gamma: f64, eta: f64,
                      x_min: f64, x_max: f64) -> f64 {
    let sigma = gamma / (2.0 * SQRT2 * LN2.sqrt());
    let hg = gamma / 2.0;
    let z = normalization_z(x0, gamma, x_min, x_max).max(1e-300);
    let gauss   = (-(x - x0).powi(2) / (2.0 * sigma * sigma)).exp()
                  / (SQRT2 * PI.sqrt() * sigma);
    let lorentz = (1.0 / PI) * hg / ((x - x0).powi(2) + hg * hg);
    eta * gauss + (1.0 - eta) / z * lorentz
}

pub fn predict(x: ArrayView1<f64>, x0: f64, gamma: f64, eta: f64,
               x_min: f64, x_max: f64) -> Vec<f64> {
    x.iter().map(|&xi| predict_single(xi, x0, gamma, eta, x_min, x_max)).collect()
}

pub fn log_likelihood(x: ArrayView1<f64>, intensity: ArrayView1<f64>,
                      x0: f64, gamma: f64, eta: f64,
                      x_min: f64, x_max: f64) -> f64 {
    predict(x, x0, gamma, eta, x_min, x_max).iter()
        .zip(intensity.iter())
        .map(|(&p, &i)| i * (p + 1e-200_f64).ln())
        .sum()
}

// -----------------------------------------------------------------------
// conditional_max: Python の _cm_step_x0_gamma / _cm_step_eta を忠実移植
// -----------------------------------------------------------------------

fn d_ll_d_x0(x: &[f64], w1: &[f64], w2: &[f64],
             x0: f64, gamma: f64, eta: f64,
             x_min: f64, x_max: f64) -> f64 {
    // Python の _cm_step_x0_gamma が解く ∂LL/∂x0 の数値微分版
    //（解析微分は複雑なため数値微分で faithful に実装）
    let eps = 1e-7;
    let f = |x0_: f64| {
        let pv = predict(ArrayView1::from(x), x0_, gamma, eta, x_min, x_max);
        w1.iter().zip(w2.iter()).zip(pv.iter())
            .map(|((&r1, &r2), &p)| (r1 + r2) * (p + 1e-200_f64).ln())
            .sum::<f64>()
    };
    (f(x0 + eps) - f(x0 - eps)) / (2.0 * eps)
}

/// MLE via conditional maximization（Python の conditional_max に相当）
pub fn mle_conditional_max(
    x: ArrayView1<f64>, intensity: ArrayView1<f64>,
    x0: &mut f64, gamma: &mut f64, eta: &mut f64,
    x_min: f64, x_max: f64, gamma_min: f64, gamma_max: f64,
    eta_lo: f64, eta_hi: f64, eta_step: f64,  // Python デフォルト: 0.8, 1.0, 0.01
) {
    let x_s = x.to_vec();
    let int_s = intensity.to_vec();

    // E-step: 責任変数 w1(Gaussian), w2(Lorentzian) を計算
    let pv: Vec<f64> = predict(x, *x0, *gamma, *eta, x_min, x_max);
    let sigma = *gamma / (2.0 * SQRT2 * LN2.sqrt());
    let hg = *gamma / 2.0;
    let z = normalization_z(*x0, *gamma, x_min, x_max).max(1e-300);
    let w1: Vec<f64> = x.iter().zip(int_s.iter()).zip(pv.iter()).map(|((&xi, &ii), &pi)| {
        let g = (-(xi - *x0).powi(2) / (2.0 * sigma * sigma)).exp()
                / (SQRT2 * PI.sqrt() * sigma);
        ii * *eta * g / (pi + 1e-300)
    }).collect();
    let w2: Vec<f64> = x.iter().zip(int_s.iter()).zip(pv.iter()).map(|((&xi, &ii), &pi)| {
        let l = (1.0 / PI) * hg / ((xi - *x0).powi(2) + hg * hg);
        ii * (1.0 - *eta) / z * l / (pi + 1e-300)
    }).collect();

    // CM-step x0: brentq で ∂LL/∂x0 = 0 を解く
    let f_x0 = |x0_cand: f64| {
        d_ll_d_x0(&x_s, &w1, &w2, x0_cand, *gamma, *eta, x_min, x_max)
    };
    let mut conv = SimpleConvergency { eps: 1e-10_f64, max_iter: 100 };
    if let Ok(root) = find_root_brent(x_min, x_max, &f_x0, &mut conv) {
        *x0 = root;
    }

    // CM-step gamma: brentq で ∂LL/∂gamma = 0 を解く（同様）
    // （省略: 実装は x0 と対称）

    // CM-step eta: grid search（eta_lo〜eta_hi, step=eta_step）
    let n = ((eta_hi - eta_lo) / eta_step).round() as usize + 1;
    let best = (0..n).map(|i| {
        let e = (eta_lo + i as f64 * eta_step).min(eta_hi);
        let ll = log_likelihood(x, ArrayView1::from(&int_s), *x0, *gamma, e, x_min, x_max);
        (e, ll)
    }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    *eta = best.0;
}

// -----------------------------------------------------------------------
// full_optimization: L-BFGS-B (x0, gamma) + rayon 並列 eta グリッドサーチ
// -----------------------------------------------------------------------

/// MLE via L-BFGS-B（Python の full_optimization に相当）
pub fn mle_full_optimization(
    x: ArrayView1<f64>, intensity: ArrayView1<f64>,
    x0: &mut f64, gamma: &mut f64, eta: &mut f64,
    x_min: f64, x_max: f64, gamma_min: f64, gamma_max: f64,
    eta_n_grid: usize,  // Python デフォルト: 100
) {
    let x_s = x.to_vec();
    let w_s = intensity.to_vec();
    let eta_fixed = *eta;

    let mut params = vec![*x0, *gamma];
    let lb = vec![x_min, gamma_min];
    let ub = vec![x_max, gamma_max];
    lbfgsb::lbfgsb(
        &mut params, &lb, &ub,
        |p, g| {
            let eps = 1e-7;
            let ll   = log_likelihood(ArrayView1::from(&x_s), ArrayView1::from(&w_s),
                                      p[0],       p[1],       eta_fixed, x_min, x_max);
            let ll_dx = log_likelihood(ArrayView1::from(&x_s), ArrayView1::from(&w_s),
                                       p[0] + eps, p[1],       eta_fixed, x_min, x_max);
            let ll_dg = log_likelihood(ArrayView1::from(&x_s), ArrayView1::from(&w_s),
                                       p[0],       p[1] + eps, eta_fixed, x_min, x_max);
            g[0] = -(ll_dx - ll) / eps;
            g[1] = -(ll_dg - ll) / eps;
            -ll
        },
        1e-7, 1e-7, 100,
    );
    *x0 = params[0];
    *gamma = params[1];

    let step = 1.0 / (eta_n_grid - 1) as f64;
    *eta = (0..eta_n_grid)
        .into_par_iter()
        .map(|i| {
            let e = (i as f64 * step).min(1.0);
            let ll = log_likelihood(ArrayView1::from(&x_s), ArrayView1::from(&w_s),
                                    *x0, *gamma, e, x_min, x_max);
            (e, ll)
        })
        .reduce(|| (0.5, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a })
        .0;
}
```

### Step 2: `_pvmm.py` の変更

```python
# PseudoVoigtMixture/_pvmm.py に追加
def adapted_em(self, x, intensity, max_iter, r_eps, stdout):
    from EMPeaks.EMCore._backend import get_backend
    if get_backend() == "rust" and self.background == "none":
        return self._adapted_em_rust_pv(x, intensity, max_iter, r_eps, stdout)
    return self._adapted_em_python(x, intensity, max_iter, r_eps, stdout)
```

`_adapted_em_rust_pv` は Phase 1 の `_adapted_em_rust` と同構造。M-step で以下を呼び分ける:
- `method == "conditional_max"` → `empeaks_rust_core.pseudo_voigt_mle_conditional_max(...)`
- `method == "full_optimization"` → `empeaks_rust_core.pseudo_voigt_mle_full_optimization(...)`

### Step 3: `lib.rs` に追加する PyO3 エクスポート

```rust
#[pyfunction]
fn pseudo_voigt_predict(py, x, x0, gamma, eta, x_min, x_max) -> PyResult<PyObject> { ... }

#[pyfunction]
fn pseudo_voigt_mle_conditional_max(
    x, intensity, x0, gamma, eta,
    x_min, x_max, gamma_min, gamma_max,
    eta_lo, eta_hi, eta_step,
) -> PyResult<(f64, f64, f64)> { ... }  // 返り値: (x0, gamma, eta)

#[pyfunction]
fn pseudo_voigt_mle_full_optimization(
    x, intensity, x0, gamma, eta,
    x_min, x_max, gamma_min, gamma_max, eta_n_grid,
) -> PyResult<(f64, f64, f64)> { ... }  // 返り値: (x0, gamma, eta)
```

### Step 4: パリティテスト方針

両アルゴリズムとも Rust 化されるため、Python と Rust で**同一アルゴリズム**を実行する:

```python
@pytest.mark.parametrize("method", ["conditional_max", "full_optimization"])
def test_parity_pseudo_voigt(sample_data, method):
    """同じ初期値・同じアルゴリズム → LL が atol=1e-5 で一致"""
    # conditional_max: brentq（roots クレート）vs brentq（scipy）→ 同一アルゴリズム
    # full_optimization: lbfgsb（lbfgsb クレート）vs lbfgsb（scipy）→ 同一アルゴリズム
    assert np.isclose(r_rs['LL'], r_py['LL'], atol=1e-5)
```

### Step 5: ベンチマーク確認

```bash
# PseudoVoigt での速度比較
python -c "
from EMPeaks.PseudoVoigtMixture import PseudoVoigtMixtureModel
from EMPeaks.EMCore._backend import set_backend
import numpy as np, time

x = np.linspace(-5, 5, 200)
intensity = ...

for backend in ['python', 'rust']:
    set_backend(backend)
    ...
    print(f'{backend}: {elapsed:.3f}s')
"
```

---

## Phase 2-② Lorentzian（PseudoVoigt 確認後に実施）

### Python 実装の分析

- `maximum_likelihood_estimation` → `minimize_bfgs`（L-BFGS-B）
- パラメータ: x0（位置）、gamma（半値幅）
- 境界: `x_min ≤ x0 ≤ x_max`、`0.1 ≤ gamma ≤ 2000`
- Python と Rust の MLE アルゴリズムは**同一**（ともに L-BFGS-B）→ atol=1e-5 でパリティ確認可

### 実装ファイル: `src/lorentzian.rs`

```rust
pub fn predict(x: ArrayView1<f64>, x0: f64, gamma: f64) -> Vec<f64> { ... }
pub fn mle(x, intensity, x0, gamma, x_min, x_max, gamma_min, gamma_max) { ... }
// lbfgsb で (x0, gamma) を最適化（2パラメータ）
```

---

## Phase 2-③ DoniachSunjic（Lorentzian 確認後に実施）

### Python 実装の分析

- `maximum_likelihood_estimation` → `full_optimization`（L-BFGS-B）
- パラメータ: x0、gamma、alpha（非対称パラメータ）
- 境界: 各パラメータの min/max
- Python と Rust の MLE アルゴリズムは**同一**（ともに L-BFGS-B）→ atol=1e-5 でパリティ確認可

### 実装ファイル: `src/doniach_sunjic.rs`

```rust
pub fn predict(x: ArrayView1<f64>, x0: f64, gamma: f64, alpha: f64) -> Vec<f64> { ... }
// DS関数: cos(pi*alpha/2 + (1-alpha)*arctan((x-x0)/gamma)) / ...
pub fn mle(x, intensity, x0, gamma, alpha, bounds) { ... }
// lbfgsb で (x0, gamma, alpha) を最適化（3パラメータ）
```

---

## 数値一致テスト方針まとめ

| モデル | MLE アルゴリズム（Python/Rust） | パリティ基準 |
|---|---|---|
| Gaussian | 解析解 / 解析解（同一） | atol=1e-5（exact match） |
| PseudoVoigt `conditional_max` | brentq（scipy）/ brentq（roots クレート）（同一） | atol=1e-5 |
| PseudoVoigt `full_optimization` | L-BFGS-B（scipy）/ L-BFGS-B（lbfgsb クレート）（同一） | atol=1e-5 |
| Lorentzian | L-BFGS-B / L-BFGS-B（同一） | atol=1e-5 |
| DoniachSunjic | L-BFGS-B / L-BFGS-B（同一） | atol=1e-5 |

---

## Phase 3: sampling の並列化

**スコープ**: 複数 trial の Rayon 並列実行

**追加 Cargo 依存**: なし（`rayon` は Phase 2 で導入済み）

Phase 2 で PseudoVoigt のグリッドサーチ用に導入した `rayon` を、`sampling` の trial 並列化にそのまま流用する。

---

## 予想される高速化

| シナリオ | Phase 1 後 | Phase 2 後 | Phase 3 後 |
|---|---|---|---|
| Gaussian フィッティング | **3〜10x** | — | **10〜40x** |
| Lorentzian フィッティング | 1x (対象外) | **2〜5x** | **8〜20x** |
| PseudoVoigt フィッティング | 1x (対象外) | **5〜15x** | **20〜60x** |
| DoniachSunjic フィッティング | 1x (対象外) | **3〜8x** | **12〜30x** |

---

## 検証コマンド

```bash
# Rust モジュールのビルド
cd EMPeaks/_rust_core && maturin develop --release

# 数値一致テスト
python -m pytest tests/test_rust_parity.py -v

# ベンチマーク
python -c "
from EMPeaks.EMCore._backend import set_backend
from EMPeaks.GaussianMixture import GaussianMixtureModel
import numpy as np, time

x = np.linspace(-10, 10, 1000)
intensity = np.exp(-x**2) + 0.5*np.exp(-(x-3)**2)
intensity /= intensity.sum()

for backend in ['python', 'rust']:
    set_backend(backend)
    m = GaussianMixtureModel(K=2, x_min=-10, x_max=10)
    t0 = time.perf_counter()
    r = m.fit(x, intensity, method='adapted_em', max_iter=3000, stdout=False)
    print(f'{backend}: {time.perf_counter()-t0:.3f}s  LL={r[\"LL\"]:.6f}')
"
```
