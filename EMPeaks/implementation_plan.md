# EMPeaks/EMCore Rust 高速化 実装計画 (v3)

## 概要

EMPeaks の計算コアを Rust で書き換え、Python バインディング (PyO3 + maturin) を通じて呼び出すことで、EMアルゴリズムのフィッティング処理を高速化する。Rust 実装を**標準**としつつ、Python 実装も保持して切り替え可能にする。

---

## 完了済み作業

| 作業 | 内容 |
|---|---|
| Python 要件引き上げ | `python_requires = >=3.11`（setup.cfg） |
| classifiers 更新 | Python 3.11 / 3.12 / 3.13 を追加 |
| 依存ライブラリ更新 | numpy>=1.23.0, scipy>=1.9.0, matplotlib>=3.6.0, pandas>=1.5.0 |
| `trapz` 置換 | `integrate.trapz` → `integrate.trapezoid`、`np.trapz` → `np.trapezoid`（6ファイル） |
| SyntaxWarning 修正 | `is 0` → `== 0`、`is not 1` → `!= 1`（_lorentz.py） |
| Python 3.11 動作確認 | Gaussian / Lorentzian / PseudoVoigt / DoniachSunjic 全モデル OK |

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

### スコープ

- Lorentzian: predict + MLE（`lbfgsb` による L-BFGS-B、パラメータ: x0, gamma）
- PseudoVoigt: predict + MLE（`lbfgsb` による L-BFGS-B で x0/gamma、`rayon` による alpha グリッドサーチ並列化）
- DoniachSunjic: predict + MLE（`lbfgsb` による L-BFGS-B、パラメータ: x0, gamma, alpha）
- 背景モデル predict の Rust 統合

### 追加ファイル構成

```
_rust_core/src/
├── lib.rs            ← [MODIFY] Phase 2 関数を追加登録
├── gaussian.rs       ← [既存]
├── em_engine.rs      ← [既存]
├── lorentzian.rs     ← [NEW] predict + MLE
├── pseudo_voigt.rs   ← [NEW] predict + MLE（rayon 並列グリッドサーチ）
├── doniach_sunjic.rs ← [NEW] predict + MLE
└── background.rs     ← [NEW] 背景モデル predict
```

### 追加 Cargo 依存

```toml
[dependencies]
lbfgsb = "0.3"   # scipy と同一 Fortran L-BFGS-B v3.0 ラッパー
rayon  = "1.10"  # PseudoVoigt alpha グリッドサーチの並列化
```

### 実装パターン（Lorentzian を例に）

```rust
// lorentzian.rs
use lbfgsb::lbfgsb;

pub struct Lorentzian { pub x0: f64, pub gamma: f64 }

impl Lorentzian {
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let denom = std::f64::consts::PI * self.gamma;
        x.iter().map(|&xi| {
            let d = xi - self.x0;
            (1.0 / denom) / (1.0 + (d / (self.gamma / 2.0)).powi(2))
        }).collect()
    }

    pub fn log_likelihood(&self, x: &[f64], intensity: &[f64]) -> f64 {
        let p = self.predict(x);
        intensity.iter().zip(&p)
            .map(|(&i, &pi)| i * (pi + 1e-200_f64).ln())
            .sum()
    }

    /// MLE via L-BFGS-B（scipy と同一 Fortran コード）
    pub fn mle(&mut self, x: &[f64], intensity: &[f64],
               x_min: f64, x_max: f64, gamma_min: f64, gamma_max: f64) {
        let mut params = vec![self.x0, self.gamma];
        let lb = vec![x_min, gamma_min];
        let ub = vec![x_max, gamma_max];

        lbfgsb(
            &mut params,
            &lb, &ub,
            |p, g| {
                self.x0 = p[0]; self.gamma = p[1];
                let ll = self.log_likelihood(x, intensity);
                // 数値微分（Phase 2 初期）; 解析勾配は Phase 2 後半で追加
                let eps = 1e-7;
                self.x0 = p[0] + eps;
                g[0] = -(self.log_likelihood(x, intensity) - ll) / eps;
                self.x0 = p[0]; self.gamma = p[1] + eps;
                g[1] = -(self.log_likelihood(x, intensity) - ll) / eps;
                self.gamma = p[1];
                -ll
            },
            1e-7, 1e-7, 100,
        );
        self.x0 = params[0]; self.gamma = params[1];
    }
}
```

### PseudoVoigt: alpha グリッドサーチの rayon 並列化

```rust
// pseudo_voigt.rs
use rayon::prelude::*;

pub fn mle_alpha_grid(
    x: &[f64], intensity: &[f64],
    x0: f64, gamma: f64,
    alpha_min: f64, alpha_max: f64, n_grid: usize,
) -> f64 {
    let alphas: Vec<f64> = (0..n_grid)
        .map(|i| alpha_min + (alpha_max - alpha_min) * i as f64 / (n_grid - 1) as f64)
        .collect();

    alphas.par_iter()   // rayon で並列評価
        .map(|&alpha| {
            let pv = PseudoVoigt { x0, gamma, alpha };
            (alpha, pv.log_likelihood(x, intensity))
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(alpha, _)| alpha)
        .unwrap_or(0.5)
}
```

### 数値一致テスト（Phase 2 用）

```python
# tests/test_rust_parity.py に追加
@pytest.mark.parametrize("ModelClass", [
    LorentzianMixtureModel,
    PseudoVoigtMixtureModel,
    DoniachSunjicMixtureModel,
])
def test_parity_phase2(ModelClass, sample_data):
    """scipy L-BFGS-B と lbfgsb クレートの数値一致（atol=1e-5）"""
    x, intensity = sample_data
    np.random.seed(0)

    set_backend("python")
    m_py = ModelClass(K=2, x_min=-5, x_max=5)
    m_py.init_param_uniform()
    r_py = m_py.fit(x, intensity, method='adapted_em', max_iter=500)

    set_backend("rust")
    m_rs = ModelClass(K=2, x_min=-5, x_max=5)
    m_rs.init_param_uniform()  # 同一初期値
    r_rs = m_rs.fit(x, intensity, method='adapted_em', max_iter=500)

    assert np.allclose(r_py['LL'], r_rs['LL'], atol=1e-5), \
        f"LL mismatch ({ModelClass.__name__}): py={r_py['LL']:.8f}, rs={r_rs['LL']:.8f}"
```

> [!NOTE]
> `lbfgsb` が scipy と同一 Fortran コードを使うため atol は `1e-5` 程度（浮動小数点演算順序の差異）で十分。Gaussian の `1e-6` より若干緩めに設定。

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
