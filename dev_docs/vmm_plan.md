# VoigtMixture Rust 高速化 実装計画（最終版 v6）

## 目標
1. Python 版 `_voigt.py` のバグ修正とスマートCDFアルゴリズムへの切り替え
2. `VoigtMixtureModel` の EM フィッティングを `_rust_core` に移植（高速化・並列化）
3. 全モデルの Rust 実装に `fix_` フラグ分岐を横展開

---

## 検証結果のまとめ

### 4段ロケット戦略 → 不要
20回のランダム初期値テストで直接フル最適化と有意差なし。Rust 側では**フル最適化のみ**実装。Python 側は変更しない。

### `full_optimization_fix_sigma_gamma` の init/bounds 不一致
`init = [x0, self.sigma, self.gamma]`（3要素）に `bounds = [(x_min, x_max)]`（1要素）。scipy が余分な要素を無視するため動作上は偶然正しいが、修正すべき。

### スマートCDFアルゴリズム
ユーザー発案の「裾野引き算方式」を検証・採用。詳細は [smart_cdf_algorithm.md](file:///Users/yasunobu/.gemini/antigravity-ide/brain/00209d1e-2de7-4bd4-9b26-30b83b813b7a/smart_cdf_algorithm.md) を参照。

---

## フェーズ 0: Python 版の修正（Rust 化の前に実施）

### Step 0.1: [FIX] `full_optimization_fix_sigma_gamma` の init 修正
#### [MODIFY] [_voigt.py](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/VoigtMixture/_voigt.py#L227)
```diff
-        init = [x0, self.sigma, self.gamma]
+        init = [x0]
```

### Step 0.2: [MODIFY] `_voigt.py` をスマートCDFに切り替え
現在の「複合ガウス求積（3区間分割）」を「スマートCDF（裾野引き算方式）」に置換する。

```python
# モジュールレベル定数（旧 _GAUSS_X_PEAK 等を置換）
_GX, _GW = leggauss(100)
_U = 0.5 * _GX + 0.5   # [0, 1] にマッピング
_W = 0.5 * _GW

def _tail_integral(limit, direction, x0, sigma, gamma):
    """半無限区間の積分。direction=+1: [limit,∞), direction=-1: (-∞,limit]"""
    x = limit + direction * (_U / (1.0 - _U))
    jac = 1.0 / ((1.0 - _U)**2)
    y = voigt_profile(x - x0, sigma, gamma)
    return np.sum(_W * y * jac)

def _smart_cdf(x, x0, sigma, gamma):
    """ピークを跨がない累積分布関数"""
    if x <= x0:
        return _tail_integral(x, -1, x0, sigma, gamma)
    else:
        return 1.0 - _tail_integral(x, 1, x0, sigma, gamma)
```

`_Z()` と `cdf()` をスマートCDFベースに変更:

```python
def _Z(self):
    return _smart_cdf(self.x_max, self.x0, self.sigma, self.gamma) \
         - _smart_cdf(self.x_min, self.x0, self.sigma, self.gamma)

def cdf(self, x):
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    f_min = _smart_cdf(self.x_min, self.x0, self.sigma, self.gamma)
    results = np.array([
        _smart_cdf(xi, self.x0, self.sigma, self.gamma) - f_min
        for xi in x_arr
    ])
    return results.item() if np.ndim(x) == 0 else results
```

### Step 0.3: Python 版の動作検証
既存テスト（`test_gauss_deg.py`）および手動テストで精度・速度を確認。

---

## フェーズ 1: Voigt の Rust 化

### Step 1.1: 依存関係の追加
#### [MODIFY] [Cargo.toml](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/_rust_core/Cargo.toml)
```toml
errorfunctions = "0.2"  # Faddeeva 関数 w(z) (f64)
```

> [!NOTE]
> `xsf` クレートは既に依存に含まれているが、Faddeeva 関数は `errorfunctions` の方が API が直接的。`xsf` にも `dawson` 等はあるが Faddeeva w(z) の直接的なインターフェースがないため `errorfunctions` を採用。

---

### Step 1.2: [NEW] `voigt.rs` の実装

既存の [pseudo_voigt.rs](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/_rust_core/src/pseudo_voigt.rs) のパターンに倣い、以下の構造で実装する。

```rust
// voigt.rs
use errorfunctions::RealErrorFunctions;  // Faddeeva w(z)

const PI: f64 = std::f64::consts::PI;
const SQRT_2PI: f64 = 2.5066282746310002;  // sqrt(2*pi)

// ================================================================
// ガウス・ルジャンドル求積（100点）のノード・重み（コンパイル時定数）
// ================================================================
// Python の numpy.polynomial.legendre.leggauss(100) から生成し、
// [0, 1] にマッピング済みの u, w を const 配列として埋め込む
const N_QUAD: usize = 100;
const QUAD_U: [f64; N_QUAD] = [ /* 事前計算値 */ ];
const QUAD_W: [f64; N_QUAD] = [ /* 事前計算値 */ ];

// ================================================================
// Voigt プロファイル（Faddeeva 関数ベース）
// ================================================================
fn voigt_profile(x: f64, sigma: f64, gamma: f64) -> f64 {
    // z = (x + i*gamma) / (sigma * sqrt(2))
    // V(x) = Re[w(z)] / (sigma * sqrt(2*pi))
    let sigma_sqrt2 = sigma * std::f64::consts::SQRT_2;
    let z_re = x / sigma_sqrt2;
    let z_im = gamma / sigma_sqrt2;
    // errorfunctions::faddeeva(z_re, z_im) → (re, im)
    let (w_re, _w_im) = faddeeva_w(z_re, z_im);
    w_re / (sigma * SQRT_2PI)
}

// ================================================================
// スマートCDF: 裾野引き算方式
// ================================================================
fn tail_integral(limit: f64, direction: f64, x0: f64, sigma: f64, gamma: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..N_QUAD {
        let u = QUAD_U[i];
        let w = QUAD_W[i];
        let t = limit + direction * (u / (1.0 - u));
        let jac = 1.0 / ((1.0 - u) * (1.0 - u));
        sum += w * voigt_profile(t - x0, sigma, gamma) * jac;
    }
    sum
}

fn smart_cdf(x: f64, x0: f64, sigma: f64, gamma: f64) -> f64 {
    if x <= x0 {
        tail_integral(x, -1.0, x0, sigma, gamma)
    } else {
        1.0 - tail_integral(x, 1.0, x0, sigma, gamma)
    }
}

pub fn compute_z(x0: f64, sigma: f64, gamma: f64, x_min: f64, x_max: f64) -> f64 {
    smart_cdf(x_max, x0, sigma, gamma) - smart_cdf(x_min, x0, sigma, gamma)
}

// ================================================================
// predict / LL
// ================================================================
pub fn predict_inplace(x: &[f64], x0: f64, sigma: f64, gamma: f64,
                       x_min: f64, x_max: f64, out: &mut [f64]) {
    let z = compute_z(x0, sigma, gamma, x_min, x_max).max(1e-300);
    for i in 0..x.len() {
        out[i] = voigt_profile(x[i] - x0, sigma, gamma) / z;
    }
}

fn ll_voigt(x: &[f64], intensity: &[f64], x0: f64, sigma: f64, gamma: f64,
            x_min: f64, x_max: f64) -> f64 {
    let z = compute_z(x0, sigma, gamma, x_min, x_max).max(1e-300);
    x.iter().zip(intensity.iter())
        .map(|(&xi, &ii)| {
            let p = voigt_profile(xi - x0, sigma, gamma) / z;
            ii * (p + 1e-200).ln()
        })
        .sum()
}

// ================================================================
// MLE（L-BFGS-B、fix_ フラグ対応）
// ================================================================
pub fn mle_voigt(
    x: &[f64], intensity: &[f64],
    x0: &mut f64, sigma: &mut f64, gamma: &mut f64,
    fix_x0: bool, fix_sigma: bool, fix_gamma: bool,
    x_min: f64, x_max: f64,
    sigma_min: f64, sigma_max: f64,
    gamma_min: f64, gamma_max: f64,
) {
    if fix_x0 && fix_sigma && fix_gamma { return; }

    // 自由パラメータを動的に構成
    let mut init = Vec::new();
    let mut bounds = Vec::new();

    if !fix_x0    { init.push(*x0);    bounds.push((x_min, x_max)); }
    if !fix_sigma { init.push(*sigma); bounds.push((sigma_min, sigma_max)); }
    if !fix_gamma { init.push(*gamma); bounds.push((gamma_min, gamma_max)); }

    let eps_num = 1e-7;
    let x_s = x.to_vec();
    let w_s = intensity.to_vec();

    // クロージャ内でパラメータをアンパック
    let unpack = |p: &[f64]| -> (f64, f64, f64) {
        let mut idx = 0;
        let x0_v  = if fix_x0    { *x0 }    else { let v = p[idx]; idx += 1; v };
        let sig_v = if fix_sigma { *sigma } else { let v = p[idx]; idx += 1; v };
        let gam_v = if fix_gamma { *gamma } else { let v = p[idx]; idx += 1; v };
        (x0_v, sig_v, gam_v)
    };

    if let Ok(state) = lbfgsb::lbfgsb(init, &bounds, |p, g| {
        let (x0_v, sig_v, gam_v) = unpack(p);
        let ll = ll_voigt(&x_s, &w_s, x0_v, sig_v, gam_v, x_min, x_max);

        // 数値微分
        for i in 0..p.len() {
            let mut p_eps = p.to_vec();
            p_eps[i] += eps_num;
            let (x0_e, sig_e, gam_e) = unpack(&p_eps);
            let ll_e = ll_voigt(&x_s, &w_s, x0_e, sig_e, gam_e, x_min, x_max);
            g[i] = -(ll_e - ll) / eps_num;
        }
        Ok(-ll)
    }) {
        let (x0_r, sig_r, gam_r) = unpack(state.x());
        *x0    = x0_r;
        *sigma = sig_r;
        *gamma = gam_r;
    }
}
```

> [!IMPORTANT]
> `errorfunctions` クレートの Faddeeva 関数 `w(z)` の具体的な呼び出しAPIは実装時に確認する。`errorfunctions` は `Complex<f64>` を入力とする `faddeeva` メソッドを提供している可能性があるため、`num-complex` の追加が必要になる場合がある。

---

### Step 1.3: [NEW] `em_engine.rs` に `run_voigt_em_loop` を追加

既存の [run_pv_em_loop](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/_rust_core/src/em_engine.rs#L115-L200) パターンに倣う。

```rust
// em_engine.rs に追加
use crate::voigt;

pub fn run_voigt_em_loop(
    x: &[f64],
    intensity: &[f64],
    x0: &mut Vec<f64>,
    sigma: &mut Vec<f64>,        // Voigt 固有: ガウス σ
    gamma_v: &mut Vec<f64>,      // Voigt 固有: ローレンツ γ
    pi: &mut Vec<f64>,
    dirichlet_alpha: &[f64],
    fix_x0: &[bool],            // fix_ フラグ（各ピーク）
    fix_sigma: &[bool],
    fix_gamma: &[bool],
    x_min: f64, x_max: f64,
    sigma_min: f64, sigma_max: f64,
    gamma_min: f64, gamma_max: f64,
    max_iter: usize,
    r_eps: f64,
    bg_type: u8,
    mut s_tri: f64,
) -> (usize, Vec<f64>, Vec<f64>, f64) {
    let k_peaks = x0.len();
    let k_all = k_peaks + if bg_type != background::BG_NONE { 1 } else { 0 };
    let mut predictions = vec![vec![0.0; x.len()]; k_all];
    let mut mixture = vec![0.0; x.len()];
    let mut gamma = vec![vec![0.0; x.len()]; k_all];

    // 初期 predict
    predictions[0..k_peaks].par_iter_mut().enumerate().for_each(|(k, pred)| {
        voigt::predict_inplace(x, x0[k], sigma[k], gamma_v[k], x_min, x_max, pred);
    });
    update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
    let mut ll_0 = em_gamma_ll::compute_gamma_and_ll_inplace(
        intensity, &predictions, pi, &mut mixture, &mut gamma);

    let mut ll_hist = vec![ll_0];
    let mut res_hist = vec![0.0_f64];
    let mut total_iter = max_iter;

    for it in 0..max_iter {
        em_gamma_ll::update_pi(pi, intensity, &gamma, dirichlet_alpha);

        // M-step: 各ピークの MLE を並列実行
        let results: Vec<(f64, f64, f64)> = (0..k_peaks).into_par_iter().map(|k| {
            let w: Vec<f64> = intensity.iter().zip(gamma[k].iter())
                .map(|(&i, &g)| i * g).collect();
            let (mut x0_k, mut sig_k, mut gam_k) = (x0[k], sigma[k], gamma_v[k]);
            voigt::mle_voigt(
                x, &w, &mut x0_k, &mut sig_k, &mut gam_k,
                fix_x0[k], fix_sigma[k], fix_gamma[k],
                x_min, x_max, sigma_min, sigma_max, gamma_min, gamma_max,
            );
            (x0_k, sig_k, gam_k)
        }).collect();

        for (k, (x0_k, sig_k, gam_k)) in results.into_iter().enumerate() {
            x0[k] = x0_k; sigma[k] = sig_k; gamma_v[k] = gam_k;
        }

        // Background MLE
        if bg_type == background::BG_LINEAR {
            let w_bg: Vec<f64> = intensity.iter().zip(gamma[k_peaks].iter())
                .map(|(&i, &g)| i * g).collect();
            s_tri = background::mle_linear(x, &w_bg, x_min, x_max);
        }

        // predict 更新
        predictions[0..k_peaks].par_iter_mut().enumerate().for_each(|(k, pred)| {
            voigt::predict_inplace(x, x0[k], sigma[k], gamma_v[k], x_min, x_max, pred);
        });
        update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
        let ll = em_gamma_ll::compute_gamma_and_ll_inplace(
            intensity, &predictions, pi, &mut mixture, &mut gamma);

        let residual = (ll - ll_0) / ll_0.abs();
        ll_hist.push(ll);
        res_hist.push(residual);

        if residual.abs() < r_eps {
            total_iter = it + 1;
            break;
        }
        ll_0 = ll;
    }

    (total_iter, ll_hist, res_hist, s_tri)
}
```

---

### Step 1.4: [MODIFY] `lib.rs` にバインディング追加

#### [MODIFY] [lib.rs](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/_rust_core/src/lib.rs)

1. `mod voigt;` を追加（L12付近）
2. `run_voigt_em_loop` の `#[pyfunction]` ラッパーを追加（既存の `run_pv_em_loop` ラッパーのパターンに倣う）

```rust
#[pyfunction]
#[pyo3(signature = (x, intensity, x0, sigma, gamma_v, pi, dirichlet_alpha,
                    fix_x0, fix_sigma, fix_gamma,
                    x_min, x_max, sigma_min, sigma_max, gamma_min, gamma_max,
                    max_iter, r_eps, bg_type=0u8, s_tri=0.0f64))]
fn run_voigt_em_loop<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    intensity: PyReadonlyArray1<'py, f64>,
    mut x0: PyReadwriteArray1<'py, f64>,
    mut sigma: PyReadwriteArray1<'py, f64>,
    mut gamma_v: PyReadwriteArray1<'py, f64>,
    mut pi: PyReadwriteArray1<'py, f64>,
    dirichlet_alpha: PyReadonlyArray1<'py, f64>,
    fix_x0: Vec<bool>,
    fix_sigma: Vec<bool>,
    fix_gamma: Vec<bool>,
    x_min: f64, x_max: f64,
    sigma_min: f64, sigma_max: f64,
    gamma_min: f64, gamma_max: f64,
    max_iter: usize, r_eps: f64,
    bg_type: u8, s_tri: f64,
) -> PyResult<(usize, PyObject, PyObject, f64)> {
    // ... 既存パターン通り: Vec 化 → allow_threads → 結果書き戻し
}
```

3. `empeaks_rust_core` モジュールに `run_voigt_em_loop` を登録

---

### Step 1.5: [MODIFY] `_vmm.py` に Rust バックエンド分岐を追加

#### [MODIFY] [_vmm.py](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/VoigtMixture/_vmm.py)

`VoigtMixtureModel` クラスに以下を追加:

```python
def adapted_em(self, x, intensity, max_iter, r_eps, stdout):
    """Rust バックエンドが利用可能ならそちらに委譲"""
    from EMPeaks.EMCore._backend import get_backend
    if (get_backend() == "rust"
        and self.background in {"none", "uniform", "squareroot", "linear"}):
        return self._adapted_em_rust_voigt(x, intensity, max_iter, r_eps, stdout)
    return self._adapted_em_python(x, intensity, max_iter, r_eps, stdout)

def _adapted_em_rust_voigt(self, x, intensity, max_iter, r_eps, stdout):
    import empeaks_rust_core
    # 既存の _adapted_em_rust パターンに倣い:
    # 1. model からパラメータ抽出 (x0, sigma, gamma, fix_*)
    # 2. empeaks_rust_core.run_voigt_em_loop() 呼び出し
    # 3. 結果を model に書き戻し
    ...
```

---

### Step 1.6: ガウス・ルジャンドル求積点の定数生成

Rust の `const` 配列として埋め込むために、Python スクリプトで 100 点の Gauss-Legendre ノード・重みを [0, 1] マッピング済みで生成し、`voigt.rs` にペーストする。

```python
# 生成スクリプト
from numpy.polynomial.legendre import leggauss
gx, gw = leggauss(100)
u = 0.5 * gx + 0.5
w = 0.5 * gw
print(f"const QUAD_U: [f64; {len(u)}] = [")
for v in u: print(f"    {v:.18e},")
print("];")
# 同様に QUAD_W
```

---

## フェーズ 2: 全関数への `fix_` 分岐の横展開（VMM 完了後）

### 対象モデルと fix パラメータ一覧

| モデル | Rust ファイル | fix パラメータ | 現状 |
|---|---|---|---|
| Gaussian | [gaussian.rs](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/_rust_core/src/gaussian.rs) | `fix_mu`, `fix_sigma` | フル最適化のみ |
| Lorentzian | [lorentzian.rs](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/_rust_core/src/lorentzian.rs) | `fix_x0`, `fix_gamma` | フル最適化のみ |
| PseudoVoigt | [pseudo_voigt.rs](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/_rust_core/src/pseudo_voigt.rs) | `fix_x0`, `fix_gamma`, `fix_eta` | フル最適化のみ |
| DoniachSunjic | [doniach_sunjic.rs](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/_rust_core/src/doniach_sunjic.rs) | `fix_x0`, `fix_gamma`, `fix_alpha` | フル最適化のみ |
| TSDC | [tsdc.rs](file:///Users/yasunobu/Documents/GitHub/EMPeaks/EMPeaks/_rust_core/src/tsdc.rs) | `fix_ea`, `fix_tau0` | フル最適化のみ |

### 実装方針
Voigt の `mle_voigt` で確立した「動的パラメータ構成」パターン（`fix_*` フラグに基づいて `init` と `bounds` を動的に構築し、`unpack` で復元する）を、各モデルの `mle_*` 関数に適用する。

### 各モデルへの手順（Step 2.1〜2.5）
1. `.rs` の MLE 関数に `fix_*: bool` 引数を追加し、動的パラメータ構成を実装
2. `em_engine.rs` の各 EM ループに `fix_*` ベクトルを受け渡し
3. `lib.rs` のバインディングに `fix_*` 引数を追加
4. Python 側ラッパーで `model[k].fix_*` を収集して渡す
5. テスト

---

## 検証計画

### 自動テスト
```bash
cd EMPeaks && maturin develop --release && python -m pytest
```

### 手動検証
1. `VoigtMixtureModel` で既知パラメータのデータを生成し、Rust/Python 両バックエンドでフィッティング結果が一致することを確認
2. `fix_x0=True` 等の部分固定パラメータで正しく動作することを確認
3. 速度ベンチマーク: `sampling(trial=10)` の実行時間を Python vs Rust で比較

---

## User Review Required

> [!IMPORTANT]
> - フェーズ 0（Python バグ修正 + スマートCDF 切り替え）→ フェーズ 1（Rust 化）→ フェーズ 2（横展開）の順序で実装します。
> - `errorfunctions` クレートの Faddeeva 関数が `num-complex` を要求する場合、`Cargo.toml` に `num-complex` を追加します。
> - Voigt の Rust 化では `fix_` フラグ対応を最初から組み込みます（後付けにしない）。
