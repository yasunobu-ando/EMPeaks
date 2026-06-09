# Refactoring Plan: Background Support Consolidation

このプランは2つの目標を統合したものです。
1. **BackgroundFactory Refactoring**: `EMCore` のバックグラウンド初期化ロジックを `BackgroundFactory` に完全に委譲し、`_em_core.py` をクリーンにする。
2. **b_spline Rust 対応の全モデルへの展開**: GMM で実装済みの b_spline Rust バックエンドを、PV / Voigt / Lorentzian / DoniachSunjic / TSDC に展開する。

---

## 既知の不具合（緊急修正済み）

### GUI 実行時 TypeError: run_em_loop() takes from 12 to 14 positional arguments but 15 were given

**原因**: `_em_core.py` の `_adapted_em_rust` が b_spline 以外の背景でも `spline_basis_preds=None` を15番目の**位置引数**として渡していたため、古い `.so`（引数14個まで）をロードしている環境で失敗する。

**修正内容（適用済み）**: `spline_basis_preds` は b_spline の場合のみキーワード引数として渡す `**extra_kw` パターンに変更。

```python
# 変更前（問題あり: None を位置引数として渡す）
total_iter, ... = empeaks_rust_core.run_em_loop(
    ..., bg_type, s_tri_in,
    spline_basis_preds,      # ← None でも15番目の位置引数として送信
)

# 変更後（修正済み: b_spline のときのみキーワード引数で付加）
extra_kw = {}
if self.background == "b_spline":
    extra_kw['spline_basis_preds'] = [...]

total_iter, ... = empeaks_rust_core.run_em_loop(
    ..., bg_type, s_tri_in,
    **extra_kw,              # ← b_spline 時のみ渡る
)
```

**Phase 3-B の全モデルにも同じパターンを適用**すること（後述）。

---

## Phase 1: EMCore BackgroundFactory Refactoring

**対象ファイル**: `EMPeaks/EMCore/_em_core.py`

**依存関係**: このフェーズが後続 Phase 3 の前提（`n_spline_basis` 属性をなくすため）。

### 1-A. `__init__` のリファクタ

現在の `if/elif` ブロックを以下に置き換え:

```python
background_models = self._bg_factory.create(self.background)
self.model.extend(background_models)
self.K_all = self.K + len(background_models)
use_random = self.background in ('ramp_sum', 'b_spline')
self._update_mixture_weights(len(background_models), use_random=use_random)
```

削除する属性: `self.degree_spline`, `self.n_spline_basis`, `self.ramp_node`

> **想定されるトラブル**: ノートブックやスクリプトが `model.n_spline_basis` / `model.ramp_node` を直接参照している場合、`AttributeError` になる。
> **対策**: 削除せずに `@property` 経由でファクトリから読み返す shim を設ける。
> ```python
> @property
> def n_spline_basis(self):
>     return self._bg_factory.n_spline_basis
>
> @property
> def ramp_node(self):
>     return self._bg_factory.ramp_node
> ```
> これにより外部コードを壊さずに `EMCore` 内部から直参照を排除できる。属性 shim を入れる場合、後述の Phase 1-C は不要になる（`self.n_spline_basis` が引き続き動作するため）。

### 1-B. `set_param_background` のリファクタ

```python
def set_param_background(self, **param):
    if self.background == 'none':
        self.K_all = self.K
        return
    s_tri = param.get('s_tri', None)
    background_models = self._bg_factory.create(self.background, s_tri=s_tri)
    self.model.extend(background_models)
    self.K_all = self.K + len(background_models)
```

### 1-C. `_adapted_em_rust` の `n_spline_basis` 参照を修正（shim を入れない場合のみ）

`self.n_spline_basis` を `self.K_all - self.K` に置き換え:

```python
extra_kw = {}
if self.background == "b_spline":
    n_bg = self.K_all - self.K          # n_spline_basis の代替
    extra_kw['spline_basis_preds'] = [
        self.model[self.K + i].predict(x_f64).tolist()
        for i in range(n_bg)
    ]
```

---

## Phase 2: Visualizer / GUI Cleanup

**対象ファイル**: `EMPeaks/EMCore/visualizer.py`, `gui/module/fitting_board_constructor.py`

### 2-A. `visualizer.py`: `n_spline_basis` / `k_ramp` 引数の廃止

`Visualizer.plot()` のシグネチャから `k_ramp` と `n_spline_basis` を削除し、背景コンポーネント数を `len(model) - K` で動的に決定:

```python
# 変更前
elif background == 'ramp_sum':
    y = np.sum([model[K+k].predict(x) * N[K+k] for k in range(k_ramp+2)], axis=0)
elif background == 'b_spline':
    y = np.sum([model[K+k].predict(x) * N[K+k] for k in range(n_spline_basis)], axis=0)

# 変更後（multi-component backgrounds を統一）
elif background in ('ramp_sum', 'b_spline'):
    n_bg = len(model) - K
    y = np.sum([model[K+k].predict(x) * N[K+k] for k in range(n_bg)], axis=0)
    ax.plot(x, y, label=background)
```

### 2-B. `fitting_board_constructor.py`: 動的な背景コンポーネント数

```python
# 変更前
k_ramp = getattr(mi, 'k_ramp', 5)
y_bg_arr = np.sum([... for k in range(k_ramp + 2)], axis=0)
# ...
n_spline_basis = getattr(mi, 'n_spline_basis', 0)
y_bg_arr = np.sum([... for k in range(n_spline_basis)], axis=0)

# 変更後（ramp_sum / b_spline を統一）
num_bg_models = mi.K_all - display_K
y_bg_arr = np.sum([
    mi.model[display_K + k].predict(x_data.values) * mi.pi[display_K + k] * mi.N_tot
    for k in range(num_bg_models)
], axis=0)
```

---

## Phase 3: b_spline Rust 対応の全モデル展開

**前提**: Phase 1 完了後（`n_spline_basis` 参照を `K_all - K` または shim で解決済み）

### 3-A. `lib.rs`: 5ループへの `spline_basis_preds` 追加

`run_em_loop` と同じパターンで5関数を更新:

```rust
#[pyo3(signature = (..., bg_type=0u8, s_tri=0.0f64, spline_basis_preds=None))]
fn run_pv_em_loop<'py>(
    ...
    spline_basis_preds: Option<Vec<Vec<f64>>>,
) -> PyResult<...> {
    let sbp: Vec<Vec<f64>> = spline_basis_preds.unwrap_or_default();
    let (...) = py.allow_threads(|| {
        em_engine::run_pv_em_loop(..., bg_type, s_tri, &sbp)
    });
```

対象: `run_pv_em_loop`, `run_voigt_em_loop`, `run_lorentzian_em_loop`, `run_ds_em_loop`, `run_tsdc_em_loop`  
現在 `&[]` をハードコードしている5箇所を `&sbp` に置き換える。

### 3-B. Python 側: 5モデルの `_adapted_em_rust_*` 更新

**共通パターン**（`**extra_kw` を使用。None を位置引数で渡さない）:

```python
# _BG_TYPE_MAP と _BG_RUST_SUPPORTED を更新（クラス変数 or インライン）
_BG_TYPE_MAP = {"none": 0, "uniform": 1, "squareroot": 2, "linear": 3, "b_spline": 4}
_BG_RUST_SUPPORTED = {"none", "uniform", "squareroot", "linear", "b_spline"}

# Rust 呼び出し前に extra_kw を構築
extra_kw = {}
if self.background == "b_spline":
    n_bg = self.K_all - self.K
    extra_kw['spline_basis_preds'] = [
        self.model[self.K + i].predict(x_f64).tolist()
        for i in range(n_bg)
    ]

total_iter, ll_hist_np, res_hist_np, s_tri_out = empeaks_rust_core.run_pv_em_loop(
    ..., bg_type, s_tri_in,
    **extra_kw,     # b_spline のときのみ spline_basis_preds が渡る
)
```

**対象モデルと Rust 関数の対応**:

| モデル | ファイル | Rust 関数 |
|---|---|---|
| PseudoVoigtMixtureModel | `_pvmm.py` | `run_pv_em_loop` |
| VoigtMixtureModel | `_vmm.py` | `run_voigt_em_loop` |
| LorentzianMixtureModel | `_lmm.py` | `run_lorentzian_em_loop` |
| DoniachSunjicMixtureModel | `_dsmm.py` | `run_ds_em_loop` |
| TSDCMixtureModel | `_tsdc_mixture.py` | `run_tsdc_em_loop` |

> **TSDC 注記**: b_spline 背景は温度域スペクトルでは物理的に稀なユースケースだが、実装上は対称なので追加する。デフォルト推奨背景には含めない。

*(GMM b_spline Rust 対応: Phase 3 開始前に実装済み)*

---

## 変更ファイル一覧

| Phase | ファイル | 変更内容 |
|---|---|---|
| 緊急修正済み | `EMPeaks/EMCore/_em_core.py` | `spline_basis_preds` を `**extra_kw` パターンに修正 |
| 1 | `EMPeaks/EMCore/_em_core.py` | `__init__`, `set_param_background` 簡略化、shim or `K_all - K` 参照 |
| 2 | `EMPeaks/EMCore/visualizer.py` | `k_ramp`/`n_spline_basis` パラメータ廃止 |
| 2 | `gui/module/fitting_board_constructor.py` | `K_all - K` による動的背景合算 |
| 3 | `EMPeaks/_rust_core/src/lib.rs` | 5ループに `spline_basis_preds` 追加 |
| 3 | `EMPeaks/PseudoVoigtMixture/_pvmm.py` | b_spline Rust 対応 |
| 3 | `EMPeaks/VoigtMixture/_vmm.py` | b_spline Rust 対応 |
| 3 | `EMPeaks/LorentzianMixture/_lmm.py` | b_spline Rust 対応 |
| 3 | `EMPeaks/DoniachSunjicMixture/_dsmm.py` | b_spline Rust 対応 |
| 3 | `EMPeaks/TSDCMixture/_tsdc_mixture.py` | b_spline Rust 対応 |

---

## フェーズ間の依存関係

```
緊急修正（適用済み）
    │
Phase 1 (EMCore refactor)
    ├─ Phase 2 (Visualizer/GUI)  ← 独立して先行実施も可
    └─ Phase 3 (b_spline Rust 展開)  ← Phase 1 完了が前提
```

Phase 2 は Phase 1 の完了を待たずに先行してもよい（`getattr` フォールバックを現状維持しつつ変更可能）。ただし Phase 1 と同時コミットが最もクリーン。

---

## 検証計画

### 自動テスト
- 既存の Gaussian/PV/Voigt/Lorentzian/DS の Rust 経路が non-spline 背景で引き続き動作することを確認
- b_spline × 各モデルの組み合わせで `adapted_em` が Rust バックエンドを使用していることをログで確認

### 手動検証
- GUI を起動し、non-b_spline 背景（uniform, linear 等）で TypeError が出ないことを確認（緊急修正の回帰テスト）
- `b_spline` および `ramp_sum` モデルで fitting が実行・プロットされることを確認
- `IndexError` / `AttributeError` が出ないこと
