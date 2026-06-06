# EMPeaks/EMCore Rust 高速化 実装計画 (v2)

## 概要

EMPeaks の計算コアを Rust で書き換え、Python バインディング (PyO3 + maturin) を通じて呼び出すことで、EMアルゴリズムのフィッティング処理を高速化する。Rust 実装を**標準**としつつ、Python 実装も保持して切り替え可能にする。

---

## Q1: Rust 拡張パッケージを標準にする場合の問題

### 想定される問題

| 問題 | 深刻度 | 対策 |
|---|---|---|
| **ビルド済みホイールがない環境で `pip install` 失敗** | 🔴 高 | 主要プラットフォーム向けホイールを CI で自動ビルド |
| **ユーザーに Rust ツールチェーン要求** | 🔴 高 | abi3 ホイールで対応バージョン数を削減 + sdist フォールバック |
| **Linux ディストリ間の glibc 互換性** | 🟡 中 | manylinux Docker イメージでビルド |
| **macOS arm64/x86_64 の両対応** | 🟡 中 | universal2 ホイールまたは個別ビルド |
| **CI/CD の複雑化** | 🟡 中 | `PyO3/maturin-action` で自動化 |
| **デバッグの難易度上昇** | 🟢 低 | Python フォールバックを維持して切り替え可能に |

### 推奨対策

1. **abi3 (Stable ABI) の活用**
   - PyO3 の `abi3-py310` feature を使えば、Python 3.10+ 向けに**1つのホイール**で全バージョン対応可能
   - ホイールビルド数: `3プラットフォーム × 1abi3 = 3ホイール` まで削減

2. **GitHub Actions CI/CD**
   ```yaml
   # .github/workflows/release.yml
   - uses: PyO3/maturin-action@v1
     with:
       target: ${{ matrix.target }}
       args: --release --out dist
       manylinux: auto  # manylinux2014 自動選択
   ```
   対象: `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-pc-windows-msvc`

3. **sdist フォールバック**
   - ホイールが見つからない場合、sdist から Rust コンパイルを試行
   - Rust が未インストールの場合は **Python 純粋実装** にフォールバック（後述の切り替え機構）

> [!IMPORTANT]
> **結論**: Rust 拡張を標準にすることは十分実現可能。ただし、CI/CD でのビルド済みホイール配布が必須条件。abi3 を活用すればホイール数は最小限に抑えられる。

---

## Q2: Python 対応バージョン

### 2026年6月時点のサポート状況

| バージョン | ステータス | EOL |
|---|---|---|
| 3.8 | ❌ **EOL済** (2024/10) | — |
| 3.9 | ❌ **EOL済** (2025/10) | — |
| 3.10 | ⚠️ セキュリティのみ | **2026/10** |
| 3.11 | ⚠️ セキュリティのみ | 2027/10 |
| 3.12 | ✅ セキュリティ | 2028/10 |
| 3.13 | ✅ バグ修正 | 2029/10 |
| 3.14 | ✅ バグ修正 | 2030/10 |

### 推奨

**`python_requires >= 3.10`** に引き上げ。理由：

1. 3.8/3.9 は既に EOL で、セキュリティパッチもない
2. 3.10 は 2026/10 に EOL だが、現時点ではまだサポート範囲内
3. PyO3 の abi3 feature で `abi3-py310` を指定すれば 3.10+ 全対応
4. 多くの科学計算ライブラリ（NumPy 2.x, pandas 2.x）も 3.9+ を最低要件としている
5. match 文（3.10+）等の新機能が使える

> [!NOTE]
> もし保守的に行く場合は `python_requires >= 3.11` でも合理的。3.10 の EOL は 2026/10 と迫っている。

---

## Q3: scipy.optimize の Rust 化（Phase 2 論点）

### 分析: 最適化ルーチンのどこが遅いか？

scipy.optimize の L-BFGS-B / brentq は内部的に**高度に最適化された Fortran コード**を呼んでいます。アルゴリズム自体の速度は Rust で書き直しても大きな差は出にくいです。

**高速化が見込めるのは「関数評価のコスト」です。**

```
scipy.optimize.minimize(func_ll, ...)
  └── func_ll() ← Python コールバック（毎回 predict → log_likelihood を計算）
       └── predict(x) ← ここが重い
```

| 分布モデル | MLE 中の predict 呼出回数 | Rust化の効果 |
|---|---|---|
| **Lorentzian** (minimize_bfgs) | L-BFGS-B: 〜50-200回 | ⭐⭐ 中 |
| **PseudoVoigt** (conditional_max) | brentq + grid search: 〜1000回以上 | ⭐⭐⭐ **高** |
| **DoniachSunjic** (full_optimization) | L-BFGS-B: 〜50-200回 + 毎回 trapz | ⭐⭐⭐ **高** |

### Phase 2 での方針

```
戦略A: Rust で最適化アルゴリズム自体も実装 (argmin クレート)
  → predict + optimizer がすべて Rust 内で完結
  → Python ↔ Rust の境界越えがゼロ
  → 効果: ⭐⭐⭐ 高（特に PseudoVoigt, DoniachSunjic）

戦略B: scipy.optimize は維持、predict だけ Rust 化
  → Python から Rust predict を呼ぶコールバック構造
  → Python ↔ Rust の境界越えは残る
  → 効果: ⭐⭐ 中（境界越えコストが支配的な場合は効果薄）
```

> [!IMPORTANT]
> **推奨: 戦略A（完全 Rust 化）**
> 
> 理由: EMPeaks で使用する最適化は L-BFGS-B と brentq の2種類のみ。Rust の `argmin` クレートがどちらもサポートしており、実装コストは妥当。何より、**predict の呼出が数百〜数千回** あるため、Python ↔ Rust の境界越えを排除する効果が大きい。
> 
> ただし、Phase 2 実装時にまず戦略B で実装し、ベンチマークで境界越えのコストを測定してから戦略A に進むのが安全。

---

## Q4: Python / Rust 切り替え機構の設計

### 切り替え方法（3段階）

#### 1. 自動検出（デフォルト）

```python
# EMPeaks/EMCore/_backend.py [NEW]

import os

_BACKEND = None  # "rust" or "python"

def get_backend():
    """現在のバックエンドを取得"""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    
    # 環境変数で強制指定
    env = os.environ.get("EMPEAKS_BACKEND", "auto")
    if env == "python":
        _BACKEND = "python"
        return _BACKEND
    
    if env == "rust" or env == "auto":
        try:
            import empeaks_rust_core
            _BACKEND = "rust"
        except ImportError:
            if env == "rust":
                raise ImportError(
                    "EMPEAKS_BACKEND='rust' が指定されましたが、"
                    "empeaks_rust_core がインストールされていません。"
                )
            _BACKEND = "python"
    
    return _BACKEND

def set_backend(backend: str):
    """バックエンドを動的に切り替え"""
    global _BACKEND
    if backend not in ("rust", "python"):
        raise ValueError(f"backend は 'rust' または 'python' を指定: {backend}")
    _BACKEND = backend
```

#### 2. 環境変数

```bash
# Python 実装を強制
export EMPEAKS_BACKEND=python

# Rust 実装を強制（未インストール時はエラー）
export EMPEAKS_BACKEND=rust

# 自動検出（デフォルト）
export EMPEAKS_BACKEND=auto
```

#### 3. Python API

```python
import EMPeaks
from EMPeaks.EMCore._backend import set_backend, get_backend

# 現在のバックエンドを確認
print(get_backend())  # "rust" or "python"

# ベンチマーク用に切り替え
set_backend("python")
result_py = model.fit(x, intensity)

set_backend("rust")
result_rs = model.fit(x, intensity)
```

### ベンチマーク用ユーティリティ

```python
# EMPeaks/benchmark.py [NEW]
def compare_backends(model, x, intensity, **fit_kwargs):
    """Python と Rust の実行時間・結果を比較"""
    results = {}
    for backend in ["python", "rust"]:
        set_backend(backend)
        start = time.time()
        info = model.fit(x, intensity, **fit_kwargs)
        elapsed = time.time() - start
        results[backend] = {
            "time": elapsed,
            "LL": info["LL"],
            "RMSE": info["RMSE"],
            "iterations": info["total_iter"],
        }
    # 結果の比較表を出力
    ...
    return results
```

---

## 実装計画（更新版）

### Phase 1: Gaussian コア + EMループの Rust 化

**スコープ**: Gaussian predict/MLE + E-step/M-step/EMループ
**スコープ外**: scipy.optimize 依存の処理（leastsq_for_normalization_factor 等）

---

#### [NEW] `EMPeaks/_rust_core/` — Rust プロジェクト

```
EMPeaks/_rust_core/
├── Cargo.toml
├── pyproject.toml       ← maturin 設定
└── src/
    ├── lib.rs           ← PyO3 モジュール定義
    ├── gaussian.rs      ← Gaussian predict/MLE
    ├── background.rs    ← 背景モデル predict
    └── em_engine.rs     ← E-step/M-step/EMループ
```

**Cargo.toml**:
```toml
[package]
name = "empeaks_rust_core"
version = "0.1.0"
edition = "2021"

[lib]
name = "empeaks_rust_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py310"] }
numpy = "0.22"
ndarray = "0.16"
```

---

#### [NEW] `EMPeaks/EMCore/_backend.py` — バックエンド切り替え

上記の切り替え機構を実装。

---

#### [MODIFY] `EMPeaks/EMCore/_em_core.py`

- `adapted_em` メソッドに Rust パスと Python パスの分岐を追加
- Python パスは既存コードをそのまま維持
- Rust パスでは `empeaks_rust_core.run_em_loop()` を呼び出し

```python
def adapted_em(self, x, intensity, max_iter, r_eps, stdout):
    if get_backend() == "rust":
        return self._adapted_em_rust(x, intensity, max_iter, r_eps, stdout)
    else:
        return self._adapted_em_python(x, intensity, max_iter, r_eps, stdout)
```

---

#### [MODIFY] `setup.cfg`

```diff
-python_requires = >=3.8
+python_requires = >=3.10
```

---

### Phase 2: 他の分布モデル + MLE 最適化

**スコープ**:
- Lorentzian predict + MLE (argmin L-BFGS-B)
- PseudoVoigt predict + MLE (argmin brentq)
- DoniachSunjic predict + MLE (argmin L-BFGS-B)
- 背景モデル predict の Rust 統合

**追加 Cargo 依存**:
```toml
[dependencies]
argmin = "0.10"
argmin-math = { version = "0.4", features = ["ndarray_latest-nolinalg"] }
```

**方針**: まず戦略B（predict のみ Rust、optimizer は scipy のまま）を実装してベンチマーク。境界越えコストが大きければ戦略A（argmin による完全 Rust 化）に移行。

---

### Phase 3: sampling の並列化

**スコープ**: 複数 trial の Rayon 並列実行

**追加 Cargo 依存**:
```toml
[dependencies]
rayon = "1.10"
```

---

## 予想される高速化

| シナリオ | Phase 1 後 | Phase 2 後 | Phase 3 後 |
|---|---|---|---|
| Gaussian フィッティング | **3〜10x** | — | **10〜40x** |
| Lorentzian フィッティング | 1x (対象外) | **2〜5x** | **8〜20x** |
| PseudoVoigt フィッティング | 1x (対象外) | **5〜15x** | **20〜60x** |
| DoniachSunjic フィッティング | 1x (対象外) | **3〜8x** | **12〜30x** |

---

## 検証計画

### 自動テスト

```bash
# Rust モジュールのビルド
cd EMPeaks/_rust_core && maturin develop --release

# 数値一致テスト（Python vs Rust）
python -m pytest tests/test_rust_parity.py -v

# ベンチマーク
python -c "
from EMPeaks.benchmark import compare_backends
from EMPeaks.GaussianMixture import GaussianMixtureModel
import numpy as np

model = GaussianMixtureModel(K=3)
x = np.linspace(-10, 10, 1000)
intensity = np.random.rand(1000)
results = compare_backends(model, x, intensity, method='adapted_em')
"
```

### テスト項目

1. **数値一致**: 同一入力で Python/Rust の出力が `np.allclose(atol=1e-12)` で一致
2. **収束一致**: 同一初期値で両実装が同じ結果に収束
3. **フォールバック**: Rust モジュール未インストール時に Python 実装で動作
4. **切り替え**: `set_backend()` での動的切り替えが正常動作
5. **ベンチマーク**: 各データサイズ × K で実行時間比較
