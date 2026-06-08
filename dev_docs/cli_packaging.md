# CLIパッケージング構成と iCloud Drive 問題

## 背景

本プロジェクトは当初 `setup.cfg` + `setuptools` + `console_scripts` 方式でCLIを提供していたが、以下の理由で `pyproject.toml` + `maturin` + `script-files` 方式に移行した。

1. `setup.cfg` は setuptools の旧フォーマット。現代のPythonプロジェクトでは `pyproject.toml` が標準 (PEP 517/518/621)。
2. `setuptools` editable install は `.pth` ファイル（plain pathのみ記載）を生成し、macOS iCloud Drive 環境でこれが機能しなくなる問題がある（後述）。
3. 本プロジェクトは `EMPeaks/_rust_core/` に PyO3 ベースの Rust 拡張を持つため、maturin への統合が自然。

---

## iCloud Drive と `.pth` 問題

### 発生条件

- リポジトリが `~/Documents/` 以下に置かれている（iCloud Drive の同期対象）
- `.venv` がプロジェクト内に作成されている

### 根本原因

macOS の **File Provider** デーモン（iCloud Drive の管理プロセス）は、`~/Documents/` 以下の `.venv/lib/*/site-packages/` 内のファイルに `UF_HIDDEN` フラグ (BSD flag `0x8000`) を自動付与する。

Python の `site` モジュール (`addpackage()` 関数) はこのフラグを検査し、フラグが立ったファイルをスキップする:

```python
# CPython site.py (Python 3.11+)
if ((getattr(st, 'st_flags', 0) & stat.UF_HIDDEN) or ...):
    _trace(f"Skipping hidden .pth file: {fullname!r}")
    return
```

`setuptools` の `console_scripts` が生成するエントリポイントスクリプトは `from cli.main import main` を先頭近くで実行するが、`.pth` がスキップされているため `cli` パッケージが見つからない。

```
ModuleNotFoundError: No module named 'cli'
```

### 再現条件

`~/Documents/` 以下に `.venv` を置く限り、他の Mac でも同様に発生する。`chflags nohidden` でフラグを手動解除しても、iCloud デーモンが再設定するため根本解決にならない。

---

## 解決策: maturin + `script-files` + セルフリゾルブ方式

### ビルドバックエンドを maturin に変更した理由

- **`setuptools` editable install**: `.pth` ファイルに plain path を記載 → iCloud により `UF_HIDDEN` → スキップ → `ModuleNotFoundError`
- **`maturin` editable install**: Rust 拡張 (`.so`) を site-packages に**直接コピー** → `.pth` 不要。Python パッケージは後述の `bin/empeaks` で解決

### `bin/empeaks` によるセルフリゾルブ

`pyproject.toml` で `script-files = ["bin/empeaks"]` を指定。`bin/empeaks` はソースとして管理するPythonスクリプトで、`__file__` から自身の位置を計算してプロジェクトルートを `sys.path` に追加する:

```python
#!/usr/bin/env python3
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
del _os, _root
from cli.main import main
main()
```

`pip install` / `maturin develop` 時にこのスクリプトが `.venv/bin/empeaks` にコピーされ、shebang が venv の Python パスに書き換えられる。コピー先でのパス計算:

```
.venv/bin/empeaks
  dirname x1 → .venv/bin/
  dirname x2 → .venv/
  dirname x3 → プロジェクトルート ✓
```

### `.pth` フリーになる仕組み

```
empeaks deck 実行時の import 解決フロー:

.venv/bin/empeaks (script-files でコピー)
  └─ sys.path.insert(0, project_root)     ← .pth 不要
       ├─ from cli.main import main        ← project_root/cli/ で解決
       └─ import gui (from _launch_deck)   ← project_root/gui/ で解決

empeaks_rust_core (Rust .so)
  └─ site-packages/empeaks_rust_core.so   ← maturin が直接コピー、.pth 不要
```

---

## インストール手順

```bash
# Makefile を使う場合
make install

# 直接実行する場合
maturin develop
```

---

## 運用上の注意

### `bin/empeaks` を変更した場合

`script-files` 方式ではスクリプトがインストール時にコピーされる（editable install でもシンボリックリンクではない）。`bin/empeaks` の内容を変更した場合は再インストールが必要:

```bash
maturin develop
```

### venv を iCloud 外に置く場合 (オプション)

環境変数 `UV_PROJECT_ENVIRONMENT` で venv の場所を指定することで、iCloud 同期対象外に置ける:

```bash
# ~/.zshrc に追加
export UV_PROJECT_ENVIRONMENT=~/.venvs/EMPeaks
```
