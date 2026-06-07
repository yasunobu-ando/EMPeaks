# Deterministic Annealing (DA) の Rust 対応状況と実装プラン

## 1. 現在の解析結果：DAは「実質的にRustで高速化」されているが「完全連携」されていない

現在の `_em_core.py` の `deterministic_annealing` の実装を解析した結果、以下のことが判明しました。

1. **EMループ自体の計算はRustで実行されている**
   Python側のDA関数は、温度 `temp` を変えながら内部で `self.fit(x, intensity/temp, method='adapted_em')` をループして呼び出します。この `self.fit()` は自動的に Rust 側の `run_em_loop` に委譲されるため、最も計算負荷の高い最適化プロセス自体はすでにRustによって完全に高速化されています。

2. **他メソッドとの連携が切れている（エラーになる）**
   DAを直接呼び出す分には問題ありませんが、例えば **`sampling(method='deterministic_annealing')` を実行しようとすると、`fit()` メソッド内に DA へのルーティングが存在しないためエラーになります**。また、`sampling` 内のマルチスレッド並列化（RustのGIL解放を利用した並列実行）の対象から外れており、Pythonの直列ループにフォールバックしてしまいます。

## 2. 課題の解決策

温度のループ処理（通常は5〜10回程度）自体は非常に軽いため、Rust側に無理に移植しても計算速度の向上はマイクロ秒単位にとどまります。
真の課題は **「`sampling` 機能などで DA を指定した際に、Rustの高速並列処理エコシステムに正しく乗るようにすること」** です。

そのため、以下の実装プランを提案します。

---

## 3. 実装プラン

### Step 1: `fit` メソッドでの `deterministic_annealing` サポート
`fit` メソッドに `method='deterministic_annealing'` の分岐を追加し、適切にルーティングさせます。

**修正対象:** `EMPeaks/EMCore/_em_core.py` の `fit` 関数
```python
        elif method == 'adapted_em':
            print("**** Start spectrum fitting via EM algorithm ****")
            info = self.adapted_em(x, intensity, max_iter, r_eps, stdout)
            return info

        elif method == 'deterministic_annealing':
            # fit() 経由でも DA を呼び出せるようにする
            info = self.deterministic_annealing(x, intensity, stdout=stdout)
            return info
```

### Step 2: `sampling` のマルチスレッド並列化対象に DA を追加
`sampling` 内の Rust マルチスレッド実行の条件式に `deterministic_annealing` を追加します。
これにより、`trial=50` などで DA を指定した際、50個の独立した DA プロセスが並列で実行されるようになります（GIL解放の恩恵をフルに受けられます）。

**修正対象:** `EMPeaks/EMCore/_em_core.py` の `sampling` 関数
```python
        # 'deterministic_annealing' を追加
        if get_backend() == "rust" and method in ['adapted_em', 'smart', 'deterministic_annealing']:
            import concurrent.futures
            def _run_trial(i):
                # 中略 ... model_copy.fit(...) がマルチスレッドで呼ばれる
```

### Step 3: DA 内部の `fit` 呼び出しループでの標準出力制御
DA は内部で複数回 `fit` を呼び出します。並列実行時に大量のログが標準出力に混ざるのを防ぐため、内部の `fit` 呼び出し時の `stdout` フラグを制御します。

---

## User Review Required

> [!IMPORTANT]
> 上記の解析の通り、数学的・計算量的に最も重い処理はすでに Rust 化されており、ボトルネックはありません。
> したがって、Rust の C++ 的な層に新しい `run_da_loop` を新設するよりも、**Python 側のルーティングと並列処理のディスパッチを修正するアプローチ**が最も効果的かつシンプルだと考えます。
>
> この方針（Python側のディスパッチ修正のみで完全な並列・高速DAを実現する）で進めてよろしいでしょうか？あるいは、温度ループの処理もすべて Rust 側に隠蔽する「完全なるRust移植」をご希望でしょうか？
