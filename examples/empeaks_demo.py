import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from EMPeaks import GaussianMixture
    return GaussianMixture, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # EMPeaks デモ - Gaussian Mixture Model

    このノートブックでは、EMPeaksパッケージを使ってスペクトルデータのピーク分析を行います。

    EMPeaksは、Spectrum Adapted EMアルゴリズムを使った高スループットピーク分析パッケージです。
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. サンプルデータの生成
    """)
    return


@app.cell
def _(np):
    # エネルギー範囲の設定
    x = np.linspace(0, 100, 1000)

    # 2つのガウシアンピークを持つサンプルデータを生成
    peak1 = 2.0 * np.exp(-((x - 30) ** 2) / (2 * 3**2))
    peak2 = 1.5 * np.exp(-((x - 60) ** 2) / (2 * 5**2))
    noise = np.random.normal(0, 0.05, len(x))

    y = peak1 + peak2 + noise + 0.1  # ノイズと小さなバックグラウンドを追加
    return x, y


@app.cell
def _(mo, plt, x, y):
    # 生データのプロット
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, y, 'k-', linewidth=1, label='Raw Data')
    ax1.set_xlabel('Energy (a.u.)')
    ax1.set_ylabel('Intensity (a.u.)')
    ax1.set_title('Generated Sample Data with Two Gaussian Peaks')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.md(f"""
    ### 生成されたサンプルデータ

    2つのガウシアンピーク（中心: 30と60）を含むサンプルデータを生成しました。
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Gaussian Mixture Modelによるフィッティング
    """)
    return


@app.cell
def _(mo):
    # ピーク数の選択
    k_slider = mo.ui.slider(1, 5, value=2, label="ピーク数 (K):")
    k_slider
    return (k_slider,)


@app.cell
def _(GaussianMixture, k_slider, x, y):
    # ガウス混合モデルの作成とフィッティング
    K = k_slider.value
    gmm = GaussianMixture.GaussianMixtureModel(K=K)
    gmm.fit(x, y)

    # フィッティング結果の取得
    y_fitted = gmm.predict(x)
    return K, y_fitted


@app.cell
def _(K, mo, plt, x, y, y_fitted):
    # フィッティング結果のプロット
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(x, y, 'k-', linewidth=1, alpha=0.7, label='Raw Data')
    ax2.plot(x, y_fitted, 'r-', linewidth=2, label=f'Fitted Model (K={K})')
    ax2.set_xlabel('Energy (a.u.)')
    ax2.set_ylabel('Intensity (a.u.)')
    ax2.set_title(f'Gaussian Mixture Model Fitting (K={K})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.md(f"""
    ### フィッティング結果

    {K}個のガウシアンピークでデータをフィッティングしました。
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. 実データの読み込み（オプション）
    """)
    return


@app.cell
def _(mo, np):
    # リポジトリ内のサンプルデータを読み込み
    try:
        data = np.loadtxt('GFET0126_25V_C1s.txt')
        x_real = data[:, 0]
        y_real = data[:, 1]
        data_loaded = True
        mo.md("""
        ✅ サンプルデータ `GFET0126_25V_C1s.txt` を読み込みました。
        """)
    except Exception as e:
        x_real = None
        y_real = None
        data_loaded = False
        mo.md(f"""
        ⚠️ サンプルデータの読み込みに失敗しました: {str(e)}
        """)
    return data_loaded, x_real, y_real


@app.cell
def _(GaussianMixture, data_loaded, mo, plt, x_real, y_real):
    if data_loaded:
        # 実データでのフィッティング
        gmm_real = GaussianMixture.GaussianMixtureModel(K=3)
        gmm_real.fit(x_real, y_real)
        y_real_fitted = gmm_real.predict(x_real)

        # プロット
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(x_real, y_real, 'k-', linewidth=1, alpha=0.7, label='Real Data')
        ax3.plot(x_real, y_real_fitted, 'r-', linewidth=2, label='Fitted Model')
        ax3.set_xlabel('Binding Energy (eV)')
        ax3.set_ylabel('Intensity (a.u.)')
        ax3.set_title('Real XPS Data Fitting (C1s)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()

        mo.md("""
        ### 実データのフィッティング結果

        XPS（X線光電子分光）のC1sスペクトルデータをフィッティングしました。
        """)
    else:
        mo.md("")
    return


@app.cell
def _(mo):
    mo.md("""
    ## まとめ

    このデモでは以下を確認しました：
    1. サンプルデータの生成とプロット
    2. Gaussian Mixture Modelによるフィッティング
    3. ピーク数をインタラクティブに変更
    4. 実データ（XPSスペクトル）の解析

    スライダーでピーク数を変更して、フィッティングの変化を確認できます！
    """)
    return


if __name__ == "__main__":
    app.run()
