# EMPeaks GUI

EMPeaksのピークフィッティング機能をGUIで操作するためのStreamlitアプリケーションです。

## 起動方法

```bash
# 依存関係のインストール
pip install -r requirements-gui.txt

# アプリの起動
streamlit run gui/app.py
```

### CLIコマンドで起動する（推奨）

EMPeaks をインストール済みの場合は、以下のコマンドでも起動できます。

```bash
pip install -e .   # 初回のみ
empeaks deck
```

ポートを変更する場合:

```bash
empeaks deck --port 8502
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## 機能

- 📁 **データ入力**: CSVファイルのアップロード or サンプルデータ
- ⚙️ **パラメータ設定**: ピーク数、背景モデル、手法の選択
- 📈 **可視化**: フィッティング結果のグラフ表示
- 💾 **エクスポート**: CSV/JSON形式での結果保存
