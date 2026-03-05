# EMPeaks GUI

EMPeaksのピークフィッティング機能をGUIで操作するためのStreamlitアプリケーションです。

## 起動方法

```bash
# 依存関係のインストール
pip install -r requirements-gui.txt

# アプリの起動
streamlit run gui/app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## 機能

- 📁 **データ入力**: CSVファイルのアップロード or サンプルデータ
- ⚙️ **パラメータ設定**: ピーク数、背景モデル、手法の選択
- 📈 **可視化**: フィッティング結果のグラフ表示
- 💾 **エクスポート**: CSV/JSON形式での結果保存
