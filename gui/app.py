"""
EMPeaks GUI - Peak Fitting Application
Streamlit-based GUI for EMPeaks library (EMPeaks-Deck Style)
"""

import streamlit as st
from module.data_constructor import data_constructor
from module.model_constructor import model_constructor
from module.fitting_board_constructor import fitting_board_constructor
from module.utils import init_application

# CSSスタイル
hide_st_style = """
<style>
    footer:after {
        content: "Copyright 2025 @ National Institute of Advanced Industrial Science and Technology (AIST)";
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1rem;
    }
</style>
"""


def main():
    # ページ設定
    st.set_page_config(
        page_title="EMPeaks GUI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(hide_st_style, unsafe_allow_html=True)
    
    # ヘッダー
    st.markdown('<div class="main-header">📊 EMPeaks GUI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Spectrum Peak Fitting Application</div>', unsafe_allow_html=True)
    
    # アプリケーションの初期化
    init_application()
    
    # Data Constructor: ファイルからデータを読み込み、解析する変数と範囲を指定
    chart_db, x_variable = data_constructor()
    
    st.divider()
    
    # Model Constructor: フィッティングモデルの初期設定
    model_param = model_constructor(chart_db, x_variable)
    
    # FittingBoard Constructor: フィッティングを実行・結果を表示
    fitting_board_constructor(chart_db, model_param)
    
    # フッター
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
        "EMPeaks GUI v0.2.0 | Powered by Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
