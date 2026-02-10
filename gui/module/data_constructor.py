"""データ読み込み・表示モジュール"""
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from module.utils import refresh


def data_constructor():
    """データダッシュボードを構築"""
    with st.container():
        st.header('📊 Data Dashboard')
        
        # サイドバーでファイルアップロード
        file_uploaded = st.sidebar.file_uploader("データアップロード", type=['csv', 'xlsx'])
        
        if file_uploaded:
            if file_uploaded.name.endswith('.xlsx'):
                st.session_state['Data'] = pd.read_excel(file_uploaded, header=0)
            else:
                st.session_state['Data'] = pd.read_csv(file_uploaded, header=0)
        elif st.session_state['Data'] is None:
            # サンプルデータ生成
            np.random.seed(42)
            x = np.linspace(0, 100, 500)
            y = (50 * np.exp(-((x-30)**2)/(2*5**2)) + 
                 80 * np.exp(-((x-60)**2)/(2*8**2)) + 
                 np.random.normal(0, 2, len(x)))
            st.session_state['Data'] = pd.DataFrame({'Energy': x, 'Intensity': y})
        
        data = st.session_state['Data']
        data_column = data.columns.values
        
        # 3カラムレイアウト
        data_col = st.columns([2, 3, 5])
        
        # 変数選択
        x_variable = data_col[0].selectbox('X変数', data_column, index=0, on_change=refresh)
        y_variable = data_col[0].selectbox('Y変数', data_column, index=min(1, len(data_column)-1), on_change=refresh)
        
        # 範囲スライダー
        data_col[0].markdown("### 📐 範囲設定")
        x_data_min = float(data[x_variable].min())
        x_data_max = float(data[x_variable].max())
        x_min, x_max = data_col[0].slider(
            "X範囲",
            min_value=x_data_min,
            max_value=x_data_max,
            value=(x_data_min, x_data_max),
            on_change=refresh
        )
        
        # データフィルタリング
        chart_db = data.query(f"{x_min} <= `{x_variable}` <= {x_max}")
        
        # データテーブル表示
        data_col[1].dataframe(chart_db, height=385)
        
        # Altairチャート
        data_fig = alt.Chart(chart_db).mark_line(
            color='#1E88E5',
            strokeWidth=2
        ).encode(
            x=alt.X(x_variable, title=x_variable),
            y=alt.Y(y_variable, title=y_variable)
        ).properties(
            height=385
        ).configure_axis(
            labelFontSize=14,
            titleFontSize=16
        )
        data_col[2].altair_chart(data_fig, use_container_width=True)
        
        # 返すデータ
        chart_db = pd.concat([chart_db[x_variable], chart_db[y_variable]], axis=1)
        return chart_db, x_variable
