"""フィッティングダッシュボードモジュール"""
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import json
from module.utils import refresh, run


def fitting_board_constructor(db, param):
    """フィッティングダッシュボードを構築"""
    fitting_summary = {}
    K = st.session_state['K']
    data_column = db.columns.values
    
    with st.container():
        st.header("🔬 Fitting Dashboard")
        
        # 設定概要行
        set_col = st.columns([5, 2, 2, 2])
        set_col[0].metric("モデル", st.session_state['MixtureModel'])
        set_col[1].metric("K", st.session_state['K'])
        set_col[2].metric("背景", st.session_state['BackgroundModel'])
        set_col[3].metric("試行回数", st.session_state['TrialFrequency'])
        
        # チャートとコントロール
        chart_col = st.columns([1, 5])
        
        # 左側: コントロール
        chart_col[0].subheader("表示設定")
        check_plot = {
            'Mixture Model': chart_col[0].checkbox('全体モデル', value=True),
            'Peaks': [chart_col[0].checkbox(f'Peak #{k+1}') for k in range(K)],
            'Background Model': False
        }
        if st.session_state['BackgroundModel'] != 'none':
            check_plot['Background Model'] = chart_col[0].checkbox('背景モデル')
        
        # ボタン
        if chart_col[0].button('▶️ 最適化開始', type='primary'):
            st.session_state['Fitted'] = True
        if chart_col[0].button('🔄 リフレッシュ'):
            refresh()
            st.rerun()
        
        # フィッティング実行
        if st.session_state['Fitted'] and st.session_state['ModelInstance'] is not None:
            with st.spinner('フィッティング中...'):
                st.session_state['ModelInstance'], info = run(
                    st.session_state['ModelInstance'],
                    db[data_column[0]],
                    db[data_column[1]],
                    st.session_state['TrialFrequency']
                )
            
            ref_id = info['index_best']
            
            # メトリクス表示
            metric_col = st.columns(len(param) + 2)
            metric_col[0].subheader('📊 結果')
            metric_col[0].metric('Log Likelihood', f"{info['LL_hist'][ref_id]:.4e}")
            metric_col[0].metric('RMSE', f"{info['RMSE_hist'][ref_id]:.4e}")
            metric_col[0].metric('TIME [s]', f"{info['time_hist'][ref_id]:.4f}")
            
            fitting_summary['Log Likelihood'] = info['LL_hist'][ref_id]
            fitting_summary['RMSE'] = info['RMSE_hist'][ref_id]
            
            # パラメータ表示
            model = st.session_state['ModelInstance']
            param_val = model.export_param()
            for i, param_name in enumerate(param):
                metric_col[i+1].subheader(param_name)
                metric_col[i+1].caption(param[param_name])
                for k in range(K):
                    if param_name in param_val:
                        metric_col[i+1].metric(f'Peak #{k+1}', f"{param_val[param_name][k]:.3f}")
        
        # チャート作成
        mi = st.session_state['ModelInstance']
        x = db[data_column[0]]
        y = db[data_column[1]]
        
        # ベースデータ
        chart_data = pd.DataFrame({
            'x': x,
            'Raw Data': y
        })
        
        # Altairチャート（生データ）
        base_chart = alt.Chart(chart_data).mark_line(
            color='gray',
            strokeWidth=3
        ).encode(
            x=alt.X('x', title=data_column[0]),
            y=alt.Y('Raw Data', title=data_column[1])
        )
        
        chart = base_chart
        
        # フィッティング結果の描画
        if st.session_state['Fitted'] and mi is not None:
            if mi.N_tot is None or mi.N_tot == 0:
                mi.N_tot = y.sum()
            
            # 全体モデル
            if check_plot['Mixture Model']:
                y_model = mi.predict(x.values) * mi.N_tot
                model_data = pd.DataFrame({'x': x, 'Mixture Model': y_model})
                model_chart = alt.Chart(model_data).mark_line(
                    color='blueviolet',
                    strokeWidth=2
                ).encode(x='x', y='Mixture Model')
                chart = chart + model_chart
            
            # 各ピーク
            for k in range(K):
                if check_plot['Peaks'][k]:
                    y_peak = mi.model[k].predict(x.values) * mi.pi[k] * mi.N_tot
                    peak_data = pd.DataFrame({'x': x, f'Peak #{k+1}': y_peak})
                    peak_chart = alt.Chart(peak_data).mark_line(
                        strokeDash=[5, 5],
                        strokeWidth=2
                    ).encode(x='x', y=f'Peak #{k+1}')
                    chart = chart + peak_chart
        
        chart = chart.properties(height=400).configure_axis(labelFontSize=14, titleFontSize=16)
        chart_col[1].altair_chart(chart, use_container_width=True)
        
        # エクスポート
        if st.session_state['Fitted']:
            _export_data(fitting_summary, db)


def _export_data(summary, db):
    """データエクスポート"""
    st.divider()
    export_col = st.columns(2)
    
    export_col[0].download_button(
        '📄 JSON形式でダウンロード',
        data=json.dumps(summary, indent=4, default=str),
        file_name='fitting_summary.json',
        mime='application/json',
        use_container_width=True
    )
    
    export_col[1].download_button(
        '📊 CSV形式でダウンロード',
        data=db.to_csv(index=False).encode('utf-8'),
        file_name='chart_data.csv',
        mime='text/csv',
        use_container_width=True
    )
