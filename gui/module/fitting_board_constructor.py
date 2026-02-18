"""Fitting dashboard module"""
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import json
from module.utils import refresh, run
from module.i18n import t


def fitting_board_constructor(db, param):
    """Build the Fitting Dashboard"""
    fitting_summary = {}
    K = st.session_state['K']
    data_column = db.columns.values

    with st.container():
        st.header(t("fitting_dashboard"))

        # Settings summary row
        set_col = st.columns([5, 2, 2, 2])
        set_col[0].metric(t("fitting_model"), st.session_state['MixtureModel'])
        set_col[1].metric("K", st.session_state['K'])
        set_col[2].metric(t("background_model"), st.session_state['BackgroundModel'])
        set_col[3].metric(t("trial_frequency"), st.session_state['TrialFrequency'])

        # Chart area with controls
        chart_col = st.columns([1, 5])

        # Left column: display items
        chart_col[0].subheader(t("items"))
        check_plot = {
            'Mixture Model': chart_col[0].checkbox(t("mixture_model"), value=True),
            'Peaks': [chart_col[0].checkbox(f'{t("peak")} #{k}') for k in range(K)],
            'Background Model': False
        }
        if st.session_state['BackgroundModel'] != 'none':
            check_plot['Background Model'] = chart_col[0].checkbox(t("background_model"))

        # Buttons
        if chart_col[0].button(t("start_optimization")):
            st.session_state['Fitted'] = True
        if chart_col[0].button(t("refresh")):
            refresh()
            st.rerun()

        # Run fitting
        if st.session_state['Fitted'] and st.session_state['ModelInstance'] is not None:
            with st.spinner(t("fitting_spinner")):
                st.session_state['ModelInstance'], info = run(
                    st.session_state['ModelInstance'],
                    db[data_column[0]],
                    db[data_column[1]],
                    st.session_state['TrialFrequency']
                )

            ref_id = info['index_best']

            # Metrics row
            metric_col = st.columns(len(param) + 2)
            metric_col[0].subheader(t("metrics"))
            metric_col[0].caption(t("optimization_summary"))
            metric_col[0].metric(t("log_likelihood"), "{:8.6e}".format(info['LL_hist'][ref_id]))
            metric_col[0].metric(t("rmse"), "{:8.6e}".format(info['RMSE_hist'][ref_id]))
            metric_col[0].metric(t("time_s"), "{:6.4f}".format(info['time_hist'][ref_id]))

            fitting_summary[t("log_likelihood")] = info['LL_hist'][ref_id]
            fitting_summary[t("rmse")] = info['RMSE_hist'][ref_id]
            fitting_summary[t("time_s")] = info['time_hist'][ref_id]

            # Parameter display
            i = 0
            model = st.session_state['ModelInstance']
            param_val = model.export_param()
            for param_name in param:
                i += 1
                metric_col[i].subheader(param_name)
                metric_col[i].caption(param[param_name])
                if param_name in param_val:
                    for k in range(K):
                        metric_col[i].metric(f'Model #{k}', "{:5.3f}".format(param_val[param_name][k]))
                fitting_summary[param_name] = list(param[param_name]) if isinstance(param[param_name],
                                                                                     list) else param[param_name]

        # Build chart
        mi = st.session_state['ModelInstance']
        if st.session_state['Fitted'] is False and mi is not None:
            try:
                src = st.session_state['Data']
                mi.N_tot = src.iloc[:, 1].sum() if len(src.columns) > 1 else src.iloc[:, 0].sum()
            except Exception:
                mi.N_tot = db.iloc[:, 1].sum() if len(db.columns) > 1 else db.iloc[:, 0].sum()

        # Safely extract x/y as Series (use iloc to avoid duplicate-column issues)
        x_data = pd.Series(db.iloc[:, 0].values, name=data_column[0])
        y_data = pd.Series(db.iloc[:, 1].values, name=data_column[1])
        fitting_summary['Energy'] = x_data.tolist()
        fitting_summary['Intensity'] = y_data.tolist()

        # Base chart: raw data (gray, thick)
        tmp_db = pd.concat([x_data, y_data], axis=1)
        chart = alt.Chart(tmp_db).mark_line(
            color='gray', strokeWidth=8
        ).encode(
            x=data_column[0],
            y=data_column[1]
        )

        # Mixture model overlay (blueviolet)
        if st.session_state['Fitted'] and mi is not None:
            tmp_db2 = x_data.to_frame()
            if check_plot['Mixture Model']:
                y_model = pd.Series(mi.predict(x.values) * mi.N_tot, name='mixture_model', index=x_data.index)
                fitting_summary['Mixture Model'] = y_model.tolist()
                tmp_db2 = pd.concat([tmp_db2, y_model], axis=1)
                chart = chart + alt.Chart(tmp_db2).mark_line(
                    color='blueviolet', strokeWidth=3
                ).encode(x=data_column[0], y='mixture_model')

            # Individual peaks (dashed)
            tmp_peaks = x_data.to_frame()
            flag = False
            for k in range(K):
                if check_plot['Peaks'][k]:
                    flag = True
                    y_peak = pd.Series(mi.model[k].predict(x.values) * mi.pi[k] * mi.N_tot,
                                       name=f"Peak_#{k}")
                    fitting_summary[f'Peak #{k:02d}'] = y_peak.tolist()
                    tmp_peaks = pd.concat([tmp_peaks, y_peak], axis=1)
            if flag:
                tmp_peaks_long = tmp_peaks.melt(id_vars=data_column[0], var_name='data_type')
                chart += alt.Chart(tmp_peaks_long).mark_line(
                    strokeDash=[5, 5], strokeWidth=3
                ).encode(
                    x=data_column[0],
                    y='value',
                    color='data_type:N'
                )

            # Background model
            if check_plot['Background Model'] and len(mi.model) > K:
                y_bg = pd.Series(mi.model[-1].predict(x.values) * mi.pi[-1] * mi.N_tot,
                                 name=f"BG: {st.session_state['BackgroundModel']}")
                fitting_summary[f"BG: {st.session_state['BackgroundModel']}"] = y_bg.tolist()

        # Render chart
        chart = chart.properties(height=400).configure_axis(labelFontSize=15, titleFontSize=24)
        chart_col[1].altair_chart(chart, use_container_width=True)

        # Export section
        if st.session_state['Fitted']:
            _export_data(metric_col, fitting_summary, db)


def _export_data(col, summary, db):
    """Export data buttons"""
    col[-1].subheader(t("export"))
    col[-1].download_button(
        t("download_json"),
        data=json.dumps(summary, indent=4, default=str),
        file_name='fitting_summary.json',
        mime='application/json'
    )
    col[-1].download_button(
        t("download_csv"),
        data=db.to_csv(index=False).encode('utf-8'),
        file_name='chart_db.csv',
        mime='text/csv'
    )
