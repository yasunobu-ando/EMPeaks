"""Fitting dashboard module"""
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import json
from module.utils import run, init_model
from module.i18n import t


def fitting_board_constructor(db, param):
    """Build the Fitting Dashboard"""
    fitting_summary = {}
    K = st.session_state['K']
    data_column = db.columns.values

    # Execute fitting if triggered from sidebar button
    if st.session_state.get('TriggerOptimization'):
        st.session_state['TriggerOptimization'] = False
        with st.spinner(t("fitting_spinner")):
            x_min = db[data_column[0]].min()
            x_max = db[data_column[0]].max()
            st.session_state['ModelInstance'] = init_model(st.session_state['K'], x_min, x_max)
            st.session_state['ModelInstance'], info = run(
                st.session_state['ModelInstance'],
                db[data_column[0]],
                db[data_column[1]],
                st.session_state['TrialFrequency']
            )
        st.session_state['Fitted'] = True
        st.session_state['FitInfo'] = info
        st.session_state['FitParam'] = param  # snapshot of param names at fit time

    mi = st.session_state['ModelInstance']
    is_fitted = st.session_state['Fitted']
    display_K = mi.K if (is_fitted and mi is not None and hasattr(mi, 'K')) else K
    fit_param = st.session_state.get('FitParam') or param

    with st.container():
        st.header(t("fitting_dashboard"))

        # Settings summary row
        set_col = st.columns([4, 2, 2, 2, 2])
        set_col[0].metric(t("fitting_model"), st.session_state['MixtureModel'])
        set_col[1].metric("K", st.session_state['K'])
        set_col[2].metric(t("background_model"), st.session_state['BackgroundModel'])
        set_col[3].metric(t("trial_frequency"), st.session_state['TrialFrequency'])
        set_col[4].metric(t("fitting_method"), st.session_state.get('FittingMethod', 'sampling'))

        # Chart area with controls
        chart_col = st.columns([1, 5])

        # Left column: display items
        chart_col[0].subheader(t("items"))
        check_plot = {
            'Mixture Model': chart_col[0].checkbox(t("mixture_model"), value=True),
            'Peaks': [chart_col[0].checkbox(f'{t("peak")} #{k}') for k in range(display_K)],
            'Background Model': False
        }
        if st.session_state['BackgroundModel'] != 'none':
            check_plot['Background Model'] = chart_col[0].checkbox(t("background_model"))

        # Display fitting results
        if is_fitted and st.session_state.get('FitInfo') is not None:
            info = st.session_state['FitInfo']
            ref_id = info['index_best']

            # Metrics row
            metric_col = st.columns(len(fit_param) + 2)
            metric_col[0].subheader(t("metrics"))
            metric_col[0].caption(t("optimization_summary"))
            metric_col[0].metric(t("log_likelihood"), "{:8.6e}".format(info['LL_hist'][ref_id]))
            metric_col[0].metric(t("rmse"), "{:8.6e}".format(info['RMSE_hist'][ref_id]))
            metric_col[0].metric(t("time_s"), "{:6.4f}".format(info['time_hist'][ref_id]))

            fitting_summary[t("log_likelihood")] = info['LL_hist'][ref_id]
            fitting_summary[t("rmse")] = info['RMSE_hist'][ref_id]
            fitting_summary[t("time_s")] = info['time_hist'][ref_id]

            # Parameter display using fit_param snapshot
            i = 0
            model = st.session_state['ModelInstance']
            param_val = model.export_param()
            for param_name in fit_param:
                i += 1
                metric_col[i].subheader(param_name)
                metric_col[i].caption(fit_param[param_name])
                if param_name in param_val:
                    for k in range(display_K):
                        val = param_val[param_name][k]
                        val = float(np.ravel(val)[0]) if hasattr(val, '__len__') else float(val)
                        if abs(val) < 0.01 or abs(val) > 1e6:
                            fmt = "{:.3e}"
                        else:
                            fmt = "{:5.3f}"
                        metric_col[i].metric(f'Model #{k}', fmt.format(val))
                fitting_summary[param_name] = list(fit_param[param_name]) if isinstance(
                    fit_param[param_name], list) else fit_param[param_name]

        # Build chart
        if not is_fitted and mi is not None:
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
        if is_fitted and mi is not None:
            tmp_db2 = x_data.to_frame()
            if check_plot['Mixture Model']:
                y_model = pd.Series(mi.predict(x_data.values) * mi.N_tot, name='mixture_model', index=x_data.index)
                fitting_summary['Mixture Model'] = y_model.tolist()
                tmp_db2 = pd.concat([tmp_db2, y_model], axis=1)
                chart = chart + alt.Chart(tmp_db2).mark_line(
                    color='blueviolet', strokeWidth=3
                ).encode(x=data_column[0], y='mixture_model')

            # Individual peaks + background
            tmp_series = x_data.to_frame()
            has_series = False

            for k in range(display_K):
                if check_plot['Peaks'][k]:
                    has_series = True
                    y_peak = pd.Series(mi.model[k].predict(x_data.values) * mi.pi[k] * mi.N_tot,
                                       name=f"Peak_#{k}")
                    fitting_summary[f'Peak #{k:02d}'] = y_peak.tolist()
                    tmp_series = pd.concat([tmp_series, y_peak], axis=1)

            if check_plot['Background Model'] and len(mi.model) > display_K:
                has_series = True
                bg_label = f"BG: {st.session_state['BackgroundModel']}"
                bg_col = f"BG_{st.session_state['BackgroundModel']}"
                y_bg = pd.Series(mi.model[-1].predict(x_data.values) * mi.pi[-1] * mi.N_tot,
                                 name=bg_col)
                fitting_summary[bg_label] = y_bg.tolist()
                tmp_series = pd.concat([tmp_series, y_bg], axis=1)

            if has_series:
                tmp_series_long = tmp_series.melt(id_vars=data_column[0], var_name='data_type')
                chart += alt.Chart(tmp_series_long).mark_line(
                    strokeDash=[5, 5], strokeWidth=3
                ).encode(
                    x=data_column[0],
                    y='value',
                    color='data_type:N'
                )

        # Render chart
        chart = chart.properties(height=400).configure_axis(labelFontSize=15, titleFontSize=24)
        chart_col[1].altair_chart(chart, width='stretch')

        # Export section
        if is_fitted:
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
