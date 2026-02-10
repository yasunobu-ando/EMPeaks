"""Data loading and display module"""
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from module.utils import refresh
from module.i18n import t


def data_constructor():
    """Build the Data Dashboard"""
    with st.container():
        st.header(t("data_dashboard"))

        # Sidebar: file upload
        file_uploaded = st.sidebar.file_uploader(t("data_upload"), type=['csv', 'xlsx'])

        if file_uploaded:
            if file_uploaded.name.endswith('.xlsx'):
                st.session_state['Data'] = pd.read_excel(file_uploaded, header=0)
            else:
                st.session_state['Data'] = pd.read_csv(file_uploaded, header=0)
        elif st.session_state['Data'] is None:
            # Generate sample data
            np.random.seed(42)
            x = np.linspace(0, 100, 500)
            y = (50 * np.exp(-((x - 30) ** 2) / (2 * 5 ** 2)) +
                 80 * np.exp(-((x - 60) ** 2) / (2 * 8 ** 2)) +
                 np.random.normal(0, 2, len(x)))
            st.session_state['Data'] = pd.DataFrame({'Energy': x, 'Intensity': y})

        data = st.session_state['Data']
        data_column = data.columns.values

        # 3-column layout: [controls, table, chart]
        data_col = st.columns([2, 3, 5])

        # Column 0: variable selection
        x_variable = data_col[0].selectbox(t("x_variable"), data_column, index=0, on_change=refresh)
        y_variable = data_col[0].selectbox(t("y_variable"), data_column, index=min(1, len(data_column) - 1),
                                           on_change=refresh)

        # Axis range
        data_col[0].markdown(f"### *{t('axis')}*")
        data_col[0].caption(t("axis_caption"))
        x_data_min = int(data[x_variable].min())
        x_data_max = int(data[x_variable].max())
        x_min, x_max = data_col[0].slider(
            t("x_range_slider"),
            min_value=x_data_min,
            max_value=x_data_max,
            value=(x_data_min, x_data_max),
            on_change=refresh
        )

        # Filter data
        chart_db = data.query(f"{x_min} <= `{x_variable}` <= {x_max}")

        # Column 1: data table
        data_col[1].dataframe(chart_db, height=385)

        # Column 2: Altair chart
        data_fig = alt.Chart(chart_db) \
            .mark_line().encode(x=x_variable, y=y_variable) \
            .properties(height=385) \
            .configure_axis(labelFontSize=15, titleFontSize=24)
        data_col[2].altair_chart(data_fig, use_container_width=True)

        chart_db = pd.concat([chart_db[x_variable], chart_db[y_variable]], axis=1)
        return chart_db, x_variable
