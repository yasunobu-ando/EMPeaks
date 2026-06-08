"""Data loading and display module"""
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from module.utils import refresh
from module.i18n import t, AGGRID_LOCALE_JA, FILE_UPLOADER_CSS


def data_constructor():
    """Build the Data Dashboard"""
    with st.container():
        st.header(t("data_dashboard"))

        # Check current language
        lang = st.session_state.get("lang", "en")

        # Inject CSS for File Uploader if language is Japanese
        if lang == "ja":
            st.markdown(FILE_UPLOADER_CSS, unsafe_allow_html=True)

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
            y_true = (50 * np.exp(-((x - 30) ** 2) / (2 * 5 ** 2)) +
                      80 * np.exp(-((x - 60) ** 2) / (2 * 8 ** 2)))
            dx = x[1] - x[0]
            y_true = y_true * (10000 / (y_true.sum() * dx))
            y = np.random.poisson(y_true).astype(float)
            st.session_state['Data'] = pd.DataFrame({'Energy': x, 'Intensity': y})

        data = st.session_state['Data']
        # Deduplicate column names so selectbox / indexing works reliably
        if data.columns.duplicated().any():
            cols = data.columns.tolist()
            seen: dict[str, int] = {}
            new_cols: list[str] = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            data.columns = new_cols
            st.session_state['Data'] = data
        data_column = data.columns.values

        # 3-column layout: [controls, table, chart]
        data_col = st.columns([2, 3, 5])

        # Column 0: variable selection
        x_variable = data_col[0].selectbox(t("x_variable"), data_column, index=0, on_change=refresh)

        # Filter Y options: exclude the selected X variable
        y_options = [c for c in data_column if c != x_variable]
        if len(y_options) == 0:
            # Only 1 column in data – fall back to same column
            data_col[0].warning(t("same_axis_warning"))
            y_variable = x_variable
        else:
            y_variable = data_col[0].selectbox(t("y_variable"), y_options, index=0, on_change=refresh)

        # Axis range
        data_col[0].markdown(f"### *{t('axis')}*")
        data_col[0].caption(t("axis_caption"))
        x_col = data[x_variable] if isinstance(data[x_variable], pd.Series) else data[x_variable].iloc[:, 0]
        x_data_min = int(x_col.min())
        x_data_max = int(x_col.max())
        x_min, x_max = data_col[0].slider(
            t("x_range_slider"),
            min_value=x_data_min,
            max_value=x_data_max,
            value=(x_data_min, x_data_max),
            on_change=refresh
        )

        # Filter data
        chart_db = data.query(f"{x_min} <= `{x_variable}` <= {x_max}")

        # Column 1: data table (AgGrid)
        with data_col[1]:
            gb = GridOptionsBuilder.from_dataframe(chart_db)
            gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
            gb.configure_side_bar()
            
            # Apply Japanese locale if selected
            if lang == "ja":
                gb.configure_grid_options(localeText=AGGRID_LOCALE_JA)
                
            gridOptions = gb.build()

            AgGrid(
                chart_db,
                gridOptions=gridOptions,
                enable_enterprise_modules=False,
                height=385,
                width='100%',
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=False,
            )

        # Column 2: Altair chart
        data_fig = alt.Chart(chart_db) \
            .mark_line().encode(x=x_variable, y=y_variable) \
            .properties(height=385) \
            .configure_axis(labelFontSize=15, titleFontSize=24)
        data_col[2].altair_chart(data_fig, width='stretch')

        # Build a clean 2-column DataFrame for downstream use
        x_series = chart_db[x_variable] if isinstance(chart_db[x_variable], pd.Series) else chart_db[x_variable].iloc[:, 0]
        y_series = chart_db[y_variable] if isinstance(chart_db[y_variable], pd.Series) else chart_db[y_variable].iloc[:, 0]
        chart_db = pd.DataFrame({x_variable: x_series.values, y_variable: y_series.values})
        return chart_db, x_variable
