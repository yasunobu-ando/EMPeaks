"""
EMPeaks GUI - Peak Fitting Application
Streamlit-based GUI (EMPeaks-Deck Style)
"""

import streamlit as st
from module.data_constructor import data_constructor
from module.model_constructor import model_constructor
from module.fitting_board_constructor import fitting_board_constructor
from module.utils import init_application
from module.i18n import init_language

hide_st_style = """
<style>
footer:after{content: "; Copyright 2025 @ National Institute of Advanced Industrial Science and Technology (AIST)"}
</style>
"""


def main():
    st.set_page_config(layout="wide")
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Initialize application
    init_application()

    # Language selector (top of sidebar)
    init_language()

    # Data Constructor: load data and select variables/range
    chart_db, x_variable = data_constructor()

    # Model Constructor: configure fitting model
    model_param = model_constructor(chart_db, x_variable)

    # FittingBoard Constructor: run fitting and display results
    fitting_board_constructor(chart_db, model_param)


if __name__ == "__main__":
    main()
