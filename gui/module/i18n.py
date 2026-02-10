"""Internationalization (i18n) module for EMPeaks GUI"""
import streamlit as st

# ---------------------------------------------------------------------------
# Translation dictionaries
# ---------------------------------------------------------------------------
TRANSLATIONS = {
    "en": {
        # App
        "app_title": "EMPeaks GUI",
        "app_subtitle": "Spectrum Peak Fitting Application",

        # Sidebar - Language
        "language": "Language",

        # Sidebar - Data Upload
        "data_upload": "Data Upload",

        # Sidebar - EMPeaks Parameters
        "empeaks_parameters": "EMPeaks Parameters",
        "select_mixture_model": "Select Mixture Model",
        "k_component_number": "K (Component Number)",
        "select_background_model": "Select Background Model",
        "input_trial_frequency": "Input Trial Frequency",
        "convergence_threshold": "Convergence threshold for LL",
        "max_iteration": "Max Iteration",

        # Data Dashboard
        "data_dashboard": "Data Dashboard",
        "x_variable": "x variable",
        "y_variable": "y variable",
        "axis": "AXIS",
        "axis_caption": "Integer values only",
        "x_range_slider": "x Range Slider",

        # Fitting Dashboard
        "fitting_dashboard": "Fitting Dashboard",
        "fitting_model": "Fitting Model",
        "background_model": "Background Model",
        "trial_frequency": "Trial Frequency",
        "items": "items",
        "mixture_model": "Mixture Model",
        "peak": "Peak",
        "start_optimization": "START Optimization",
        "refresh": "Refresh",
        "fitting_spinner": "Fitting...",

        # Metrics & Export
        "metrics": "Metrics",
        "optimization_summary": "Optimization Summary",
        "log_likelihood": "Log Likelihood",
        "rmse": "RMSE",
        "time_s": "TIME [s]",
        "export": "Export",
        "download_json": "Download fitting summary(.json)",
        "download_csv": "Download Chart Data (chart_db.csv)",

        # Model parameter descriptions
        "peak_position": "peak position",
        "standard_deviation": "standard deviation",
        "mixing_ratio": "mixing ratio",
        "hwhm": "Half Width Half Maximum",
        "mixing_parameter": "mixing parameter",
        "asymmetry_parameter": "asymmetry parameter",
    },
    "ja": {
        # App
        "app_title": "EMPeaks GUI",
        "app_subtitle": "スペクトル ピークフィッティング アプリケーション",

        # Sidebar - Language
        "language": "言語",

        # Sidebar - Data Upload
        "data_upload": "データアップロード",

        # Sidebar - EMPeaks Parameters
        "empeaks_parameters": "EMPeaks パラメータ",
        "select_mixture_model": "混合モデル選択",
        "k_component_number": "K（ピーク数）",
        "select_background_model": "背景モデル選択",
        "input_trial_frequency": "試行回数",
        "convergence_threshold": "収束閾値",
        "max_iteration": "最大反復回数",

        # Data Dashboard
        "data_dashboard": "データ ダッシュボード",
        "x_variable": "X変数",
        "y_variable": "Y変数",
        "axis": "範囲設定",
        "axis_caption": "整数値のみ対応",
        "x_range_slider": "x範囲",

        # Fitting Dashboard
        "fitting_dashboard": "フィッティング ダッシュボード",
        "fitting_model": "モデル",
        "background_model": "背景",
        "trial_frequency": "試行回数",
        "items": "表示設定",
        "mixture_model": "全体モデル",
        "peak": "Peak",
        "start_optimization": "最適化開始",
        "refresh": "リフレッシュ",
        "fitting_spinner": "フィッティング中...",

        # Metrics & Export
        "metrics": "メトリクス",
        "optimization_summary": "最適化サマリ",
        "log_likelihood": "対数尤度",
        "rmse": "RMSE",
        "time_s": "処理時間 [s]",
        "export": "エクスポート",
        "download_json": "フィッティング結果をダウンロード(.json)",
        "download_csv": "チャートデータをダウンロード(.csv)",

        # Model parameter descriptions
        "peak_position": "ピーク位置",
        "standard_deviation": "標準偏差",
        "mixing_ratio": "混合比率",
        "hwhm": "半値半幅",
        "mixing_parameter": "混合パラメータ",
        "asymmetry_parameter": "非対称パラメータ",
    },
}

LANGUAGE_OPTIONS = {"English": "en", "日本語": "ja"}
LANGUAGE_LABELS = list(LANGUAGE_OPTIONS.keys())        # ["English", "日本語"]


def _on_language_change():
    """Callback – sync the internal lang code when the dropdown changes."""
    label = st.session_state["_lang_select"]           # widget value (label)
    st.session_state["lang"] = LANGUAGE_OPTIONS[label]


def init_language():
    """Initialize language setting in session state and render selector."""
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

    # Compute default index only on first run; afterward Streamlit keeps
    # the widget value via its key.
    if "_lang_select" not in st.session_state:
        # reverse-lookup: code -> label
        code_to_label = {v: k for k, v in LANGUAGE_OPTIONS.items()}
        st.session_state["_lang_select"] = code_to_label[st.session_state["lang"]]

    st.sidebar.selectbox(
        "Language / 言語",
        LANGUAGE_LABELS,
        key="_lang_select",
        on_change=_on_language_change,
    )


def t(key: str) -> str:
    """Return the translated string for *key* in the current language."""
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
