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
        "same_axis_warning": "X and Y must be different columns.",
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
        "same_axis_warning": "X変数とY変数は異なる列を選択してください。",
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


# ---------------------------------------------------------------------------
# AgGrid Locale (Japanese)
# Reference: https://www.ag-grid.com/javascript-data-grid/localisation/
# ---------------------------------------------------------------------------
AGGRID_LOCALE_JA = {
    # Set Filter
    "selectAll": "(すべて選択)",
    "selectAllSearchResults": "(すべての検索結果を選択)",
    "searchOoo": "検索...",
    "blanks": "(空白)",
    "noMatches": "一致なし",

    # Number Filter & Text Filter
    "filterOoo": "フィルター...",
    "equals": "等しい",
    "notEqual": "等しくない",
    "blank": "空白",
    "notBlank": "空白でない",
    "empty": "選択してください",

    # Number Filter
    "lessThan": "より小さい",
    "greaterThan": "より大きい",
    "lessThanOrEqual": "以下",
    "greaterThanOrEqual": "以上",
    "inRange": "範囲内",
    "inRangeStart": "から",
    "inRangeEnd": "まで",

    # Text Filter
    "contains": "含む",
    "notContains": "含まない",
    "startsWith": "で始まる",
    "endsWith": "で終わる",

    # Date Filter
    "dateFormatOoo": "yyyy-mm-dd",

    # Filter Conditions
    "andCondition": "AND",
    "orCondition": "OR",

    # Header of the Default Group Column
    "group": "グループ",

    # Tooltips
    "columns": "列",
    "filters": "フィルター",
    "pivotMode": "ピボットモード",
    "groups": "行グループ",
    "rowGroupColumnsEmptyMessage": "ここに列をドラッグして行グループを設定",
    "values": "値",
    "valueColumnsEmptyMessage": "ここに列をドラッグして集計",
    "pivots": "列ラベル",
    "pivotColumnsEmptyMessage": "ここに列をドラッグして列ラベルを設定",

    # Header of the Default Group Column
    "group": "グループ",

    # Row Drag
    "rowDragRow": "行",
    "rowDragRows": "行",

    # Other
    "loadingOoo": "読み込み中...",
    "loadingError": "読み込みエラー",
    "noRowsToShow": "表示する行がありません",
    "enabled": "有効",

    # Menu
    "pinColumn": "列を固定",
    "pinLeft": "左に固定",
    "pinRight": "右に固定",
    "noPin": "固定なし",
    "valueAggregation": "集計",
    "autosizeThiscolumn": "この列の幅を自動調整",
    "autosizeAllColumns": "すべての列の幅を自動調整",
    "groupBy": "グループ化",
    "ungroupBy": "グループ化解除",
    "addToValues": "${variable} を値に追加",
    "removeFromValues": "${variable} を値から削除",
    "addToLabels": "${variable} をラベルに追加",
    "removeFromLabels": "${variable} をラベルから削除",
    "resetColumns": "列のリセット",
    "expandAll": "すべて展開",
    "collapseAll": "すべて折りたたむ",
    "copy": "コピー",
    "ctrlC": "Ctrl+C",
    "copyWithHeaders": "ヘッダー付きでコピー",
    "copyWithGroupHeaders": "グループヘッダー付きでコピー",
    "paste": "貼り付け",
    "ctrlV": "Ctrl+V",
    "export": "エクスポート",
    "csvExport": "CSVエクスポート",
    "excelExport": "Excelエクスポート",

    # Enterprise Menu Aggregation and Status Bar
    "sum": "合計",
    "min": "最小",
    "max": "最大",
    "none": "なし",
    "count": "カウント",
    "avg": "平均",
    "filteredRows": "フィルター済み",
    "selectedRows": "選択済み",
    "totalRows": "総行数",
    "totalAndFilteredRows": "行",
    "more": "詳細",
    "to": "-",
    "of": "/",
    "page": "ページ",
    "nextPage": "次のページ",
    "lastPage": "最後のページ",
    "firstPage": "最初のページ",
    "previousPage": "前のページ",
}


# ---------------------------------------------------------------------------
# CSS Hacks for File Uploader Localization
# ---------------------------------------------------------------------------
FILE_UPLOADER_CSS = """
<style>
/* ── File Uploader: hide original English text ── */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    display: none;
}

/* ── Inject Japanese drag-and-drop instruction ── */
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {
    content: "ここにドラッグ＆ドロップ";
    font-size: 0.875rem;
    display: block;
    margin-bottom: 4px;
    color: inherit;
}

/* ── Inject Japanese limit text ── */
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
    content: "制限 200MB • CSV, XLSX";
    font-size: 0.75rem;
    display: block;
    color: rgba(49, 51, 63, 0.6);
}

/* ── Button: hide original English text, inject Japanese ── */
[data-testid="stFileUploaderDropzone"] button {
    position: relative;
    color: transparent !important;
    min-width: 0 !important;
    padding: 0.375rem 1rem !important;
}

[data-testid="stFileUploaderDropzone"] button > * {
    visibility: hidden;
}

[data-testid="stFileUploaderDropzone"] button::after {
    content: "ファイルを選択";
    font-size: 0.875rem;
    color: #31333F;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    white-space: nowrap;
    visibility: visible;
}

/* ── Dark mode adjustments ── */
[data-testid="stApp"][data-theme="dark"]
    [data-testid="stFileUploaderDropzoneInstructions"] > div::after {
    color: rgba(250, 250, 250, 0.5);
}

[data-testid="stApp"][data-theme="dark"]
    [data-testid="stFileUploaderDropzone"] button::after {
    color: #fafafa;
}

@media (prefers-color-scheme: dark) {
    [data-testid="stFileUploaderDropzoneInstructions"] > div::after {
        color: rgba(250, 250, 250, 0.5);
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        color: #fafafa;
    }
}
</style>
"""
