"""
EMPeaks GUI - Peak Fitting Application
Streamlit-based GUI for EMPeaks library
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from io import StringIO
import sys
import os

# Configure matplotlib for Japanese fonts
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'Yu Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# Add parent directory to path for EMPeaks import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from EMPeaks.GaussianMixture import GaussianMixtureModel
    EMPEAKS_AVAILABLE = True
except ImportError:
    EMPEAKS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="EMPeaks GUI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 EMPeaks GUI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Spectrum Peak Fitting Application</div>', unsafe_allow_html=True)

# Check EMPeaks availability
if not EMPEAKS_AVAILABLE:
    st.warning("⚠️ EMPeaks ライブラリが見つかりません。デモモードで動作します。")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'fitted' not in st.session_state:
    st.session_state.fitted = False
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar
with st.sidebar:
    st.header("⚙️ パラメータ設定")
    
    # File upload section
    st.subheader("📁 データ入力")
    
    data_source = st.radio(
        "データソース",
        ["サンプルデータ", "ファイルアップロード"],
        horizontal=True
    )
    
    if data_source == "サンプルデータ":
        sample_type = st.selectbox(
            "サンプルタイプ",
            ["2ピーク（シンプル）", "3ピーク（複雑）", "背景あり"]
        )
        
        # Generate sample data
        np.random.seed(42)
        x = np.linspace(0, 100, 500)
        
        if sample_type == "2ピーク（シンプル）":
            y = (50 * np.exp(-((x-30)**2)/(2*5**2)) + 
                 80 * np.exp(-((x-60)**2)/(2*8**2)) + 
                 np.random.normal(0, 2, len(x)))
        elif sample_type == "3ピーク（複雑）":
            y = (40 * np.exp(-((x-25)**2)/(2*4**2)) + 
                 70 * np.exp(-((x-50)**2)/(2*6**2)) + 
                 50 * np.exp(-((x-75)**2)/(2*5**2)) + 
                 np.random.normal(0, 2, len(x)))
        else:  # 背景あり
            y = (50 * np.exp(-((x-40)**2)/(2*6**2)) + 
                 60 * np.exp(-((x-70)**2)/(2*7**2)) + 
                 0.3 * x + 5 +  # Linear background
                 np.random.normal(0, 2, len(x)))
        
        st.session_state.data = pd.DataFrame({'x': x, 'y': y})
        
    else:
        uploaded_file = st.file_uploader(
            "ファイルを選択",
            type=['csv', 'txt', 'xlsx'],
            help="CSVファイル（x, y列）をアップロードしてください"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                # Try to find x and y columns
                if len(df.columns) >= 2:
                    st.session_state.data = pd.DataFrame({
                        'x': df.iloc[:, 0].values,
                        'y': df.iloc[:, 1].values
                    })
                    st.success(f"✅ {len(df)} データポイントを読み込みました")
                else:
                    st.error("❌ 少なくとも2列必要です")
            except Exception as e:
                st.error(f"❌ ファイル読み込みエラー: {e}")
    
    st.divider()
    
    # Fitting parameters
    st.subheader("🔧 フィッティング設定")
    
    K = st.slider(
        "ピーク数 (K)",
        min_value=1,
        max_value=10,
        value=2,
        help="フィッティングするガウスピークの数"
    )
    
    background = st.selectbox(
        "背景モデル",
        ['none', 'uniform', 'linear', 'ramp_sum'],
        help="スペクトルのバックグラウンドモデル"
    )
    
    method = st.selectbox(
        "フィッティング手法",
        ['adapted_em'],
        help="adapted_em: EMアルゴリズムによるフィッティング"
    )
    
    with st.expander("🔬 詳細設定"):
        trial = st.slider("試行回数 (trial)", 1, 30, 10)
        max_iter = st.slider("最大反復回数", 100, 5000, 1000)
    
    st.divider()
    
    # Run button
    run_button = st.button(
        "▶️ フィッティング実行",
        type="primary",
        use_container_width=True,
        disabled=(st.session_state.data is None)
    )

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📈 グラフ")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot original data
        ax.scatter(data['x'], data['y'], alpha=0.5, s=10, label='データ', color='#1E88E5')
        
        # Plot fitted result if available
        if st.session_state.fitted and st.session_state.result is not None:
            result = st.session_state.result
            x_fit = np.linspace(data['x'].min(), data['x'].max(), 500)
            
            # Plot each peak
            colors = plt.cm.Set2(np.linspace(0, 1, len(result['mu'])))
            for i, (mu, sigma, pi) in enumerate(zip(result['mu'], result['sigma'], result['pi'][:len(result['mu'])])):
                y_peak = pi * result['N_tot'] * np.exp(-((x_fit - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
                ax.plot(x_fit, y_peak, '--', color=colors[i], linewidth=2, 
                       label=f'Peak {i+1}: μ={mu:.1f}, σ={sigma:.1f}')
            
            # Plot total fit
            if 'y_fit' in result:
                ax.plot(x_fit, result['y_fit'], 'r-', linewidth=2.5, label='フィット結果')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
        
        # Data preview
        with st.expander("📋 データプレビュー"):
            st.dataframe(data.head(20), use_container_width=True)
    else:
        st.info("👈 左のサイドバーからデータを選択または読み込んでください")

with col2:
    st.subheader("📋 フィッティング結果")
    
    if run_button and st.session_state.data is not None:
        with st.spinner('フィッティング中...'):
            data = st.session_state.data
            x = data['x'].values
            y = data['y'].values
            
            if EMPEAKS_AVAILABLE:
                try:
                    # Create model and fit
                    model = GaussianMixtureModel(
                        K=K,
                        x_min=x.min(),
                        x_max=x.max(),
                        background=background
                    )
                    
                    result_info = model.fit(x, y, method=method, trial=trial, max_iter=max_iter)
                    params = model.export_param()
                    
                    # Store result
                    st.session_state.result = {
                        'mu': params['mu'],
                        'sigma': params['sigma'],
                        'pi': params['pi'],
                        'N_tot': model.N_tot,
                        'RMSE': result_info['RMSE'] if isinstance(result_info, dict) else result_info[1]['RMSE'],
                        'y_fit': model.predict(np.linspace(x.min(), x.max(), 500)) * model.N_tot
                    }
                    st.session_state.model = model
                    st.session_state.fitted = True
                    st.success("✅ フィッティング完了!")
                    st.rerun()  # Rerun to update the graph
                    
                except Exception as e:
                    st.error(f"❌ エラー: {e}")
                    import traceback
                    with st.expander("エラー詳細"):
                        st.code(traceback.format_exc())
                    st.session_state.fitted = False
            else:
                # Demo mode - simulate fitting
                st.info("🔄 デモモード: シミュレーション結果を表示")
                
                # Simple peak detection simulation
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(y, height=np.mean(y))
                
                # Ensure we have enough peaks or use defaults
                if len(peaks) < K:
                    # Generate evenly spaced peak positions
                    peak_positions = np.linspace(x.min() + (x.max()-x.min())*0.1, 
                                                  x.max() - (x.max()-x.min())*0.1, K)
                    mu_list = list(peak_positions)
                else:
                    mu_list = [float(x[p]) for p in peaks[:K]]
                
                sigma_list = [5.0 + np.random.rand()*3 for _ in range(K)]
                pi_list = [1.0/K for _ in range(K)]
                
                st.session_state.result = {
                    'mu': mu_list,
                    'sigma': sigma_list,
                    'pi': pi_list,
                    'N_tot': float(np.sum(y)),
                    'RMSE': float(np.random.rand() * 5)
                }
                st.session_state.fitted = True
                st.success("✅ フィッティング完了 (デモ)")
                st.rerun()
    
    # Display results
    if st.session_state.fitted and st.session_state.result is not None:
        result = st.session_state.result
        
        # Create results dataframe
        result_df = pd.DataFrame({
            'Peak': [f'Peak {i+1}' for i in range(len(result['mu']))],
            'μ (位置)': [f"{mu:.2f}" for mu in result['mu']],
            'σ (幅)': [f"{sigma:.2f}" for sigma in result['sigma']],
            '比率': [f"{pi:.3f}" for pi in result['pi'][:len(result['mu'])]]
        })
        
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        # Statistics
        st.markdown("##### 📊 統計情報")
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("RMSE", f"{result['RMSE']:.4f}")
        with stat_col2:
            st.metric("N_tot", f"{result['N_tot']:.2e}")
        
        st.divider()
        
        # Export section
        st.markdown("##### 💾 結果のエクスポート")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # CSV export
            csv_data = result_df.to_csv(index=False)
            st.download_button(
                label="📄 CSV",
                data=csv_data,
                file_name="empeaks_result.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col2:
            # JSON export
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            def to_serializable(val):
                if isinstance(val, np.ndarray):
                    return val.tolist()
                if isinstance(val, (np.float32, np.float64)):
                    return float(val)
                if isinstance(val, (np.int32, np.int64)):
                    return int(val)
                if isinstance(val, list):
                    return [to_serializable(v) for v in val]
                return val
            
            json_data = json.dumps({
                'mu': to_serializable(result['mu']),
                'sigma': to_serializable(result['sigma']),
                'pi': to_serializable(result['pi'][:len(result['mu'])]),
                'N_tot': to_serializable(result['N_tot']),
                'RMSE': to_serializable(result['RMSE'])
            }, indent=2)
            st.download_button(
                label="📋 JSON",
                data=json_data,
                file_name="empeaks_result.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("フィッティングを実行すると結果がここに表示されます")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
    "EMPeaks GUI v0.1.0 | Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)
