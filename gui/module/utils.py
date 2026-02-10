"""ユーティリティ関数"""
import streamlit as st

# EMPeaks imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from EMPeaks.GaussianMixture import GaussianMixtureModel
    from EMPeaks.LorentzianMixture import LorentzianMixtureModel
    from EMPeaks.PseudoVoigtMixture import PseudoVoigtMixtureModel
    from EMPeaks.DoniachSunjicMixture import DoniachSunjicMixtureModel
    EMPEAKS_AVAILABLE = True
except ImportError:
    EMPEAKS_AVAILABLE = False
    GaussianMixtureModel = None
    LorentzianMixtureModel = None
    PseudoVoigtMixtureModel = None
    DoniachSunjicMixtureModel = None


def init_application():
    """アプリケーションの初期化"""
    if ('init' not in st.session_state) or (st.session_state['init'] is False):
        st.session_state['K'] = 3
        st.session_state['MixtureModel'] = 'GaussianMixture'
        st.session_state['BackgroundModel'] = 'none'
        st.session_state['TrialFrequency'] = 1
        st.session_state['ModelInstance'] = None
        st.session_state['Threshold'] = 1e-8
        st.session_state['MaxIteration'] = 1000
        st.session_state['Fitted'] = False
        st.session_state['Data'] = None
        st.session_state['init'] = True


def init_model(K, x_min, x_max):
    """モデルインスタンスを初期化"""
    if not EMPEAKS_AVAILABLE:
        return None
    
    model_type = st.session_state['MixtureModel']
    background = st.session_state['BackgroundModel']
    
    if model_type == 'GaussianMixture':
        return GaussianMixtureModel(K=K, x_min=x_min, x_max=x_max, background=background)
    elif model_type == 'LorentzianMixture':
        return LorentzianMixtureModel(K=K, x_min=x_min, x_max=x_max, background=background)
    elif model_type == 'PseudoVoigtMixture':
        return PseudoVoigtMixtureModel(K=K, x_min=x_min, x_max=x_max, background=background)
    elif model_type == 'DoniachSunjicMixture':
        return DoniachSunjicMixtureModel(K=K, x_min=x_min, x_max=x_max, background=background)
    return None


def run(model_instance, x_val, y_val, trial):
    """フィッティングを実行"""
    out_info = model_instance.sampling(
        x=x_val, 
        intensity=y_val, 
        trial=trial,
        max_iter=st.session_state['MaxIteration'],
        r_eps=st.session_state['Threshold']
    )
    return model_instance, out_info


def refresh():
    """状態をリフレッシュ"""
    st.session_state['Fitted'] = False
    st.cache_data.clear()
