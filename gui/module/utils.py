"""ユーティリティ関数"""
import streamlit as st
import numpy as np

# EMPeaks imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from EMPeaks.GaussianMixture import GaussianMixtureModel
    from EMPeaks.LorentzianMixture import LorentzianMixtureModel
    from EMPeaks.PseudoVoigtMixture import PseudoVoigtMixtureModel
    from EMPeaks.DoniachSunjicMixture import DoniachSunjicMixtureModel
    from EMPeaks.VoigtMixture import VoigtMixtureModel
    from EMPeaks.TSDCMixture import TSDCMixtureModel
    EMPEAKS_AVAILABLE = True
except ImportError:
    EMPEAKS_AVAILABLE = False
    GaussianMixtureModel = None
    LorentzianMixtureModel = None
    PseudoVoigtMixtureModel = None
    DoniachSunjicMixtureModel = None
    VoigtMixtureModel = None
    TSDCMixtureModel = None


def init_application():
    """アプリケーションの初期化"""
    if ('init' not in st.session_state) or (st.session_state['init'] is False):
        st.session_state['K'] = 3
        st.session_state['MixtureModel'] = 'GaussianMixture'
        st.session_state['BackgroundModel'] = 'none'
        st.session_state['TrialFrequency'] = 1
        st.session_state['ModelInstance'] = None
        st.session_state['Threshold'] = 1e-9
        st.session_state['MaxIteration'] = 1000
        st.session_state['Fitted'] = False
        st.session_state['Data'] = None
        st.session_state['FittingMethod'] = 'sampling'
        st.session_state['TriggerOptimization'] = False
        st.session_state['FitParam'] = None
        # TSDC-specific parameters
        st.session_state['TSDC_beta'] = 0.0833
        st.session_state['TSDC_T_min'] = 300
        st.session_state['TSDC_T_max'] = 900
        st.session_state['TSDC_Ea_min'] = 0.05
        st.session_state['TSDC_Ea_max'] = 3.0
        # B-Spline specific parameters
        st.session_state['degree_spline'] = 3
        st.session_state['n_section'] = 5
        st.session_state['k_ramp'] = 5
        st.session_state['init'] = True


def init_model(K, x_min, x_max):
    """モデルインスタンスを初期化"""
    if not EMPEAKS_AVAILABLE:
        return None

    model_type = st.session_state['MixtureModel']
    background = st.session_state['BackgroundModel']

    kwargs = {}
    if background == 'b_spline':
        kwargs['degree_spline'] = st.session_state.get('degree_spline', 3)
        kwargs['n_section'] = st.session_state.get('n_section', 5)
    elif background == 'ramp_sum':
        kwargs['k_ramp'] = st.session_state.get('k_ramp', 5)

    if model_type == 'GaussianMixture':
        return GaussianMixtureModel(K=K, x_min=x_min, x_max=x_max, background=background, **kwargs)
    elif model_type == 'LorentzianMixture':
        return LorentzianMixtureModel(K=K, x_min=x_min, x_max=x_max, background=background, **kwargs)
    elif model_type == 'PseudoVoigtMixture':
        return PseudoVoigtMixtureModel(K=K, x_min=x_min, x_max=x_max, background=background, **kwargs)
    elif model_type == 'DoniachSunjicMixture':
        return DoniachSunjicMixtureModel(K=K, x_min=x_min, x_max=x_max, background=background, **kwargs)
    elif model_type == 'VoigtMixture':
        return VoigtMixtureModel(K=K, x_min=x_min, x_max=x_max, background=background, **kwargs)
    elif model_type == 'TSDCMixture':
        return TSDCMixtureModel(
            K=K,
            beta=st.session_state['TSDC_beta'],
            T_min=st.session_state['TSDC_T_min'],
            T_max=st.session_state['TSDC_T_max'],
            Ea_min=st.session_state['TSDC_Ea_min'],
            Ea_max=st.session_state['TSDC_Ea_max'],
            background=background,
            **kwargs
        )
    return None


def run(model_instance, x_val, y_val, trial):
    """フィッティングを実行"""
    fitting_method = st.session_state.get('FittingMethod', 'sampling')
    x_arr = np.array(x_val, dtype=float)
    y_arr = np.array(y_val, dtype=float)

    if fitting_method == 'sampling':
        out_info = model_instance.sampling(
            x=x_arr,
            intensity=y_arr,
            trial=trial,
            max_iter=st.session_state['MaxIteration'],
            r_eps=st.session_state['Threshold']
        )
    elif fitting_method == 'leastsq':
        run_info = model_instance.leastsq(x_arr, y_arr, stdout=False)
        out_info = _wrap_single_run_info(run_info)
    elif fitting_method == 'leastsq_tau0':
        run_info = model_instance.leastsq_tau0(x_arr, y_arr, stdout=False)
        out_info = _wrap_single_run_info(run_info)
    elif fitting_method == 'l2_div':
        run_info = model_instance.l2_div(x_arr, y_arr, stdout=False)
        out_info = _wrap_single_run_info(run_info)
    else:
        out_info = model_instance.sampling(
            x=x_arr,
            intensity=y_arr,
            trial=trial,
            max_iter=st.session_state['MaxIteration'],
            r_eps=st.session_state['Threshold']
        )

    return model_instance, out_info


def _wrap_single_run_info(run_info):
    """単一実行の結果を sampling 互換の info 形式に変換"""
    return {
        'index_best': 0,
        'LL_hist': np.array([run_info.get('LL', np.nan)]),
        'RMSE_hist': np.array([run_info.get('RMSE', np.nan)]),
        'time_hist': np.array([run_info.get('total_time', np.nan)]),
        'iter_hist': np.array([run_info.get('total_iter', np.nan)]),
    }


def refresh():
    """状態をリフレッシュ"""
    st.session_state['Fitted'] = False
    st.session_state['FitInfo'] = None
    st.session_state['ModelInstance'] = None
    st.cache_data.clear()
