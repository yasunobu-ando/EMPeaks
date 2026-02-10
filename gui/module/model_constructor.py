"""モデル設定モジュール"""
import streamlit as st
from module.utils import init_model, refresh

MODEL_LIST = ['GaussianMixture', 'LorentzianMixture', 'PseudoVoigtMixture', 'DoniachSunjicMixture']
BACKGROUND_LIST = ['none', 'uniform', 'linear', 'squareroot', 'ramp_sum']


def model_constructor(db, x_variable):
    """モデル設定パネルを構築"""
    with st.container():
        st.sidebar.header('⚙️ EMPeaks パラメータ')
        
        # モデル選択
        st.session_state['MixtureModel'] = st.sidebar.selectbox(
            'モデルタイプ',
            MODEL_LIST,
            index=0,
            on_change=refresh
        )
        
        # コンポーネント数
        st.session_state['K'] = int(st.sidebar.number_input(
            'K (ピーク数)',
            min_value=1,
            max_value=10,
            value=st.session_state.get('K', 3),
            on_change=refresh
        ))
        
        # 背景モデル
        st.session_state['BackgroundModel'] = st.sidebar.selectbox(
            '背景モデル',
            BACKGROUND_LIST,
            index=0,
            on_change=refresh
        )
        
        # 試行回数
        st.session_state['TrialFrequency'] = int(st.sidebar.number_input(
            '試行回数',
            min_value=1,
            max_value=30,
            value=st.session_state.get('TrialFrequency', 1),
            on_change=refresh
        ))
        
        # 収束条件
        st.session_state['Threshold'] = st.sidebar.number_input(
            '収束閾値',
            value=st.session_state.get('Threshold', 1e-8),
            format="%.2e",
            on_change=refresh
        )
        
        # 最大反復回数
        st.session_state['MaxIteration'] = int(st.sidebar.number_input(
            '最大反復回数',
            min_value=100,
            max_value=5000,
            value=st.session_state.get('MaxIteration', 1000),
            on_change=refresh
        ))
    
    # モデルインスタンスの初期化
    x_min = db[x_variable].min()
    x_max = db[x_variable].max()
    st.session_state['ModelInstance'] = init_model(st.session_state['K'], x_min, x_max)
    
    # モデルパラメータ情報を返す
    model_param = {
        'mu': 'ピーク位置',
        'sigma': 'ピーク幅',
        'pi': '混合比率'
    }
    
    return model_param
