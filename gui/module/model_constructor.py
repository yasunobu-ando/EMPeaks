"""Model configuration module"""
import streamlit as st
from module.utils import init_model, refresh

MODEL_LIST = ['GaussianMixture', 'LorentzianMixture', 'PseudoVoigtMixture', 'DoniachSunjicMixture']
BACKGROUND_LIST = ['none', 'uniform', 'linear', 'squareroot', 'ramp_sum']


def model_constructor(db, x_variable):
    """Build the model configuration panel in the sidebar"""
    with st.container():
        st.sidebar.header('EMPeaks Parameters')

        # Select Mixture Model
        st.session_state['MixtureModel'] = st.sidebar.selectbox(
            'Select Mixture Model',
            MODEL_LIST,
            index=0,
            on_change=refresh
        )

        # K (Component Number)
        st.session_state['K'] = int(st.sidebar.number_input(
            'K (Component Number)',
            min_value=1,
            max_value=10,
            value=st.session_state.get('K', 3),
            on_change=refresh
        ))

        # Select Background Model
        st.session_state['BackgroundModel'] = st.sidebar.selectbox(
            'Select Background Model',
            BACKGROUND_LIST,
            index=0,
            on_change=refresh
        )

        # Input Trial Frequency
        st.session_state['TrialFrequency'] = int(st.sidebar.number_input(
            'Input Trial Frequency',
            min_value=1,
            max_value=30,
            value=st.session_state.get('TrialFrequency', 1),
            format="%d",
            on_change=refresh
        ))

        # Convergence threshold for LL
        st.session_state['Threshold'] = st.sidebar.number_input(
            'Convergence threshold for LL',
            value=st.session_state.get('Threshold', 1e-8),
            step=5e-8,
            format="%f",
            on_change=refresh
        )

        # Max iteration
        st.session_state['MaxIteration'] = int(st.sidebar.number_input(
            'Convergence threshold for LL',  # Original Deck label (kept as-is)
            value=st.session_state.get('MaxIteration', 1000),
            format="%d",
            on_change=refresh
        ))

    # Initialize model instance
    x_min = db[x_variable].min()
    x_max = db[x_variable].max()
    st.session_state['ModelInstance'] = init_model(st.session_state['K'], x_min, x_max)

    # Model parameter display info
    model_param = _get_model_param_info()
    return model_param


def _get_model_param_info():
    """Get model parameter names and descriptions based on model type"""
    model_type = st.session_state['MixtureModel']

    if model_type == 'GaussianMixture':
        return {'mu': 'peak position', 'sigma': 'standard deviation', 'pi': 'mixing ratio'}
    elif model_type == 'LorentzianMixture':
        return {'x0': 'peak position', 'gamma': 'Half Width Half Maximum', 'pi': 'mixing ratio'}
    elif model_type == 'PseudoVoigtMixture':
        return {'mu': 'peak position', 'sigma': 'standard deviation', 'eta': 'mixing parameter',
                'pi': 'mixing ratio'}
    elif model_type == 'DoniachSunjicMixture':
        return {'mu': 'peak position', 'sigma': 'standard deviation', 'alpha': 'asymmetry parameter',
                'pi': 'mixing ratio'}
    return {'mu': 'peak position', 'sigma': 'standard deviation', 'pi': 'mixing ratio'}
