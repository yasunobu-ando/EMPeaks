"""Model configuration module"""
import streamlit as st
from module.utils import init_model, refresh
from module.i18n import t

MODEL_LIST = ['GaussianMixture', 'LorentzianMixture', 'PseudoVoigtMixture', 'DoniachSunjicMixture', 'TSDCMixture']
BACKGROUND_LIST = ['none', 'uniform', 'linear', 'squareroot', 'ramp_sum']
FITTING_METHOD_STANDARD = ['sampling']
FITTING_METHOD_TSDC = ['sampling', 'leastsq', 'leastsq_tau0', 'l2_div']


def model_constructor(db, x_variable):
    """Build the model configuration panel in the sidebar"""
    with st.container():
        st.sidebar.header(t("empeaks_parameters"))

        # Select Mixture Model
        st.session_state['MixtureModel'] = st.sidebar.selectbox(
            t("select_mixture_model"),
            MODEL_LIST,
            index=0,
            on_change=refresh
        )

        is_tsdc = st.session_state['MixtureModel'] == 'TSDCMixture'

        # K (Component Number)
        st.session_state['K'] = int(st.sidebar.number_input(
            t("k_component_number"),
            min_value=1,
            max_value=10,
            value=st.session_state.get('K', 3),
            on_change=refresh
        ))

        # Select Background Model
        st.session_state['BackgroundModel'] = st.sidebar.selectbox(
            t("select_background_model"),
            BACKGROUND_LIST,
            index=0,
            on_change=refresh
        )

        # Fitting method selection
        method_list = FITTING_METHOD_TSDC if is_tsdc else FITTING_METHOD_STANDARD
        st.session_state['FittingMethod'] = st.sidebar.selectbox(
            t("fitting_method"),
            method_list,
            index=0,
            on_change=refresh
        )

        # TSDC-specific parameters
        if is_tsdc:
            st.sidebar.subheader(t("tsdc_parameters"))
            st.session_state['TSDC_beta'] = st.sidebar.number_input(
                t("tsdc_beta"),
                value=st.session_state.get('TSDC_beta', 0.0833),
                format="%.4f",
                on_change=refresh
            )
            tsdc_col = st.sidebar.columns(2)
            st.session_state['TSDC_T_min'] = tsdc_col[0].number_input(
                t("tsdc_t_min"),
                value=st.session_state.get('TSDC_T_min', 300),
                format="%d",
                on_change=refresh
            )
            st.session_state['TSDC_T_max'] = tsdc_col[1].number_input(
                t("tsdc_t_max"),
                value=st.session_state.get('TSDC_T_max', 900),
                format="%d",
                on_change=refresh
            )
            ea_col = st.sidebar.columns(2)
            st.session_state['TSDC_Ea_min'] = ea_col[0].number_input(
                t("tsdc_ea_min"),
                value=st.session_state.get('TSDC_Ea_min', 0.05),
                format="%.3f",
                on_change=refresh
            )
            st.session_state['TSDC_Ea_max'] = ea_col[1].number_input(
                t("tsdc_ea_max"),
                value=st.session_state.get('TSDC_Ea_max', 3.0),
                format="%.3f",
                on_change=refresh
            )

        # Input Trial Frequency
        st.session_state['TrialFrequency'] = int(st.sidebar.number_input(
            t("input_trial_frequency"),
            min_value=1,
            max_value=30,
            value=st.session_state.get('TrialFrequency', 1),
            format="%d",
            on_change=refresh
        ))

        # Convergence threshold for LL
        st.session_state['Threshold'] = st.sidebar.number_input(
            t("convergence_threshold"),
            value=st.session_state.get('Threshold', 1e-8),
            step=5e-8,
            format="%f",
            on_change=refresh
        )

        # Max iteration
        st.session_state['MaxIteration'] = int(st.sidebar.number_input(
            t("max_iteration"),
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
        return {'mu': t('peak_position'), 'sigma': t('standard_deviation'), 'pi': t('mixing_ratio')}
    elif model_type == 'LorentzianMixture':
        return {'x0': t('peak_position'), 'gamma': t('hwhm'), 'pi': t('mixing_ratio')}
    elif model_type == 'PseudoVoigtMixture':
        return {'mu': t('peak_position'), 'sigma': t('standard_deviation'), 'eta': t('mixing_parameter'),
                'pi': t('mixing_ratio')}
    elif model_type == 'DoniachSunjicMixture':
        return {'mu': t('peak_position'), 'sigma': t('standard_deviation'), 'alpha': t('asymmetry_parameter'),
                'pi': t('mixing_ratio')}
    elif model_type == 'TSDCMixture':
        return {'Ea': t('tsdc_ea'), 'Tp': t('tsdc_tp'), 'tau0': t('tsdc_tau0'), 'pi': t('mixing_ratio')}
    return {'mu': t('peak_position'), 'sigma': t('standard_deviation'), 'pi': t('mixing_ratio')}
