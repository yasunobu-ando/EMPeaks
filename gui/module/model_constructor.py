"""Model configuration module"""
import streamlit as st
from module.utils import init_model, refresh
from module.i18n import t

MODEL_LIST = ['GaussianMixture', 'LorentzianMixture', 'PseudoVoigtMixture', 'DoniachSunjicMixture', 'VoigtMixture', 'TSDCMixture']
BACKGROUND_LIST = ['none', 'uniform', 'linear', 'squareroot', 'ramp_sum', 'b_spline']
FITTING_METHOD_STANDARD = ['sampling', 'deterministic_annealing']
FITTING_METHOD_TSDC = ['sampling', 'leastsq', 'leastsq_tau0', 'l2_div']

_THRESHOLD_SEQUENCE = [
    1e-5, 5e-6,
    1e-6, 5e-7,
    1e-7, 5e-8,
    1e-8, 5e-9,
    1e-9, 5e-10,
    1e-10, 5e-11,
    1e-11, 5e-12,
    1e-12,
]


def _threshold_label(x: float) -> str:
    exponent = int(f"{x:.0e}".split("e")[1])
    mantissa = x / (10 ** exponent)
    if abs(mantissa - 1) < 1e-12:
        return f"1e{exponent}"
    elif abs(mantissa - 5) < 1e-12:
        return f"5e{exponent}"
    return f"{x:.0e}"


def model_constructor(db, x_variable):
    """Build the model configuration panel in the sidebar"""
    with st.container():
        st.sidebar.header(t("empeaks_parameters"))

        # Select Mixture Model
        st.session_state['MixtureModel'] = st.sidebar.selectbox(
            t("select_mixture_model"),
            MODEL_LIST,
            index=0,
        )

        is_tsdc = st.session_state['MixtureModel'] == 'TSDCMixture'

        # K (Component Number)
        st.sidebar.number_input(
            t("k_component_number"),
            min_value=1,
            max_value=10,
            key='K',
        )

        # Select Background Model
        st.session_state['BackgroundModel'] = st.sidebar.selectbox(
            t("select_background_model"),
            BACKGROUND_LIST,
            index=0,
        )

        if st.session_state['BackgroundModel'] == 'b_spline':
            st.sidebar.subheader(t("bspline_parameters"))
            st.sidebar.number_input(
                t("degree_spline"),
                min_value=0, max_value=5, step=1,
                key='degree_spline',
            )
            st.sidebar.number_input(
                t("n_section"),
                min_value=1, max_value=100, step=1,
                key='n_section',
            )
        elif st.session_state['BackgroundModel'] == 'ramp_sum':
            st.sidebar.subheader(t("rampsum_parameters"))
            st.sidebar.number_input(
                t("k_ramp"),
                min_value=1, max_value=50, step=1,
                key='k_ramp',
            )

        # Fitting method selection
        method_list = FITTING_METHOD_TSDC if is_tsdc else FITTING_METHOD_STANDARD
        st.session_state['FittingMethod'] = st.sidebar.selectbox(
            t("fitting_method"),
            method_list,
            index=0,
        )

        # DA-specific parameters
        if st.session_state['FittingMethod'] == 'deterministic_annealing':
            st.sidebar.subheader(t("da_parameters"))
            alpha = st.sidebar.number_input(
                t("dirichlet_alpha"),
                min_value=0.01,
                step=0.1,
                format="%.2f",
                key='DirichletAlpha',
            )
            if alpha >= 1.0:
                st.sidebar.warning(t("da_alpha_warning"))

        # TSDC-specific parameters
        if is_tsdc:
            st.sidebar.subheader(t("tsdc_parameters"))
            st.sidebar.number_input(
                t("tsdc_beta"),
                format="%.4f",
                key='TSDC_beta'
            )
            tsdc_col = st.sidebar.columns(2)
            tsdc_col[0].number_input(
                t("tsdc_t_min"),
                format="%d",
                step=1,
                key='TSDC_T_min'
            )
            tsdc_col[1].number_input(
                t("tsdc_t_max"),
                format="%d",
                step=1,
                key='TSDC_T_max'
            )
            ea_col = st.sidebar.columns(2)
            ea_col[0].number_input(
                t("tsdc_ea_min"),
                format="%.3f",
                key='TSDC_Ea_min'
            )
            ea_col[1].number_input(
                t("tsdc_ea_max"),
                format="%.3f",
                key='TSDC_Ea_max'
            )

        # Input Trial Frequency
        st.sidebar.number_input(
            t("input_trial_frequency"),
            min_value=1,
            max_value=30,
            format="%d",
            step=1,
            key='TrialFrequency',
        )

        # Convergence threshold for LL
        st.sidebar.select_slider(
            t("convergence_threshold"),
            options=_THRESHOLD_SEQUENCE,
            format_func=_threshold_label,
            key='Threshold',
        )

        # Max iteration
        st.sidebar.number_input(
            t("max_iteration"),
            min_value=100,
            step=100,
            format="%d",
            key='MaxIteration',
        )

        # Buttons
        st.sidebar.divider()
        if st.sidebar.button(t("start_optimization"), use_container_width=True):
            st.session_state['TriggerOptimization'] = True
        if st.sidebar.button(t("refresh"), use_container_width=True):
            refresh()
            st.rerun()

    # ModelInstance が None のときだけ初期化（結果は Start Optimization まで保持）
    if st.session_state.get('ModelInstance') is None:
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
        return {'x0': t('peak_position'), 'gamma': t('hwhm'), 'eta': t('mixing_parameter'),
                'pi': t('mixing_ratio')}
    elif model_type == 'DoniachSunjicMixture':
        return {'x0': t('peak_position'), 'gamma': t('hwhm'), 'alpha': t('asymmetry_parameter'),
                'pi': t('mixing_ratio')}
    elif model_type == 'VoigtMixture':
        return {'x0': t('peak_position'), 'sigma': t('standard_deviation'), 'gamma': t('hwhm'),
                'pi': t('mixing_ratio')}
    elif model_type == 'TSDCMixture':
        return {'Ea': t('tsdc_ea'), 'Tp': t('tsdc_tp'), 'tau0': t('tsdc_tau0'), 'pi': t('mixing_ratio')}
    return {'mu': t('peak_position'), 'sigma': t('standard_deviation'), 'pi': t('mixing_ratio')}
