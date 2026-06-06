use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadwriteArray1, IntoPyArray as _};
use ndarray::Array1;

mod gaussian;
mod em_engine;
mod pseudo_voigt;
mod lorentzian;
mod doniach_sunjic;
mod tsdc;


/// run_em_loop(x, intensity, mu, sigma, pi, dirichlet_alpha, max_iter, r_eps)
///
/// Runs the Gaussian EM loop entirely in Rust.
/// mu, sigma, pi are modified in-place.
/// Returns (total_iter, ll_hist, residual_hist).
#[pyfunction]
fn run_em_loop<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    intensity: PyReadonlyArray1<'py, f64>,
    mut mu: PyReadwriteArray1<'py, f64>,
    mut sigma: PyReadwriteArray1<'py, f64>,
    mut pi: PyReadwriteArray1<'py, f64>,
    dirichlet_alpha: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    r_eps: f64,
) -> PyResult<(usize, PyObject, PyObject)> {
    let x_arr = x.as_array();
    let int_arr = intensity.as_array();
    let da = dirichlet_alpha.as_array();

    let mut mu_vec: Vec<f64>    = mu.as_array().to_vec();
    let mut sigma_vec: Vec<f64> = sigma.as_array().to_vec();
    let mut pi_vec: Vec<f64>    = pi.as_array().to_vec();
    let da_slice: Vec<f64>      = da.to_vec();

    let (total_iter, ll_hist, res_hist) = em_engine::run_em_loop(
        x_arr, int_arr,
        &mut mu_vec, &mut sigma_vec, &mut pi_vec,
        &da_slice, max_iter, r_eps,
    );

    // Write results back into the Python-owned arrays
    mu.as_array_mut().iter_mut()
        .zip(mu_vec.iter())
        .for_each(|(dst, &src)| *dst = src);
    sigma.as_array_mut().iter_mut()
        .zip(sigma_vec.iter())
        .for_each(|(dst, &src)| *dst = src);
    pi.as_array_mut().iter_mut()
        .zip(pi_vec.iter())
        .for_each(|(dst, &src)| *dst = src);

    let ll_py  = Array1::from_vec(ll_hist).into_pyarray_bound(py).into();
    let res_py = Array1::from_vec(res_hist).into_pyarray_bound(py).into();

    Ok((total_iter, ll_py, res_py))
}

/// pseudo_voigt_predict(x, x0, gamma, eta, x_min, x_max) -> ndarray
///
/// Compute PseudoVoigt mixture PDF at each point in x.
#[pyfunction]
fn pseudo_voigt_predict<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    x0: f64, gamma: f64, eta: f64,
    x_min: f64, x_max: f64,
) -> PyResult<PyObject> {
    let x_slice: Vec<f64> = x.as_array().to_vec();
    let result = pseudo_voigt::predict(&x_slice, x0, gamma, eta, x_min, x_max);
    Ok(Array1::from_vec(result).into_pyarray_bound(py).into())
}

/// pseudo_voigt_mle_conditional_max(x, intensity, x0, gamma, eta, x_min, x_max)
///     -> (x0, gamma, eta)
///
/// Single-pass conditional-maximization MLE for a PseudoVoigt component.
/// Matches Python's PseudoVoigt.conditional_max.
#[pyfunction]
fn pseudo_voigt_mle_conditional_max(
    x: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    x0: f64, gamma: f64, eta: f64,
    x_min: f64, x_max: f64,
) -> PyResult<(f64, f64, f64)> {
    let x_slice: Vec<f64> = x.as_array().to_vec();
    let int_slice: Vec<f64> = intensity.as_array().to_vec();
    let (mut x0, mut gamma, mut eta) = (x0, gamma, eta);
    pseudo_voigt::mle_conditional_max(
        &x_slice, &int_slice, &mut x0, &mut gamma, &mut eta, x_min, x_max,
    );
    Ok((x0, gamma, eta))
}

/// pseudo_voigt_mle_full_optimization(
///     x, intensity, x0, gamma, eta,
///     x_min, x_max, gamma_min, gamma_max, max_iter, r_eps
/// ) -> (x0, gamma, eta)
///
/// L-BFGS-B + eta grid-search MLE for a PseudoVoigt component.
/// Matches Python's PseudoVoigt.full_optimization.
#[pyfunction]
fn pseudo_voigt_mle_full_optimization(
    x: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    x0: f64, gamma: f64, eta: f64,
    x_min: f64, x_max: f64,
    gamma_min: f64, gamma_max: f64,
    max_iter: usize, r_eps: f64,
) -> PyResult<(f64, f64, f64)> {
    let x_slice: Vec<f64> = x.as_array().to_vec();
    let int_slice: Vec<f64> = intensity.as_array().to_vec();
    let (mut x0, mut gamma, mut eta) = (x0, gamma, eta);
    pseudo_voigt::mle_full_optimization(
        &x_slice, &int_slice,
        &mut x0, &mut gamma, &mut eta,
        x_min, x_max, gamma_min, gamma_max,
        max_iter, r_eps,
    );
    Ok((x0, gamma, eta))
}

/// lorentzian_mle(x, intensity, x0, gamma, x_min, x_max) -> (x0, gamma)
///
/// L-BFGS-B MLE for a single Lorentzian component.
/// Matches Python's Lorentzian.minimize_bfgs (resets x0/gamma to empirical values internally).
#[pyfunction]
fn lorentzian_mle(
    x: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    x0: f64,
    gamma: f64,
    x_min: f64,
    x_max: f64,
) -> PyResult<(f64, f64)> {
    let x_slice: Vec<f64> = x.as_array().to_vec();
    let int_slice: Vec<f64> = intensity.as_array().to_vec();
    let (mut x0, mut gamma) = (x0, gamma);
    lorentzian::mle_lorentzian(&x_slice, &int_slice, &mut x0, &mut gamma, x_min, x_max);
    Ok((x0, gamma))
}

/// doniach_sunjic_mle(
///     x, intensity, x0, gamma, alpha,
///     x_min, x_max, gamma_min, gamma_max, alpha_min, alpha_max
/// ) -> (x0, gamma, alpha)
///
/// L-BFGS-B MLE for a single DoniachSunjic component.
/// Matches Python's DoniachSunjic.full_optimization.
#[pyfunction]
fn doniach_sunjic_mle(
    x: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    x0: f64,
    gamma: f64,
    alpha: f64,
    x_min: f64,
    x_max: f64,
    gamma_min: f64,
    gamma_max: f64,
    alpha_min: f64,
    alpha_max: f64,
) -> PyResult<(f64, f64, f64)> {
    let x_slice: Vec<f64> = x.as_array().to_vec();
    let int_slice: Vec<f64> = intensity.as_array().to_vec();
    let (mut x0, mut gamma, mut alpha) = (x0, gamma, alpha);
    doniach_sunjic::mle_doniach_sunjic(
        &x_slice, &int_slice,
        &mut x0, &mut gamma, &mut alpha,
        x_min, x_max, gamma_min, gamma_max, alpha_min, alpha_max,
    );
    Ok((x0, gamma, alpha))
}

/// tsdc_predict(t, ea, tau0, beta) -> ndarray
#[pyfunction]
fn tsdc_predict<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    ea: f64, tau0: f64, beta: f64,
) -> PyResult<PyObject> {
    let t_slice = t.as_array().to_vec();
    let result = tsdc::predict_tsdc(&t_slice, ea, tau0, beta);
    Ok(Array1::from_vec(result).into_pyarray_bound(py).into())
}

/// tsdc_mle_find_root(t, intensity, ea, tau0, tp, ea_min, ea_max, beta) -> (ea, tau0, tp)
#[pyfunction]
fn tsdc_mle_find_root(
    t: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    ea: f64, tau0: f64, tp: f64,
    ea_min: f64, ea_max: f64, beta: f64,
) -> PyResult<(f64, f64, f64)> {
    let t_slice = t.as_array().to_vec();
    let int_slice = intensity.as_array().to_vec();
    let (mut ea_m, mut tau0_m, mut tp_m) = (ea, tau0, tp);
    match tsdc::mle_tsdc_find_root(&t_slice, &int_slice, &mut ea_m, &mut tau0_m, &mut tp_m, ea_min, ea_max, beta) {
        Ok(_) => Ok((ea_m, tau0_m, tp_m)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

/// tsdc_mle_lbfgsb(t, intensity, ea, tau0, tp, ea_min, ea_max, t_min, t_max, beta) -> (ea, tau0, tp)
#[pyfunction]
fn tsdc_mle_lbfgsb(
    t: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    ea: f64, tau0: f64, tp: f64,
    ea_min: f64, ea_max: f64, t_min: f64, t_max: f64, beta: f64,
) -> PyResult<(f64, f64, f64)> {
    let t_slice = t.as_array().to_vec();
    let int_slice = intensity.as_array().to_vec();
    let (mut ea_m, mut tau0_m, mut tp_m) = (ea, tau0, tp);
    match tsdc::mle_tsdc_lbfgsb(&t_slice, &int_slice, &mut ea_m, &mut tau0_m, &mut tp_m, ea_min, ea_max, t_min, t_max, beta) {
        Ok(_) => Ok((ea_m, tau0_m, tp_m)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

#[pymodule]
fn empeaks_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_em_loop, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_voigt_predict, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_voigt_mle_conditional_max, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_voigt_mle_full_optimization, m)?)?;
    m.add_function(wrap_pyfunction!(lorentzian_mle, m)?)?;
    m.add_function(wrap_pyfunction!(doniach_sunjic_mle, m)?)?;
    m.add_function(wrap_pyfunction!(tsdc_predict, m)?)?;
    m.add_function(wrap_pyfunction!(tsdc_mle_find_root, m)?)?;
    m.add_function(wrap_pyfunction!(tsdc_mle_lbfgsb, m)?)?;
    Ok(())
}
