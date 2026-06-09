use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadwriteArray1, IntoPyArray as _};
use ndarray::Array1;

mod gaussian;
mod em_engine;
mod em_gamma_ll;
mod pseudo_voigt;
mod lorentzian;
mod doniach_sunjic;
mod tsdc;
mod voigt;
mod background;


#[pyfunction]
#[pyo3(signature = (x, intensity, mu, sigma, pi, dirichlet_alpha,
                    fix_mu, fix_sigma, max_iter, r_eps, x_min, x_max,
                    bg_type=0u8, s_tri=0.0f64, spline_basis_preds=None))]
fn run_em_loop<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    intensity: PyReadonlyArray1<'py, f64>,
    mut mu: PyReadwriteArray1<'py, f64>,
    mut sigma: PyReadwriteArray1<'py, f64>,
    mut pi: PyReadwriteArray1<'py, f64>,
    dirichlet_alpha: PyReadonlyArray1<'py, f64>,
    fix_mu: Vec<bool>,
    fix_sigma: Vec<bool>,
    max_iter: usize,
    r_eps: f64,
    x_min: f64, x_max: f64,
    bg_type: u8, s_tri: f64,
    spline_basis_preds: Option<Vec<Vec<f64>>>,
) -> PyResult<(usize, PyObject, PyObject, f64)> {
    let x_vec: Vec<f64>  = x.as_array().to_vec();
    let int_vec: Vec<f64> = intensity.as_array().to_vec();

    let mut mu_vec: Vec<f64>    = mu.as_array().to_vec();
    let mut sigma_vec: Vec<f64> = sigma.as_array().to_vec();
    let mut pi_vec: Vec<f64>    = pi.as_array().to_vec();
    let da_slice: Vec<f64>      = dirichlet_alpha.as_array().to_vec();
    let sbp: Vec<Vec<f64>>      = spline_basis_preds.unwrap_or_default();

    let (total_iter, ll_hist, res_hist, s_tri_out) = py.allow_threads(|| {
        em_engine::run_em_loop(
            &x_vec, &int_vec,
            &mut mu_vec, &mut sigma_vec, &mut pi_vec,
            &da_slice, &fix_mu, &fix_sigma, max_iter, r_eps,
            x_min, x_max, bg_type, s_tri, &sbp,
        )
    });

    mu.as_array_mut().iter_mut().zip(mu_vec.iter()).for_each(|(dst, &src)| *dst = src);
    sigma.as_array_mut().iter_mut().zip(sigma_vec.iter()).for_each(|(dst, &src)| *dst = src);
    pi.as_array_mut().iter_mut().zip(pi_vec.iter()).for_each(|(dst, &src)| *dst = src);

    let ll_py  = Array1::from_vec(ll_hist).into_pyarray_bound(py).into();
    let res_py = Array1::from_vec(res_hist).into_pyarray_bound(py).into();

    Ok((total_iter, ll_py, res_py, s_tri_out))
}

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

#[pyfunction]
#[pyo3(signature = (x, intensity, x0, gamma, eta, x_min, x_max,
                    fix_x0=false, fix_gamma=false, fix_eta=false))]
fn pseudo_voigt_mle_conditional_max(
    x: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    x0: f64, gamma: f64, eta: f64,
    x_min: f64, x_max: f64,
    fix_x0: bool, fix_gamma: bool, fix_eta: bool,
) -> PyResult<(f64, f64, f64)> {
    let x_slice: Vec<f64> = x.as_array().to_vec();
    let int_slice: Vec<f64> = intensity.as_array().to_vec();
    let (mut x0, mut gamma, mut eta) = (x0, gamma, eta);
    pseudo_voigt::mle_conditional_max(
        &x_slice, &int_slice, &mut x0, &mut gamma, &mut eta,
        fix_x0, fix_gamma, fix_eta, x_min, x_max,
    );
    Ok((x0, gamma, eta))
}

#[pyfunction]
#[pyo3(signature = (x, intensity, x0, gamma, eta,
                    x_min, x_max, gamma_min, gamma_max, max_iter, r_eps,
                    fix_x0=false, fix_gamma=false, fix_eta=false))]
fn pseudo_voigt_mle_full_optimization(
    x: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    x0: f64, gamma: f64, eta: f64,
    x_min: f64, x_max: f64,
    gamma_min: f64, gamma_max: f64,
    max_iter: usize, r_eps: f64,
    fix_x0: bool, fix_gamma: bool, fix_eta: bool,
) -> PyResult<(f64, f64, f64)> {
    let x_slice: Vec<f64> = x.as_array().to_vec();
    let int_slice: Vec<f64> = intensity.as_array().to_vec();
    let (mut x0, mut gamma, mut eta) = (x0, gamma, eta);
    pseudo_voigt::mle_full_optimization(
        &x_slice, &int_slice,
        &mut x0, &mut gamma, &mut eta,
        fix_x0, fix_gamma, fix_eta,
        x_min, x_max, gamma_min, gamma_max,
        max_iter, r_eps,
    );
    Ok((x0, gamma, eta))
}

#[pyfunction]
#[pyo3(signature = (x, intensity, x0, gamma, x_min, x_max,
                    fix_x0=false, fix_gamma=false))]
fn lorentzian_mle(
    x: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    x0: f64,
    gamma: f64,
    x_min: f64,
    x_max: f64,
    fix_x0: bool,
    fix_gamma: bool,
) -> PyResult<(f64, f64)> {
    let x_slice: Vec<f64> = x.as_array().to_vec();
    let int_slice: Vec<f64> = intensity.as_array().to_vec();
    let (mut x0, mut gamma) = (x0, gamma);
    lorentzian::mle_lorentzian(&x_slice, &int_slice, &mut x0, &mut gamma, fix_x0, fix_gamma, x_min, x_max);
    Ok((x0, gamma))
}

#[pyfunction]
#[pyo3(signature = (x, intensity, x0, gamma, alpha,
                    x_min, x_max, gamma_min, gamma_max, alpha_min, alpha_max,
                    fix_x0=false, fix_gamma=false, fix_alpha=false))]
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
    fix_x0: bool,
    fix_gamma: bool,
    fix_alpha: bool,
) -> PyResult<(f64, f64, f64)> {
    let x_slice: Vec<f64> = x.as_array().to_vec();
    let int_slice: Vec<f64> = intensity.as_array().to_vec();
    let (mut x0, mut gamma, mut alpha) = (x0, gamma, alpha);
    doniach_sunjic::mle_doniach_sunjic(
        &x_slice, &int_slice,
        &mut x0, &mut gamma, &mut alpha,
        fix_x0, fix_gamma, fix_alpha,
        x_min, x_max, gamma_min, gamma_max, alpha_min, alpha_max,
    );
    Ok((x0, gamma, alpha))
}

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

#[pyfunction]
#[pyo3(signature = (t, intensity, ea, tau0, tp, ea_min, ea_max, beta,
                    fix_ea=false, fix_tp=false))]
fn tsdc_mle_find_root(
    t: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    ea: f64, tau0: f64, tp: f64,
    ea_min: f64, ea_max: f64, beta: f64,
    fix_ea: bool, fix_tp: bool,
) -> PyResult<(f64, f64, f64)> {
    let t_slice = t.as_array().to_vec();
    let int_slice = intensity.as_array().to_vec();
    let (mut ea_m, mut tau0_m, mut tp_m) = (ea, tau0, tp);
    match tsdc::mle_tsdc_find_root(&t_slice, &int_slice, &mut ea_m, &mut tau0_m, &mut tp_m,
                                    fix_ea, fix_tp, ea_min, ea_max, beta) {
        Ok(_) => Ok((ea_m, tau0_m, tp_m)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

#[pyfunction]
#[pyo3(signature = (t, intensity, ea, tau0, tp, ea_min, ea_max, t_min, t_max, beta,
                    fix_ea=false, fix_tp=false))]
fn tsdc_mle_lbfgsb(
    t: PyReadonlyArray1<'_, f64>,
    intensity: PyReadonlyArray1<'_, f64>,
    ea: f64, tau0: f64, tp: f64,
    ea_min: f64, ea_max: f64, t_min: f64, t_max: f64, beta: f64,
    fix_ea: bool, fix_tp: bool,
) -> PyResult<(f64, f64, f64)> {
    let t_slice = t.as_array().to_vec();
    let int_slice = intensity.as_array().to_vec();
    let (mut ea_m, mut tau0_m, mut tp_m) = (ea, tau0, tp);
    match tsdc::mle_tsdc_lbfgsb(&t_slice, &int_slice, &mut ea_m, &mut tau0_m, &mut tp_m,
                                  fix_ea, fix_tp, ea_min, ea_max, t_min, t_max, beta) {
        Ok(_) => Ok((ea_m, tau0_m, tp_m)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

#[pyfunction]
#[pyo3(signature = (x, intensity, x0, gamma_pv, eta, pi, dirichlet_alpha,
                    fix_x0, fix_gamma, fix_eta,
                    use_full_opt, x_min, x_max, gamma_min, gamma_max, max_iter, r_eps,
                    bg_type=0u8, s_tri=0.0f64, spline_basis_preds=None))]
fn run_pv_em_loop<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    intensity: PyReadonlyArray1<'py, f64>,
    mut x0: PyReadwriteArray1<'py, f64>,
    mut gamma_pv: PyReadwriteArray1<'py, f64>,
    mut eta: PyReadwriteArray1<'py, f64>,
    mut pi: PyReadwriteArray1<'py, f64>,
    dirichlet_alpha: PyReadonlyArray1<'py, f64>,
    fix_x0: Vec<bool>,
    fix_gamma: Vec<bool>,
    fix_eta: Vec<bool>,
    use_full_opt: bool,
    x_min: f64, x_max: f64, gamma_min: f64, gamma_max: f64,
    max_iter: usize, r_eps: f64,
    bg_type: u8, s_tri: f64,
    spline_basis_preds: Option<Vec<Vec<f64>>>,
) -> PyResult<(usize, PyObject, PyObject, f64)> {
    let x_vec = x.as_array().to_vec();
    let int_vec = intensity.as_array().to_vec();
    let da_vec = dirichlet_alpha.as_array().to_vec();

    let mut x0_vec = x0.as_array().to_vec();
    let mut gam_vec = gamma_pv.as_array().to_vec();
    let mut eta_vec = eta.as_array().to_vec();
    let mut pi_vec = pi.as_array().to_vec();
    let sbp: Vec<Vec<f64>> = spline_basis_preds.unwrap_or_default();

    let (total_iter, ll_hist, res_hist, s_tri_out) = py.allow_threads(|| {
        em_engine::run_pv_em_loop(
            &x_vec, &int_vec,
            &mut x0_vec, &mut gam_vec, &mut eta_vec, &mut pi_vec,
            &da_vec, &fix_x0, &fix_gamma, &fix_eta, use_full_opt,
            x_min, x_max, gamma_min, gamma_max,
            max_iter, r_eps, bg_type, s_tri, &sbp,
        )
    });

    x0.as_array_mut().iter_mut().zip(x0_vec.iter()).for_each(|(d, &s)| *d = s);
    gamma_pv.as_array_mut().iter_mut().zip(gam_vec.iter()).for_each(|(d, &s)| *d = s);
    eta.as_array_mut().iter_mut().zip(eta_vec.iter()).for_each(|(d, &s)| *d = s);
    pi.as_array_mut().iter_mut().zip(pi_vec.iter()).for_each(|(d, &s)| *d = s);

    let ll_py  = Array1::from_vec(ll_hist).into_pyarray_bound(py).into();
    let res_py = Array1::from_vec(res_hist).into_pyarray_bound(py).into();
    Ok((total_iter, ll_py, res_py, s_tri_out))
}

#[pyfunction]
#[pyo3(signature = (x, intensity, x0, gamma_lo, pi, dirichlet_alpha,
                    fix_x0, fix_gamma,
                    x_min, x_max, max_iter, r_eps, bg_type=0u8, s_tri=0.0f64,
                    spline_basis_preds=None))]
fn run_lorentzian_em_loop<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    intensity: PyReadonlyArray1<'py, f64>,
    mut x0: PyReadwriteArray1<'py, f64>,
    mut gamma_lo: PyReadwriteArray1<'py, f64>,
    mut pi: PyReadwriteArray1<'py, f64>,
    dirichlet_alpha: PyReadonlyArray1<'py, f64>,
    fix_x0: Vec<bool>,
    fix_gamma: Vec<bool>,
    x_min: f64, x_max: f64,
    max_iter: usize, r_eps: f64,
    bg_type: u8, s_tri: f64,
    spline_basis_preds: Option<Vec<Vec<f64>>>,
) -> PyResult<(usize, PyObject, PyObject, f64)> {
    let x_vec = x.as_array().to_vec();
    let int_vec = intensity.as_array().to_vec();
    let da_vec = dirichlet_alpha.as_array().to_vec();

    let mut x0_vec = x0.as_array().to_vec();
    let mut gam_vec = gamma_lo.as_array().to_vec();
    let mut pi_vec = pi.as_array().to_vec();
    let sbp: Vec<Vec<f64>> = spline_basis_preds.unwrap_or_default();

    let (total_iter, ll_hist, res_hist, s_tri_out) = py.allow_threads(|| {
        em_engine::run_lorentzian_em_loop(
            &x_vec, &int_vec,
            &mut x0_vec, &mut gam_vec, &mut pi_vec,
            &da_vec, &fix_x0, &fix_gamma, x_min, x_max,
            max_iter, r_eps, bg_type, s_tri, &sbp,
        )
    });

    x0.as_array_mut().iter_mut().zip(x0_vec.iter()).for_each(|(d, &s)| *d = s);
    gamma_lo.as_array_mut().iter_mut().zip(gam_vec.iter()).for_each(|(d, &s)| *d = s);
    pi.as_array_mut().iter_mut().zip(pi_vec.iter()).for_each(|(d, &s)| *d = s);

    let ll_py  = Array1::from_vec(ll_hist).into_pyarray_bound(py).into();
    let res_py = Array1::from_vec(res_hist).into_pyarray_bound(py).into();
    Ok((total_iter, ll_py, res_py, s_tri_out))
}

#[pyfunction]
#[pyo3(signature = (x, intensity, x0, gamma_ds, alpha, pi, dirichlet_alpha,
                    fix_x0, fix_gamma, fix_alpha,
                    x_min, x_max, gamma_min, gamma_max, alpha_min, alpha_max, max_iter, r_eps,
                    bg_type=0u8, s_tri=0.0f64, spline_basis_preds=None))]
fn run_ds_em_loop<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    intensity: PyReadonlyArray1<'py, f64>,
    mut x0: PyReadwriteArray1<'py, f64>,
    mut gamma_ds: PyReadwriteArray1<'py, f64>,
    mut alpha: PyReadwriteArray1<'py, f64>,
    mut pi: PyReadwriteArray1<'py, f64>,
    dirichlet_alpha: PyReadonlyArray1<'py, f64>,
    fix_x0: Vec<bool>,
    fix_gamma: Vec<bool>,
    fix_alpha: Vec<bool>,
    x_min: f64, x_max: f64, gamma_min: f64, gamma_max: f64,
    alpha_min: f64, alpha_max: f64,
    max_iter: usize, r_eps: f64,
    bg_type: u8, s_tri: f64,
    spline_basis_preds: Option<Vec<Vec<f64>>>,
) -> PyResult<(usize, PyObject, PyObject, f64)> {
    let x_vec = x.as_array().to_vec();
    let int_vec = intensity.as_array().to_vec();
    let da_vec = dirichlet_alpha.as_array().to_vec();

    let mut x0_vec = x0.as_array().to_vec();
    let mut gam_vec = gamma_ds.as_array().to_vec();
    let mut alp_vec = alpha.as_array().to_vec();
    let mut pi_vec = pi.as_array().to_vec();
    let sbp: Vec<Vec<f64>> = spline_basis_preds.unwrap_or_default();

    let (total_iter, ll_hist, res_hist, s_tri_out) = py.allow_threads(|| {
        em_engine::run_ds_em_loop(
            &x_vec, &int_vec,
            &mut x0_vec, &mut gam_vec, &mut alp_vec, &mut pi_vec,
            &da_vec, &fix_x0, &fix_gamma, &fix_alpha,
            x_min, x_max, gamma_min, gamma_max, alpha_min, alpha_max,
            max_iter, r_eps, bg_type, s_tri, &sbp,
        )
    });

    x0.as_array_mut().iter_mut().zip(x0_vec.iter()).for_each(|(d, &s)| *d = s);
    gamma_ds.as_array_mut().iter_mut().zip(gam_vec.iter()).for_each(|(d, &s)| *d = s);
    alpha.as_array_mut().iter_mut().zip(alp_vec.iter()).for_each(|(d, &s)| *d = s);
    pi.as_array_mut().iter_mut().zip(pi_vec.iter()).for_each(|(d, &s)| *d = s);

    let ll_py  = Array1::from_vec(ll_hist).into_pyarray_bound(py).into();
    let res_py = Array1::from_vec(res_hist).into_pyarray_bound(py).into();
    Ok((total_iter, ll_py, res_py, s_tri_out))
}

#[pyfunction]
#[pyo3(signature = (t, intensity, ea, tau0, tp, pi, dirichlet_alpha,
                    fix_ea, fix_tp,
                    beta, ea_min, ea_max, t_min, t_max, max_iter, r_eps,
                    bg_type=0u8, s_tri=0.0f64, spline_basis_preds=None))]
fn run_tsdc_em_loop<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    intensity: PyReadonlyArray1<'py, f64>,
    mut ea: PyReadwriteArray1<'py, f64>,
    mut tau0: PyReadwriteArray1<'py, f64>,
    mut tp: PyReadwriteArray1<'py, f64>,
    mut pi: PyReadwriteArray1<'py, f64>,
    dirichlet_alpha: PyReadonlyArray1<'py, f64>,
    fix_ea: Vec<bool>,
    fix_tp: Vec<bool>,
    beta: f64,
    ea_min: f64, ea_max: f64, t_min: f64, t_max: f64,
    max_iter: usize, r_eps: f64,
    bg_type: u8, s_tri: f64,
    spline_basis_preds: Option<Vec<Vec<f64>>>,
) -> PyResult<(usize, PyObject, PyObject, f64)> {
    let t_vec = t.as_array().to_vec();
    let int_vec = intensity.as_array().to_vec();
    let da_vec = dirichlet_alpha.as_array().to_vec();

    let mut ea_vec = ea.as_array().to_vec();
    let mut tau0_vec = tau0.as_array().to_vec();
    let mut tp_vec = tp.as_array().to_vec();
    let mut pi_vec = pi.as_array().to_vec();
    let sbp: Vec<Vec<f64>> = spline_basis_preds.unwrap_or_default();

    let (total_iter, ll_hist, res_hist, s_tri_out) = py.allow_threads(|| {
        em_engine::run_tsdc_em_loop(
            &t_vec, &int_vec,
            &mut ea_vec, &mut tau0_vec, &mut tp_vec, &mut pi_vec,
            &da_vec, &fix_ea, &fix_tp, beta,
            ea_min, ea_max, t_min, t_max,
            max_iter, r_eps, bg_type, s_tri, &sbp,
        )
    });

    ea.as_array_mut().iter_mut().zip(ea_vec.iter()).for_each(|(d, &s)| *d = s);
    tau0.as_array_mut().iter_mut().zip(tau0_vec.iter()).for_each(|(d, &s)| *d = s);
    tp.as_array_mut().iter_mut().zip(tp_vec.iter()).for_each(|(d, &s)| *d = s);
    pi.as_array_mut().iter_mut().zip(pi_vec.iter()).for_each(|(d, &s)| *d = s);

    let ll_py  = Array1::from_vec(ll_hist).into_pyarray_bound(py).into();
    let res_py = Array1::from_vec(res_hist).into_pyarray_bound(py).into();
    Ok((total_iter, ll_py, res_py, s_tri_out))
}

#[pyfunction]
#[pyo3(signature = (x, intensity, x0, sigma_v, gamma_v, pi, dirichlet_alpha,
                    fix_x0, fix_sigma, fix_gamma,
                    x_min, x_max, sigma_min, sigma_max, gamma_min, gamma_max,
                    max_iter, r_eps, bg_type=0u8, s_tri=0.0f64, spline_basis_preds=None))]
fn run_voigt_em_loop<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    intensity: PyReadonlyArray1<'py, f64>,
    mut x0: PyReadwriteArray1<'py, f64>,
    mut sigma_v: PyReadwriteArray1<'py, f64>,
    mut gamma_v: PyReadwriteArray1<'py, f64>,
    mut pi: PyReadwriteArray1<'py, f64>,
    dirichlet_alpha: PyReadonlyArray1<'py, f64>,
    fix_x0: Vec<bool>,
    fix_sigma: Vec<bool>,
    fix_gamma: Vec<bool>,
    x_min: f64, x_max: f64,
    sigma_min: f64, sigma_max: f64,
    gamma_min: f64, gamma_max: f64,
    max_iter: usize, r_eps: f64,
    bg_type: u8, s_tri: f64,
    spline_basis_preds: Option<Vec<Vec<f64>>>,
) -> PyResult<(usize, PyObject, PyObject, f64)> {
    let x_vec = x.as_array().to_vec();
    let int_vec = intensity.as_array().to_vec();
    let da_vec = dirichlet_alpha.as_array().to_vec();

    let mut x0_vec = x0.as_array().to_vec();
    let mut sig_vec = sigma_v.as_array().to_vec();
    let mut gam_vec = gamma_v.as_array().to_vec();
    let mut pi_vec = pi.as_array().to_vec();
    let sbp: Vec<Vec<f64>> = spline_basis_preds.unwrap_or_default();

    let (total_iter, ll_hist, res_hist, s_tri_out) = py.allow_threads(|| {
        em_engine::run_voigt_em_loop(
            &x_vec, &int_vec,
            &mut x0_vec, &mut sig_vec, &mut gam_vec, &mut pi_vec,
            &da_vec, &fix_x0, &fix_sigma, &fix_gamma,
            x_min, x_max, sigma_min, sigma_max, gamma_min, gamma_max,
            max_iter, r_eps, bg_type, s_tri, &sbp,
        )
    });

    x0.as_array_mut().iter_mut().zip(x0_vec.iter()).for_each(|(d, &s)| *d = s);
    sigma_v.as_array_mut().iter_mut().zip(sig_vec.iter()).for_each(|(d, &s)| *d = s);
    gamma_v.as_array_mut().iter_mut().zip(gam_vec.iter()).for_each(|(d, &s)| *d = s);
    pi.as_array_mut().iter_mut().zip(pi_vec.iter()).for_each(|(d, &s)| *d = s);

    let ll_py  = Array1::from_vec(ll_hist).into_pyarray_bound(py).into();
    let res_py = Array1::from_vec(res_hist).into_pyarray_bound(py).into();
    Ok((total_iter, ll_py, res_py, s_tri_out))
}

#[pymodule]
fn empeaks_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_em_loop, m)?)?;
    m.add_function(wrap_pyfunction!(run_pv_em_loop, m)?)?;
    m.add_function(wrap_pyfunction!(run_voigt_em_loop, m)?)?;
    m.add_function(wrap_pyfunction!(run_lorentzian_em_loop, m)?)?;
    m.add_function(wrap_pyfunction!(run_ds_em_loop, m)?)?;
    m.add_function(wrap_pyfunction!(run_tsdc_em_loop, m)?)?;
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
