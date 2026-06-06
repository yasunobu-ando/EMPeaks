use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadwriteArray1, IntoPyArray as _};
use ndarray::Array1;

mod gaussian;
mod em_engine;

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

#[pymodule]
fn empeaks_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_em_loop, m)?)?;
    Ok(())
}
