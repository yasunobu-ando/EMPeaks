// PseudoVoigt component MLE — faithful port of _pseudo_voigt.py
// conditional_max: Python's _e_step + _cm_step_x0_gamma + _cm_step_eta
// full_optimization: Python's full_optimization (L-BFGS-B + eta grid search)

use rayon::prelude::*;
use roots::{find_root_brent, SimpleConvergency};

const PI: f64 = std::f64::consts::PI;
const SQRT2: f64 = std::f64::consts::SQRT_2;
const LN2: f64 = std::f64::consts::LN_2;

// ============================================================
// Basic PDF / normalization helpers
// ============================================================

fn normalization_z(x0: f64, gamma: f64, x_min: f64, x_max: f64) -> f64 {
    let hg = gamma / 2.0;
    (1.0 / PI) * (((x_max - x0) / hg).atan() - ((x_min - x0) / hg).atan())
}

fn gaussian_pdf(x: f64, x0: f64, sigma: f64) -> f64 {
    let norm = (2.0 * PI).sqrt() * sigma;
    let t = (x - x0) / sigma;
    (-0.5 * t * t).exp() / norm
}

// scipy cauchy.pdf(x, loc, scale) = scale / (pi * ((x-loc)^2 + scale^2))
fn cauchy_pdf(x: f64, loc: f64, scale: f64) -> f64 {
    scale / (PI * ((x - loc).powi(2) + scale * scale))
}

fn predict_scalar(x: f64, x0: f64, gamma: f64, eta: f64, x_min: f64, x_max: f64) -> f64 {
    let sigma = gamma / (2.0 * SQRT2 * LN2.sqrt());
    let hg = gamma / 2.0;
    let z = normalization_z(x0, gamma, x_min, x_max).max(1e-300);
    eta * gaussian_pdf(x, x0, sigma) + (1.0 - eta) / z * cauchy_pdf(x, x0, hg)
}

pub fn predict(x: &[f64], x0: f64, gamma: f64, eta: f64, x_min: f64, x_max: f64) -> Vec<f64> {
    x.iter()
        .map(|&xi| predict_scalar(xi, x0, gamma, eta, x_min, x_max))
        .collect()
}

// LL using predict (with Z normalization, epsilon 1e-20 matching Python _LL)
fn ll_from_predict(
    x: &[f64], intensity: &[f64],
    x0: f64, gamma: f64, eta: f64,
    x_min: f64, x_max: f64,
) -> f64 {
    x.iter()
        .zip(intensity.iter())
        .map(|(&xi, &ii)| {
            let p = predict_scalar(xi, x0, gamma, eta, x_min, x_max);
            ii * (p + 1e-20).ln()
        })
        .sum()
}

// ============================================================
// Inner E-step (within PseudoVoigt component)
// Matches Python's _e_step: gamma2 does NOT include 1/Z factor
// ============================================================

fn e_step_inner(
    x: &[f64], x0: f64, gamma: f64, eta: f64,
    x_min: f64, x_max: f64,
) -> (Vec<f64>, Vec<f64>) {
    let sigma = gamma / (2.0 * SQRT2 * LN2.sqrt());
    let hg = gamma / 2.0;
    let mut g1 = vec![0.0_f64; x.len()];
    let mut g2 = vec![0.0_f64; x.len()];

    for (i, &xi) in x.iter().enumerate() {
        let p = predict_scalar(xi, x0, gamma, eta, x_min, x_max) + 1e-300;
        g1[i] = eta * gaussian_pdf(xi, x0, sigma) / p;
        // gamma2: WITHOUT 1/Z (matching Python's _e_step)
        g2[i] = (1.0 - eta) * cauchy_pdf(xi, x0, hg) / p;
    }
    (g1, g2)
}

// ============================================================
// Derivatives of log Z for CM-step
// Matches Python's _LogZ_m and _LogZ_g
// ============================================================

fn log_z_grad_m(x0: f64, gamma: f64, x_min: f64, x_max: f64) -> f64 {
    let z = normalization_z(x0, gamma, x_min, x_max).max(1e-300);
    let hg = gamma / 2.0;
    1.0 / z * (-cauchy_pdf(x_max, x0, hg) + cauchy_pdf(x_min, x0, hg))
}

fn log_z_grad_g(x0: f64, gamma: f64, x_min: f64, x_max: f64) -> f64 {
    let z = normalization_z(x0, gamma, x_min, x_max).max(1e-300);
    let hg = gamma / 2.0;
    let hg2 = hg * hg;
    1.0 / (2.0 * PI * z)
        * ((x0 - x_max) / (hg2 + (x0 - x_max).powi(2))
            - (x0 - x_min) / (hg2 + (x0 - x_min).powi(2)))
}

// ============================================================
// Q functions for CM-step
// Matches Python's _qm and _qg
// ============================================================

fn q_m(
    m: f64, g: f64,
    x: &[f64], intensity: &[f64], gamma1: &[f64], gamma2: &[f64],
    x_min: f64, x_max: f64,
) -> f64 {
    let hg = g / 2.0;
    let hg2 = hg * hg;

    let n1: f64 = gamma1.iter().zip(intensity.iter()).map(|(g1, i)| g1 * i).sum();
    let n2: f64 = gamma2.iter().zip(intensity.iter()).map(|(g2, i)| g2 * i).sum();
    let logz_m = log_z_grad_m(m, g, x_min, x_max);

    let sum_g1_x: f64 = gamma1
        .iter()
        .zip(intensity.iter())
        .zip(x.iter())
        .map(|((g1, i), xi)| g1 * i * xi)
        .sum();
    let sum_g2_2x: f64 = gamma2
        .iter()
        .zip(intensity.iter())
        .zip(x.iter())
        .map(|((g2, i), xi)| g2 * i * 2.0 * xi / ((xi - m).powi(2) + hg2))
        .sum();
    let sum_g2_2: f64 = gamma2
        .iter()
        .zip(intensity.iter())
        .zip(x.iter())
        .map(|((g2, i), xi)| g2 * i * 2.0 / ((xi - m).powi(2) + hg2))
        .sum();

    let a = 8.0 * LN2 / (g * g) * sum_g1_x + sum_g2_2x - n2 * logz_m;
    let b = 8.0 * LN2 / (g * g) * n1 + sum_g2_2;
    a / (b + 1e-300)
}

fn q_g_fn(
    m: f64, g: f64,
    x: &[f64], intensity: &[f64], gamma1: &[f64], gamma2: &[f64],
    x_min: f64, x_max: f64,
) -> f64 {
    let hg = g / 2.0;
    let hg2 = hg * hg;

    let n1: f64 = gamma1.iter().zip(intensity.iter()).map(|(g1, i)| g1 * i).sum();
    let n2: f64 = gamma2.iter().zip(intensity.iter()).map(|(g2, i)| g2 * i).sum();
    let logz_g = log_z_grad_g(m, g, x_min, x_max);

    let sum_g2_inv: f64 = gamma2
        .iter()
        .zip(intensity.iter())
        .zip(x.iter())
        .map(|((g2, i), xi)| g2 * i / 2.0 / ((xi - m).powi(2) + hg2))
        .sum();
    let sum_g1_xm2: f64 = gamma1
        .iter()
        .zip(intensity.iter())
        .zip(x.iter())
        .map(|((g1, i), xi)| g1 * i * (xi - m).powi(2))
        .sum();

    -g.powi(4) * sum_g2_inv
        - n2 * g.powi(3) * logz_g
        + (n2 - n1) * g.powi(2)
        + 8.0 * LN2 * sum_g1_xm2
}

// ============================================================
// Find optimal gamma for given m using brentq
// Matches Python's _opt_g
// ============================================================

fn opt_gamma_given_m(
    m: f64,
    x: &[f64], intensity: &[f64], gamma1: &[f64], gamma2: &[f64],
    x_min: f64, x_max: f64,
) -> Option<f64> {
    let g_lo = 0.1_f64;
    let g_hi = x_max - x_min;

    let fa = q_g_fn(m, g_lo, x, intensity, gamma1, gamma2, x_min, x_max);
    let fb = q_g_fn(m, g_hi, x, intensity, gamma1, gamma2, x_min, x_max);

    if fa * fb >= 0.0 {
        return None;
    }

    let f = |g: f64| q_g_fn(m, g, x, intensity, gamma1, gamma2, x_min, x_max);
    let mut conv = SimpleConvergency { eps: 1e-10_f64, max_iter: 100 };
    find_root_brent(g_lo, g_hi, &f, &mut conv).ok()
}

// ============================================================
// CM-step for x0 and gamma
// Matches Python's _cm_step_x0_gamma (N=10)
// ============================================================

fn cm_step_x0_gamma(
    x: &[f64], intensity: &[f64], gamma1: &[f64], gamma2: &[f64],
    x0: &mut f64, gamma: &mut f64, eta: f64,
    x_min: f64, x_max: f64,
) {
    const N: usize = 10;

    // Coarse grid to find valid m range (where q_g changes sign in gamma)
    let m_grid: Vec<f64> = (0..N)
        .map(|i| x_min + i as f64 * (x_max - x_min) / (N - 1) as f64)
        .collect();

    let g_lo = 0.1_f64;
    let g_hi = x_max - x_min;

    let qg_at_glo: Vec<f64> = m_grid
        .iter()
        .map(|&m| q_g_fn(m, g_lo, x, intensity, gamma1, gamma2, x_min, x_max))
        .collect();
    let qg_at_ghi: Vec<f64> = m_grid
        .iter()
        .map(|&m| q_g_fn(m, g_hi, x, intensity, gamma1, gamma2, x_min, x_max))
        .collect();

    // Keep m where sign of qg differs between g_lo and g_hi
    let valid_m: Vec<f64> = (0..N)
        .filter(|&i| qg_at_glo[i].signum() * qg_at_ghi[i].signum() < 0.0)
        .map(|i| m_grid[i])
        .collect();

    if valid_m.is_empty() {
        return;
    }

    let m_lo = valid_m[0];
    let m_hi = *valid_m.last().unwrap();

    // Finer grid within [m_lo, m_hi] for finding roots of fm
    let span = m_hi - m_lo;
    let m_grid2: Vec<f64> = if span < 1e-12 {
        vec![m_lo]
    } else {
        (0..N)
            .map(|i| m_lo + i as f64 * span / (N - 1) as f64)
            .collect()
    };

    // fm(m) = qm(m, opt_g(m)) - m
    let fm_vals: Vec<f64> = m_grid2
        .iter()
        .map(|&m| {
            opt_gamma_given_m(m, x, intensity, gamma1, gamma2, x_min, x_max)
                .map(|g| q_m(m, g, x, intensity, gamma1, gamma2, x_min, x_max) - m)
                .unwrap_or(f64::NAN)
        })
        .collect();

    // Find sign-change intervals and brentq for each
    let mut roots_x0: Vec<f64> = Vec::new();
    let mut roots_gamma: Vec<f64> = Vec::new();

    for i in 0..m_grid2.len().saturating_sub(1) {
        let y1 = fm_vals[i];
        let y2 = fm_vals[i + 1];
        if y1.is_finite() && y2.is_finite() && y1 * y2 < 0.0 {
            let ma = m_grid2[i];
            let mb = m_grid2[i + 1];
            let f_brentq = |m: f64| {
                opt_gamma_given_m(m, x, intensity, gamma1, gamma2, x_min, x_max)
                    .map(|g| q_m(m, g, x, intensity, gamma1, gamma2, x_min, x_max) - m)
                    .unwrap_or(0.0)
            };
            let mut conv = SimpleConvergency { eps: 1e-10_f64, max_iter: 100 };
            if let Ok(root_m) = find_root_brent(ma, mb, &f_brentq, &mut conv) {
                if let Some(root_g) =
                    opt_gamma_given_m(root_m, x, intensity, gamma1, gamma2, x_min, x_max)
                {
                    roots_x0.push(root_m);
                    roots_gamma.push(root_g);
                }
            }
        }
    }

    if roots_x0.is_empty() {
        return;
    }

    // Select root with highest LL; uses unnormalized mixture (matches Python _cm_step_x0_gamma)
    let best_idx = roots_x0
        .iter()
        .zip(roots_gamma.iter())
        .map(|(&rm, &rg)| {
            let sigma = rg / (2.0 * SQRT2 * LN2.sqrt());
            let hg = rg / 2.0;
            let ll: f64 = x
                .iter()
                .zip(intensity.iter())
                .map(|(&xi, &ii)| {
                    let t = eta * gaussian_pdf(xi, rm, sigma)
                        + (1.0 - eta) * cauchy_pdf(xi, rm, hg);
                    ii * (t + 1e-20).ln()
                })
                .sum();
            ll
        })
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    *x0 = roots_x0[best_idx];
    *gamma = roots_gamma[best_idx];
}

// ============================================================
// CM-step for eta (grid search)
// Matches Python's _cm_step_eta: eta in [0.80, 0.99] step 0.01
// ============================================================

fn cm_step_eta(
    x: &[f64], intensity: &[f64],
    x0: f64, gamma: f64, x_min: f64, x_max: f64,
) -> f64 {
    // np.arange(0.8, 1, 0.01) → 20 values
    (0..20usize)
        .map(|i| 0.8 + i as f64 * 0.01)
        .map(|eta| {
            let ll = ll_from_predict(x, intensity, x0, gamma, eta, x_min, x_max);
            (eta, ll)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(eta, _)| eta)
        .unwrap_or(0.9)
}

// ============================================================
// Public: MLE via conditional maximization
// Matches Python's conditional_max (single pass)
// ============================================================

pub fn mle_conditional_max(
    x: &[f64], intensity: &[f64],
    x0: &mut f64, gamma: &mut f64, eta: &mut f64,
    x_min: f64, x_max: f64,
) {
    if intensity.iter().sum::<f64>() == 0.0 {
        return;
    }
    let (g1, g2) = e_step_inner(x, *x0, *gamma, *eta, x_min, x_max);
    cm_step_x0_gamma(x, intensity, &g1, &g2, x0, gamma, *eta, x_min, x_max);
    *eta = cm_step_eta(x, intensity, *x0, *gamma, x_min, x_max);
}

// ============================================================
// Public: MLE via L-BFGS-B + eta grid search
// Matches Python's full_optimization
// ============================================================

pub fn mle_full_optimization(
    x: &[f64], intensity: &[f64],
    x0: &mut f64, gamma: &mut f64, eta: &mut f64,
    x_min: f64, x_max: f64, gamma_min: f64, gamma_max: f64,
    max_iter: usize, r_eps: f64,
) {
    if intensity.iter().sum::<f64>() == 0.0 {
        return;
    }

    let x_s: Vec<f64> = x.to_vec();
    let w_s: Vec<f64> = intensity.to_vec();

    let mut cur_x0 = *x0;
    let mut cur_gamma = *gamma;
    let mut cur_eta = *eta;
    let mut ll_prev =
        ll_from_predict(&x_s, &w_s, cur_x0, cur_gamma, cur_eta, x_min, x_max);
    let eps_num = 1e-7_f64;

    for _ in 0..max_iter {
        let eta_fixed = cur_eta;
        let bounds = vec![(x_min, x_max), (gamma_min, gamma_max)];
        let init = vec![cur_x0, cur_gamma];

        if let Ok(state) = lbfgsb::lbfgsb(init, &bounds, |p, g| {
            let ll   = ll_from_predict(&x_s, &w_s, p[0],           p[1],           eta_fixed, x_min, x_max);
            let ll_x = ll_from_predict(&x_s, &w_s, p[0] + eps_num, p[1],           eta_fixed, x_min, x_max);
            let ll_g = ll_from_predict(&x_s, &w_s, p[0],           p[1] + eps_num, eta_fixed, x_min, x_max);
            g[0] = -(ll_x - ll) / eps_num;
            g[1] = -(ll_g - ll) / eps_num;
            Ok(-ll)
        }) {
            cur_x0    = state.x()[0];
            cur_gamma = state.x()[1].clamp(gamma_min, gamma_max);
        }

        // Grid search for eta: np.arange(0, 1.0, 0.01) = 100 values (parallel)
        cur_eta = (0..100usize)
            .into_par_iter()
            .map(|i| {
                let e  = i as f64 * 0.01;
                let ll = ll_from_predict(&x_s, &w_s, cur_x0, cur_gamma, e, x_min, x_max);
                (e, ll)
            })
            .reduce(
                || (0.5_f64, f64::NEG_INFINITY),
                |a, b| if b.1 > a.1 { b } else { a },
            )
            .0;

        let ll_new = ll_from_predict(&x_s, &w_s, cur_x0, cur_gamma, cur_eta, x_min, x_max);
        let residual = (ll_new - ll_prev) / ll_prev.abs().max(1e-300);

        if residual < r_eps {
            break;
        }
        ll_prev = ll_new;
    }

    *x0    = cur_x0;
    *gamma = cur_gamma;
    *eta   = cur_eta;
}
