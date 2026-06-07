use rayon::prelude::*;
use crate::gaussian;
use crate::em_gamma_ll;
use crate::pseudo_voigt;
use crate::lorentzian;
use crate::doniach_sunjic;
use crate::tsdc;
use crate::voigt;
use crate::background;

// ---------------------------------------------------------------------------
// Gaussian EM ループ（Phase 3.5-1: ndarray 依存を完全に除去）
// ---------------------------------------------------------------------------

fn gaussian_predict_all_inplace(x: &[f64], mu: &[f64], sigma: &[f64], preds: &mut [Vec<f64>]) {
    preds.par_iter_mut().zip(mu.par_iter().zip(sigma.par_iter()))
        .for_each(|(pred, (&m, &s))| {
            gaussian::predict_inplace(x, m, s, pred);
        });
}

pub fn run_em_loop(
    x: &[f64],
    intensity: &[f64],
    mu: &mut Vec<f64>,
    sigma: &mut Vec<f64>,
    pi: &mut Vec<f64>,
    dirichlet_alpha: &[f64],
    fix_mu: &[bool],
    fix_sigma: &[bool],
    max_iter: usize,
    r_eps: f64,
    x_min: f64,
    x_max: f64,
    bg_type: u8,
    mut s_tri: f64,
) -> (usize, Vec<f64>, Vec<f64>, f64) {
    let k_peaks = mu.len();
    let k_all = k_peaks + if bg_type != background::BG_NONE { 1 } else { 0 };
    let mut predictions = vec![vec![0.0; x.len()]; k_all];
    let mut mixture = vec![0.0; x.len()];
    let mut gamma = vec![vec![0.0; x.len()]; k_all];

    gaussian_predict_all_inplace(x, mu, sigma, &mut predictions[0..k_peaks]);
    update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
    let mut ll_0 = em_gamma_ll::compute_gamma_and_ll_inplace(intensity, &predictions, pi, &mut mixture, &mut gamma);

    let mut ll_hist = vec![ll_0];
    let mut res_hist = vec![0.0_f64];
    let mut total_iter = max_iter;

    for it in 0..max_iter {
        em_gamma_ll::update_pi(pi, intensity, &gamma, dirichlet_alpha);

        // M-step: parallel MLE per component
        let results: Vec<(f64, f64)> = (0..k_peaks).into_par_iter().map(|k| {
            let w: Vec<f64> = intensity.iter().zip(gamma[k].iter())
                .map(|(&i, &g)| i * g).collect();
            gaussian::mle(x, &w, mu[k], sigma[k], fix_mu[k], fix_sigma[k])
        }).collect();
        for (k, (new_mu, new_sigma)) in results.into_iter().enumerate() {
            mu[k] = new_mu;
            sigma[k] = new_sigma;
        }

        // Background MLE (only for linear)
        if bg_type == background::BG_LINEAR {
            let w_bg: Vec<f64> = intensity.iter().zip(gamma[k_peaks].iter())
                .map(|(&i, &g)| i * g).collect();
            s_tri = background::mle_linear(x, &w_bg, x_min, x_max);
        }

        gaussian_predict_all_inplace(x, mu, sigma, &mut predictions[0..k_peaks]);
        update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
        let ll = em_gamma_ll::compute_gamma_and_ll_inplace(intensity, &predictions, pi, &mut mixture, &mut gamma);

        let residual = (ll - ll_0) / ll_0.abs();
        ll_hist.push(ll);
        res_hist.push(residual);

        if residual.abs() < r_eps {
            total_iter = it + 1;
            break;
        }
        ll_0 = ll;
            }
    (total_iter, ll_hist, res_hist, s_tri)
}

// ---------------------------------------------------------------------------
// Phase 3.5/4: モデル別専用 EM ループ
//  - rayon による K-parallel predict/MLE (Phase 3.5-3)
//  - background サポート: bg_type 0=none 1=uniform 2=squareroot 3=linear (Phase 4)
//  - bg_type != 0 のとき pi の末尾要素がバックグラウンド混合比
//    predictions は [k=0..K_peaks] + [background] の K_all 本
// ---------------------------------------------------------------------------

fn update_predictions_with_bg_inplace(
    preds: &mut [Vec<f64>],
    k_peaks: usize,
    bg_type: u8,
    x: &[f64],
    x_min: f64,
    x_max: f64,
    s_tri: f64,
) {
    if bg_type != background::BG_NONE {
        background::predict_inplace(bg_type, x, x_min, x_max, s_tri, &mut preds[k_peaks]);
    }
}

/// PseudoVoigt 専用 EM ループ
///
/// bg_type: 0=none, 1=uniform, 2=squareroot, 3=linear
/// s_tri:   linear background の三角比パラメータ（bg_type==3 のみ更新）
/// x0 / gamma_pv / eta / pi は in-place 更新。
/// Returns (total_iter, ll_hist, res_hist, s_tri)
pub fn run_pv_em_loop(
    x: &[f64],
    intensity: &[f64],
    x0: &mut Vec<f64>,
    gamma_pv: &mut Vec<f64>,
    eta: &mut Vec<f64>,
    pi: &mut Vec<f64>,
    dirichlet_alpha: &[f64],
    fix_x0: &[bool],
    fix_gamma: &[bool],
    fix_eta: &[bool],
    use_full_opt: bool,
    x_min: f64,
    x_max: f64,
    gamma_min: f64,
    gamma_max: f64,
    max_iter: usize,
    r_eps: f64,
    bg_type: u8,
    mut s_tri: f64,
) -> (usize, Vec<f64>, Vec<f64>, f64) {
    let k_peaks = x0.len();

    let k_all = k_peaks + if bg_type != background::BG_NONE { 1 } else { 0 };
    let mut predictions = vec![vec![0.0; x.len()]; k_all];
    let mut mixture = vec![0.0; x.len()];
    let mut gamma = vec![vec![0.0; x.len()]; k_all];

    predictions[0..k_peaks].par_iter_mut().enumerate().for_each(|(k, pred)| {
        pseudo_voigt::predict_inplace(x, x0[k], gamma_pv[k], eta[k], x_min, x_max, pred);
    });
    update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
    let mut ll_0 = em_gamma_ll::compute_gamma_and_ll_inplace(intensity, &predictions, pi, &mut mixture, &mut gamma);

    let mut ll_hist = vec![ll_0];
    let mut res_hist = vec![0.0_f64];
    let mut total_iter = max_iter;

    for it in 0..max_iter {
        em_gamma_ll::update_pi(pi, intensity, &gamma, dirichlet_alpha);

        // Parallel peak MLE
        let results: Vec<(f64, f64, f64)> = (0..k_peaks).into_par_iter().map(|k| {
            let w: Vec<f64> = intensity.iter().zip(gamma[k].iter())
                .map(|(&i, &g)| i * g).collect();
            let (mut x0_k, mut gam_k, mut eta_k) = (x0[k], gamma_pv[k], eta[k]);
            if use_full_opt {
                pseudo_voigt::mle_full_optimization(
                    x, &w, &mut x0_k, &mut gam_k, &mut eta_k,
                    fix_x0[k], fix_gamma[k], fix_eta[k],
                    x_min, x_max, gamma_min, gamma_max, max_iter, r_eps,
                );
            } else {
                pseudo_voigt::mle_conditional_max(
                    x, &w, &mut x0_k, &mut gam_k, &mut eta_k,
                    fix_x0[k], fix_gamma[k], fix_eta[k],
                    x_min, x_max,
                );
            }
            (x0_k, gam_k, eta_k)
        }).collect();
        for (k, (x0_k, gam_k, eta_k)) in results.into_iter().enumerate() {
            x0[k] = x0_k; gamma_pv[k] = gam_k; eta[k] = eta_k;
        }

        // Background MLE (only for linear)
        if bg_type == background::BG_LINEAR {
            let w_bg: Vec<f64> = intensity.iter().zip(gamma[k_peaks].iter())
                .map(|(&i, &g)| i * g).collect();
            s_tri = background::mle_linear(x, &w_bg, x_min, x_max);
        }

        predictions[0..k_peaks].par_iter_mut().enumerate().for_each(|(k, pred)| {
            pseudo_voigt::predict_inplace(x, x0[k], gamma_pv[k], eta[k], x_min, x_max, pred);
        });
        update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
        let ll = em_gamma_ll::compute_gamma_and_ll_inplace(intensity, &predictions, pi, &mut mixture, &mut gamma);

        let residual = (ll - ll_0) / ll_0.abs();
        ll_hist.push(ll);
        res_hist.push(residual);

        if residual.abs() < r_eps {
            total_iter = it + 1;
            break;
        }
        ll_0 = ll;
                let _ = it;
    }

    (total_iter, ll_hist, res_hist, s_tri)
}

/// Lorentzian 専用 EM ループ
pub fn run_lorentzian_em_loop(
    x: &[f64],
    intensity: &[f64],
    x0: &mut Vec<f64>,
    gamma_lo: &mut Vec<f64>,
    pi: &mut Vec<f64>,
    dirichlet_alpha: &[f64],
    fix_x0: &[bool],
    fix_gamma: &[bool],
    x_min: f64,
    x_max: f64,
    max_iter: usize,
    r_eps: f64,
    bg_type: u8,
    mut s_tri: f64,
) -> (usize, Vec<f64>, Vec<f64>, f64) {
    let k_peaks = x0.len();

    let k_all = k_peaks + if bg_type != background::BG_NONE { 1 } else { 0 };
    let mut predictions = vec![vec![0.0; x.len()]; k_all];
    let mut mixture = vec![0.0; x.len()];
    let mut gamma = vec![vec![0.0; x.len()]; k_all];

    predictions[0..k_peaks].par_iter_mut().enumerate().for_each(|(k, pred)| {
        lorentzian::predict_inplace(x, x0[k], gamma_lo[k], x_min, x_max, pred);
    });
    update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
    let mut ll_0 = em_gamma_ll::compute_gamma_and_ll_inplace(intensity, &predictions, pi, &mut mixture, &mut gamma);

    let mut ll_hist = vec![ll_0];
    let mut res_hist = vec![0.0_f64];
    let mut total_iter = max_iter;

    for it in 0..max_iter {
        em_gamma_ll::update_pi(pi, intensity, &gamma, dirichlet_alpha);

        let results: Vec<(f64, f64)> = (0..k_peaks).into_par_iter().map(|k| {
            let w: Vec<f64> = intensity.iter().zip(gamma[k].iter())
                .map(|(&i, &g)| i * g).collect();
            let (mut x0_k, mut gam_k) = (x0[k], gamma_lo[k]);
            lorentzian::mle_lorentzian(x, &w, &mut x0_k, &mut gam_k, fix_x0[k], fix_gamma[k], x_min, x_max);
            (x0_k, gam_k)
        }).collect();
        for (k, (x0_k, gam_k)) in results.into_iter().enumerate() {
            x0[k] = x0_k; gamma_lo[k] = gam_k;
        }

        if bg_type == background::BG_LINEAR {
            let w_bg: Vec<f64> = intensity.iter().zip(gamma[k_peaks].iter())
                .map(|(&i, &g)| i * g).collect();
            s_tri = background::mle_linear(x, &w_bg, x_min, x_max);
        }

        predictions[0..k_peaks].par_iter_mut().enumerate().for_each(|(k, pred)| {
            lorentzian::predict_inplace(x, x0[k], gamma_lo[k], x_min, x_max, pred);
        });
        update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
        let ll = em_gamma_ll::compute_gamma_and_ll_inplace(intensity, &predictions, pi, &mut mixture, &mut gamma);

        let residual = (ll - ll_0) / ll_0.abs();
        ll_hist.push(ll);
        res_hist.push(residual);

        if residual.abs() < r_eps {
            total_iter = it + 1;
            break;
        }
        ll_0 = ll;
                let _ = it;
    }

    (total_iter, ll_hist, res_hist, s_tri)
}

/// DoniachSunjic 専用 EM ループ
pub fn run_ds_em_loop(
    x: &[f64],
    intensity: &[f64],
    x0: &mut Vec<f64>,
    gamma_ds: &mut Vec<f64>,
    alpha: &mut Vec<f64>,
    pi: &mut Vec<f64>,
    dirichlet_alpha: &[f64],
    fix_x0: &[bool],
    fix_gamma: &[bool],
    fix_alpha: &[bool],
    x_min: f64,
    x_max: f64,
    gamma_min: f64,
    gamma_max: f64,
    alpha_min: f64,
    alpha_max: f64,
    max_iter: usize,
    r_eps: f64,
    bg_type: u8,
    mut s_tri: f64,
) -> (usize, Vec<f64>, Vec<f64>, f64) {
    let k_peaks = x0.len();

    let k_all = k_peaks + if bg_type != background::BG_NONE { 1 } else { 0 };
    let mut predictions = vec![vec![0.0; x.len()]; k_all];
    let mut mixture = vec![0.0; x.len()];
    let mut gamma = vec![vec![0.0; x.len()]; k_all];

    predictions[0..k_peaks].par_iter_mut().enumerate().for_each(|(k, pred)| {
        doniach_sunjic::predict_inplace(x, x0[k], gamma_ds[k], alpha[k], pred);
    });
    update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
    let mut ll_0 = em_gamma_ll::compute_gamma_and_ll_inplace(intensity, &predictions, pi, &mut mixture, &mut gamma);

    let mut ll_hist = vec![ll_0];
    let mut res_hist = vec![0.0_f64];
    let mut total_iter = max_iter;

    for it in 0..max_iter {
        em_gamma_ll::update_pi(pi, intensity, &gamma, dirichlet_alpha);

        let results: Vec<(f64, f64, f64)> = (0..k_peaks).into_par_iter().map(|k| {
            let w: Vec<f64> = intensity.iter().zip(gamma[k].iter())
                .map(|(&i, &g)| i * g).collect();
            let (mut x0_k, mut gam_k, mut alp_k) = (x0[k], gamma_ds[k], alpha[k]);
            doniach_sunjic::mle_doniach_sunjic(
                x, &w, &mut x0_k, &mut gam_k, &mut alp_k,
                fix_x0[k], fix_gamma[k], fix_alpha[k],
                x_min, x_max, gamma_min, gamma_max, alpha_min, alpha_max,
            );
            (x0_k, gam_k, alp_k)
        }).collect();
        for (k, (x0_k, gam_k, alp_k)) in results.into_iter().enumerate() {
            x0[k] = x0_k; gamma_ds[k] = gam_k; alpha[k] = alp_k;
        }

        if bg_type == background::BG_LINEAR {
            let w_bg: Vec<f64> = intensity.iter().zip(gamma[k_peaks].iter())
                .map(|(&i, &g)| i * g).collect();
            s_tri = background::mle_linear(x, &w_bg, x_min, x_max);
        }

        predictions[0..k_peaks].par_iter_mut().enumerate().for_each(|(k, pred)| {
            doniach_sunjic::predict_inplace(x, x0[k], gamma_ds[k], alpha[k], pred);
        });
        update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
        let ll = em_gamma_ll::compute_gamma_and_ll_inplace(intensity, &predictions, pi, &mut mixture, &mut gamma);

        let residual = (ll - ll_0) / ll_0.abs();
        ll_hist.push(ll);
        res_hist.push(residual);

        if residual.abs() < r_eps {
            total_iter = it + 1;
            break;
        }
        ll_0 = ll;
                let _ = it;
    }

    (total_iter, ll_hist, res_hist, s_tri)
}

/// TSDC 専用 EM ループ（背景モデルは対応しない: background == "none" 前提）
pub fn run_tsdc_em_loop(
    t: &[f64],
    intensity: &[f64],
    ea: &mut Vec<f64>,
    tau0: &mut Vec<f64>,
    tp: &mut Vec<f64>,
    pi: &mut Vec<f64>,
    dirichlet_alpha: &[f64],
    fix_ea: &[bool],
    fix_tp: &[bool],
    beta: f64,
    ea_min: f64,
    ea_max: f64,
    t_min: f64,
    t_max: f64,
    max_iter: usize,
    r_eps: f64,
    bg_type: u8,
    mut s_tri: f64,
) -> (usize, Vec<f64>, Vec<f64>, f64) {
    let k_peaks = ea.len();

    let k_all = k_peaks + if bg_type != background::BG_NONE { 1 } else { 0 };
    let mut predictions = vec![vec![0.0; t.len()]; k_all];
    let mut mixture = vec![0.0; t.len()];
    let mut gamma = vec![vec![0.0; t.len()]; k_all];

    predictions[0..k_peaks].iter_mut().enumerate().for_each(|(k, pred)| {
        tsdc::predict_inplace(t, ea[k], tau0[k], beta, pred);
    });
    update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, t, t_min, t_max, s_tri);
    let mut ll_0 = em_gamma_ll::compute_gamma_and_ll_inplace(intensity, &predictions, pi, &mut mixture, &mut gamma);

    let mut ll_hist = vec![ll_0];
    let mut res_hist = vec![0.0_f64];
    let mut total_iter = max_iter;

    for it in 0..max_iter {
        em_gamma_ll::update_pi(pi, intensity, &gamma, dirichlet_alpha);
        // TSDC MLE は xsf 特殊関数を使うため rayon は tsdc.rs 内の内部ループで適用済み
        for k in 0..k_peaks {
            let w: Vec<f64> = intensity.iter().zip(gamma[k].iter())
                .map(|(&i, &g)| i * g).collect();
            if w.iter().sum::<f64>() < 1e-300 { continue; }
            let ok = tsdc::mle_tsdc_find_root(
                t, &w, &mut ea[k], &mut tau0[k], &mut tp[k],
                fix_ea[k], fix_tp[k], ea_min, ea_max, beta,
            );
            if ok.is_err() {
                let _ = tsdc::mle_tsdc_lbfgsb(
                    t, &w, &mut ea[k], &mut tau0[k], &mut tp[k],
                    fix_ea[k], fix_tp[k], ea_min, ea_max, t_min, t_max, beta,
                );
            }
        }

        // Background MLE (only for linear)
        if bg_type == background::BG_LINEAR {
            let w_bg: Vec<f64> = intensity.iter().zip(gamma[k_peaks].iter())
                .map(|(&i, &g)| i * g).collect();
            s_tri = background::mle_linear(t, &w_bg, t_min, t_max);
        }

        predictions[0..k_peaks].iter_mut().enumerate().for_each(|(k, pred)| {
            tsdc::predict_inplace(t, ea[k], tau0[k], beta, pred);
        });
        update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, t, t_min, t_max, s_tri);
        let ll = em_gamma_ll::compute_gamma_and_ll_inplace(intensity, &predictions, pi, &mut mixture, &mut gamma);

        let residual = (ll - ll_0) / ll_0.abs();
        ll_hist.push(ll);
        res_hist.push(residual);

        if residual.abs() < r_eps {
            total_iter = it + 1;
            break;
        }
        ll_0 = ll;
        let _ = it;
    }

    (total_iter, ll_hist, res_hist, s_tri)
}

/// Voigt 専用 EM ループ（fix_ フラグ対応）
///
/// sigma_v: Gaussian σ パラメータ
/// gamma_v: Lorentzian γ パラメータ
/// fix_x0 / fix_sigma / fix_gamma: 各ピークの固定フラグ（長さ K）
/// Returns (total_iter, ll_hist, res_hist, s_tri)
pub fn run_voigt_em_loop(
    x: &[f64],
    intensity: &[f64],
    x0: &mut Vec<f64>,
    sigma_v: &mut Vec<f64>,
    gamma_v: &mut Vec<f64>,
    pi: &mut Vec<f64>,
    dirichlet_alpha: &[f64],
    fix_x0: &[bool],
    fix_sigma: &[bool],
    fix_gamma: &[bool],
    x_min: f64, x_max: f64,
    sigma_min: f64, sigma_max: f64,
    gamma_min: f64, gamma_max: f64,
    max_iter: usize,
    r_eps: f64,
    bg_type: u8,
    mut s_tri: f64,
) -> (usize, Vec<f64>, Vec<f64>, f64) {
    let k_peaks = x0.len();
    let k_all = k_peaks + if bg_type != background::BG_NONE { 1 } else { 0 };
    let mut predictions = vec![vec![0.0; x.len()]; k_all];
    let mut mixture = vec![0.0; x.len()];
    let mut gamma = vec![vec![0.0; x.len()]; k_all];

    predictions[0..k_peaks].par_iter_mut().enumerate().for_each(|(k, pred)| {
        voigt::predict_inplace(x, x0[k], sigma_v[k], gamma_v[k], x_min, x_max, pred);
    });
    update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
    let mut ll_0 = em_gamma_ll::compute_gamma_and_ll_inplace(
        intensity, &predictions, pi, &mut mixture, &mut gamma);

    let mut ll_hist = vec![ll_0];
    let mut res_hist = vec![0.0_f64];
    let mut total_iter = max_iter;

    for it in 0..max_iter {
        em_gamma_ll::update_pi(pi, intensity, &gamma, dirichlet_alpha);

        let results: Vec<(f64, f64, f64)> = (0..k_peaks).into_par_iter().map(|k| {
            let w: Vec<f64> = intensity.iter().zip(gamma[k].iter())
                .map(|(&i, &g)| i * g).collect();
            let (mut x0_k, mut sig_k, mut gam_k) = (x0[k], sigma_v[k], gamma_v[k]);
            voigt::mle_voigt(
                x, &w, &mut x0_k, &mut sig_k, &mut gam_k,
                fix_x0[k], fix_sigma[k], fix_gamma[k],
                x_min, x_max, sigma_min, sigma_max, gamma_min, gamma_max,
            );
            (x0_k, sig_k, gam_k)
        }).collect();
        for (k, (x0_k, sig_k, gam_k)) in results.into_iter().enumerate() {
            x0[k] = x0_k; sigma_v[k] = sig_k; gamma_v[k] = gam_k;
        }

        if bg_type == background::BG_LINEAR {
            let w_bg: Vec<f64> = intensity.iter().zip(gamma[k_peaks].iter())
                .map(|(&i, &g)| i * g).collect();
            s_tri = background::mle_linear(x, &w_bg, x_min, x_max);
        }

        predictions[0..k_peaks].par_iter_mut().enumerate().for_each(|(k, pred)| {
            voigt::predict_inplace(x, x0[k], sigma_v[k], gamma_v[k], x_min, x_max, pred);
        });
        update_predictions_with_bg_inplace(&mut predictions, k_peaks, bg_type, x, x_min, x_max, s_tri);
        let ll = em_gamma_ll::compute_gamma_and_ll_inplace(
            intensity, &predictions, pi, &mut mixture, &mut gamma);

        let residual = (ll - ll_0) / ll_0.abs();
        ll_hist.push(ll);
        res_hist.push(residual);

        if residual.abs() < r_eps {
            total_iter = it + 1;
            break;
        }
        ll_0 = ll;
        let _ = it;
    }

    (total_iter, ll_hist, res_hist, s_tri)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64).collect()
    }

    fn make_data() -> (Vec<f64>, Vec<f64>) {
        let x = linspace(-5.0, 5.0, 200);
        let mut intensity: Vec<f64> = x.iter().map(|&xi| {
            (-(xi * xi)).exp() + 0.5 * (-(xi - 2.0_f64).powi(2)).exp()
        }).collect();
        let sum: f64 = intensity.iter().sum();
        intensity.iter_mut().for_each(|v| *v /= sum);
        (x, intensity)
    }

    #[test]
    fn test_em_converges() {
        let (x, intensity) = make_data();
        let mut mu    = vec![-1.0, 1.0];
        let mut sigma = vec![1.0, 1.0];
        let mut pi    = vec![0.5, 0.5];
        let da        = vec![1.0, 1.0];
        let fix_mu    = vec![false, false];
        let fix_sigma = vec![false, false];
        let x_min = x[0];
        let x_max = *x.last().unwrap();

        let (iters, ll_hist, _, _) = run_em_loop(
            &x, &intensity,
            &mut mu, &mut sigma, &mut pi,
            &da, &fix_mu, &fix_sigma, 3000, 1e-9,
            x_min, x_max, 0, 0.0,
        );
        assert!(iters < 3000, "did not converge within 3000 iters");
        assert!(ll_hist.last().unwrap() > ll_hist.first().unwrap(),
                "LL did not increase");
    }
}
