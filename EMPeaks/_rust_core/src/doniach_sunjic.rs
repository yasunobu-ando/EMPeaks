// DoniachSunjic component MLE — port of _doniachsunjic.py full_optimization
//
// Gamma(1-alpha) is a positive scalar that cancels in pdf/z normalization,
// so it is omitted from the Rust implementation.

const PI: f64 = std::f64::consts::PI;

fn trapezoid(y: &[f64], x: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(xs, ys)| 0.5 * (ys[0] + ys[1]) * (xs[1] - xs[0]))
        .sum()
}

// Unnormalized DS value at a single point (Gamma(1-alpha) omitted — cancels in predict)
fn ds_unnorm(xi: f64, x0: f64, gamma: f64, alpha: f64) -> f64 {
    let pow_exp = (1.0 - alpha) / 2.0;
    let angle = (PI * alpha) / 2.0 + (1.0 - alpha) * ((xi - x0) / gamma).atan();
    let denom = ((xi - x0).powi(2) + gamma * gamma).powf(pow_exp);
    angle.cos() / denom
}

pub fn predict_inplace(x: &[f64], x0: f64, gamma: f64, alpha: f64, out: &mut [f64]) {
    for i in 0..x.len() {
        out[i] = ds_unnorm(x[i], x0, gamma, alpha);
    }
    let z = trapezoid(out, x);
    if z.abs() < 1e-300 {
        for i in 0..x.len() {
            out[i] = 0.0;
        }
        return;
    }
    for i in 0..x.len() {
        out[i] /= z;
    }
}

pub fn predict(x: &[f64], x0: f64, gamma: f64, alpha: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    predict_inplace(x, x0, gamma, alpha, &mut out);
    out
}

fn ll_ds(x: &[f64], intensity: &[f64], x0: f64, gamma: f64, alpha: f64) -> f64 {
    let pred = predict(x, x0, gamma, alpha);
    intensity
        .iter()
        .zip(pred.iter())
        .map(|(&ii, &p)| ii * (p + 1e-200).ln())
        .sum()
}

// MLE via L-BFGS-B on (x0, gamma, alpha): matches Python full_optimization
// Initial x0/gamma/alpha are used as-is (no empirical reset — matches Python)
pub fn mle_doniach_sunjic(
    x: &[f64],
    intensity: &[f64],
    x0: &mut f64,
    gamma: &mut f64,
    alpha: &mut f64,
    x_min: f64,
    x_max: f64,
    gamma_min: f64,
    gamma_max: f64,
    alpha_min: f64,
    alpha_max: f64,
) {
    let sum_w: f64 = intensity.iter().sum();
    if sum_w < 1e-300 {
        return;
    }

    let x_s = x.to_vec();
    let w_s = intensity.to_vec();
    let eps = 1e-7_f64;

    let bounds = vec![
        (x_min, x_max),
        (gamma_min, gamma_max),
        (alpha_min, alpha_max),
    ];
    let init = vec![*x0, *gamma, *alpha];

    if let Ok(state) = lbfgsb::lbfgsb(init, &bounds, |p, g| {
        let x0_p = p[0];
        let gam_p = p[1];
        let alp_p = p[2];
        let ll = ll_ds(&x_s, &w_s, x0_p, gam_p, alp_p);
        let ll_x = ll_ds(&x_s, &w_s, x0_p + eps, gam_p, alp_p);
        let ll_g = ll_ds(&x_s, &w_s, x0_p, gam_p + eps, alp_p);
        let ll_a = ll_ds(&x_s, &w_s, x0_p, gam_p, alp_p + eps);
        g[0] = -(ll_x - ll) / eps;
        g[1] = -(ll_g - ll) / eps;
        g[2] = -(ll_a - ll) / eps;
        Ok(-ll)
    }) {
        *x0 = state.x()[0];
        *gamma = state.x()[1].clamp(gamma_min, gamma_max);
        *alpha = state.x()[2].clamp(alpha_min, alpha_max);
    }
}
