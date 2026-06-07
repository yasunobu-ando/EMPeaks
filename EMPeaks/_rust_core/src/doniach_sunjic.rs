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
// fix_x0/fix_gamma/fix_alpha: skip optimizing those parameters.
pub fn mle_doniach_sunjic(
    x: &[f64],
    intensity: &[f64],
    x0: &mut f64,
    gamma: &mut f64,
    alpha: &mut f64,
    fix_x0: bool,
    fix_gamma: bool,
    fix_alpha: bool,
    x_min: f64,
    x_max: f64,
    gamma_min: f64,
    gamma_max: f64,
    alpha_min: f64,
    alpha_max: f64,
) {
    if fix_x0 && fix_gamma && fix_alpha { return; }

    let sum_w: f64 = intensity.iter().sum();
    if sum_w < 1e-300 { return; }

    let x_s = x.to_vec();
    let w_s = intensity.to_vec();
    let eps = 1e-7_f64;
    let x0_fixed = *x0;
    let gamma_fixed = *gamma;
    let alpha_fixed = *alpha;

    let mut init = Vec::new();
    let mut bounds = Vec::new();
    if !fix_x0    { init.push(*x0);    bounds.push((x_min, x_max)); }
    if !fix_gamma { init.push(*gamma); bounds.push((gamma_min, gamma_max)); }
    if !fix_alpha { init.push(*alpha); bounds.push((alpha_min, alpha_max)); }

    let unpack = move |p: &[f64]| -> (f64, f64, f64) {
        let mut idx = 0;
        let x0_v  = if fix_x0    { x0_fixed }    else { let v = p[idx]; idx += 1; v };
        let gam_v = if fix_gamma { gamma_fixed } else { let v = p[idx]; idx += 1; v };
        let alp_v = if fix_alpha { alpha_fixed } else { let v = p[idx]; idx += 1; v };
        (x0_v, gam_v, alp_v)
    };

    if let Ok(state) = lbfgsb::lbfgsb(init, &bounds, |p, g| {
        let (x0_v, gam_v, alp_v) = unpack(p);
        let ll = ll_ds(&x_s, &w_s, x0_v, gam_v, alp_v);
        for i in 0..p.len() {
            let mut p_e = p.to_vec();
            p_e[i] += eps;
            let (x0_e, gam_e, alp_e) = unpack(&p_e);
            let ll_e = ll_ds(&x_s, &w_s, x0_e, gam_e, alp_e);
            g[i] = -(ll_e - ll) / eps;
        }
        Ok(-ll)
    }) {
        let (x0_r, gam_r, alp_r) = unpack(state.x());
        if !fix_x0    { *x0    = x0_r; }
        if !fix_gamma { *gamma = gam_r.clamp(gamma_min, gamma_max); }
        if !fix_alpha { *alpha = alp_r.clamp(alpha_min, alpha_max); }
    }
}
