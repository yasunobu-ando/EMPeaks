// Lorentzian component MLE — port of _lorentz.py minimize_bfgs

const PI: f64 = std::f64::consts::PI;
const LN2: f64 = std::f64::consts::LN_2;

// Trapezoidal rule matching numpy.trapezoid(y, x)
fn trapezoid(y: &[f64], x: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(xs, ys)| 0.5 * (ys[0] + ys[1]) * (xs[1] - xs[0]))
        .sum()
}

// Analytical Z: integral of Cauchy PDF over [x_min, x_max]
fn lorentz_z(x0: f64, gamma: f64, x_min: f64, x_max: f64) -> f64 {
    let hg = gamma / 2.0;
    1.0 / PI * ((x_max - x0) / hg).atan() - 1.0 / PI * ((x_min - x0) / hg).atan()
}

// Double-normalized PDF: matches Python Lorentzian.predict(x)
//   prob = 1/(pi*Z) * gamma/((x-x0)^2 + (gamma/2)^2)
//   return prob / trapezoid(prob, x)
pub fn predict_inplace(x: &[f64], x0: f64, gamma: f64, x_min: f64, x_max: f64, out: &mut [f64]) {
    let hg = gamma / 2.0;
    let z = lorentz_z(x0, gamma, x_min, x_max).max(1e-300);
    for i in 0..x.len() {
        out[i] = gamma / (PI * z * ((x[i] - x0).powi(2) + hg * hg));
    }
    let z_num = trapezoid(out, x).max(1e-300);
    for i in 0..x.len() {
        out[i] /= z_num;
    }
}

pub fn predict(x: &[f64], x0: f64, gamma: f64, x_min: f64, x_max: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    predict_inplace(x, x0, gamma, x_min, x_max, &mut out);
    out
}

fn ll_lo(x: &[f64], intensity: &[f64], x0: f64, gamma: f64, x_min: f64, x_max: f64) -> f64 {
    let pred = predict(x, x0, gamma, x_min, x_max);
    intensity
        .iter()
        .zip(pred.iter())
        .map(|(&ii, &p)| ii * (p + 1e-200).ln())
        .sum()
}

// MLE via L-BFGS-B: matches Python minimize_bfgs
// fix_x0/fix_gamma: skip optimizing those parameters.
// When not fixed, x0/gamma are reset to empirical estimates (matching Python's minimize_bfgs reinit).
pub fn mle_lorentzian(
    x: &[f64],
    intensity: &[f64],
    x0: &mut f64,
    gamma: &mut f64,
    fix_x0: bool,
    fix_gamma: bool,
    x_min: f64,
    x_max: f64,
) {
    if fix_x0 && fix_gamma { return; }

    let sum_w: f64 = intensity.iter().sum();
    if sum_w < 1e-300 { return; }

    let x_s = x.to_vec();
    let w_s = intensity.to_vec();
    let eps = 1e-7_f64;
    let x0_fixed = *x0;
    let gamma_fixed = *gamma;

    let mut init = Vec::new();
    let mut bounds = Vec::new();

    if !fix_x0 {
        let x0_emp = x.iter().zip(intensity.iter()).map(|(&xi, &wi)| xi * wi).sum::<f64>() / sum_w;
        init.push(x0_emp);
        bounds.push((x_min, x_max));
    }
    if !fix_gamma {
        let x2_emp = x.iter().zip(intensity.iter()).map(|(&xi, &wi)| xi * xi * wi).sum::<f64>() / sum_w;
        let gamma_emp = (2.0 * LN2 * x2_emp).sqrt().max(0.1);
        init.push(gamma_emp);
        bounds.push((0.1_f64, 2000.0_f64));
    }

    let unpack = move |p: &[f64]| -> (f64, f64) {
        let mut idx = 0;
        let x0_v  = if fix_x0    { x0_fixed }    else { let v = p[idx]; idx += 1; v };
        let gam_v = if fix_gamma { gamma_fixed } else { let v = p[idx]; idx += 1; v };
        (x0_v, gam_v)
    };

    if let Ok(state) = lbfgsb::lbfgsb(init, &bounds, |p, g| {
        let (x0_v, gam_v) = unpack(p);
        let ll = ll_lo(&x_s, &w_s, x0_v, gam_v, x_min, x_max);
        for i in 0..p.len() {
            let mut p_e = p.to_vec();
            p_e[i] += eps;
            let (x0_e, gam_e) = unpack(&p_e);
            let ll_e = ll_lo(&x_s, &w_s, x0_e, gam_e, x_min, x_max);
            g[i] = -(ll_e - ll) / eps;
        }
        Ok(-ll)
    }) {
        let (x0_r, gam_r) = unpack(state.x());
        if !fix_x0    { *x0    = x0_r; }
        if !fix_gamma { *gamma = gam_r.clamp(0.1, 2000.0); }
    }
}
