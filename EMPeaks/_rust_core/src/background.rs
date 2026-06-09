use roots::{find_root_brent, SimpleConvergency};

pub const BG_NONE: u8 = 0;
pub const BG_UNIFORM: u8 = 1;
pub const BG_SQUAREROOT: u8 = 2;
pub const BG_LINEAR: u8 = 3;
pub const BG_SPLINE: u8 = 4;

pub fn predict_inplace(bg_type: u8, x: &[f64], x_min: f64, x_max: f64, s_tri: f64, out: &mut [f64]) {
    match bg_type {
        BG_UNIFORM => predict_uniform_inplace(x, x_min, x_max, out),
        BG_SQUAREROOT => predict_squareroot_inplace(x, x_min, x_max, out),
        BG_LINEAR => predict_linear_inplace(x, x_min, x_max, s_tri, out),
        _ => {
            for i in 0..x.len() {
                out[i] = 0.0;
            }
        }
    }
}

/// Dispatch predict for background models.
pub fn predict(bg_type: u8, x: &[f64], x_min: f64, x_max: f64, s_tri: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    predict_inplace(bg_type, x, x_min, x_max, s_tri, &mut out);
    out
}

fn predict_uniform_inplace(x: &[f64], x_min: f64, x_max: f64, out: &mut [f64]) {
    let width = x_max - x_min;
    let inv = if width > 0.0 { 1.0 / width } else { 0.0 };
    for i in 0..x.len() {
        let xi = x[i];
        out[i] = if xi >= x_min && xi <= x_max { inv } else { 0.0 };
    }
}

fn predict_uniform(x: &[f64], x_min: f64, x_max: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    predict_uniform_inplace(x, x_min, x_max, &mut out);
    out
}

fn predict_squareroot_inplace(x: &[f64], x_min: f64, x_max: f64, out: &mut [f64]) {
    let norm = 2.0 / 3.0 * (x_max.powf(1.5) - x_min.powf(1.5));
    let norm_safe = if norm.abs() > 1e-300 { norm } else { 1.0 };
    for i in 0..x.len() {
        let xi = x[i];
        out[i] = if xi >= x_min && xi <= x_max && xi >= 0.0 {
            xi.sqrt() / norm_safe
        } else {
            0.0
        };
    }
}

fn predict_squareroot(x: &[f64], x_min: f64, x_max: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    predict_squareroot_inplace(x, x_min, x_max, &mut out);
    out
}

fn predict_linear_inplace(x: &[f64], x_min: f64, x_max: f64, s_tri: f64, out: &mut [f64]) {
    let s_uni = 1.0 - s_tri;
    let width = x_max - x_min;
    let a = 2.0 * s_tri / (width * width);
    let b = s_uni / width - a * x_min;
    for i in 0..x.len() {
        let xi = x[i];
        out[i] = if xi >= x_min && xi <= x_max { (a * xi + b).max(0.0) } else { 0.0 };
    }
}

fn predict_linear(x: &[f64], x_min: f64, x_max: f64, s_tri: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    predict_linear_inplace(x, x_min, x_max, s_tri, &mut out);
    out
}

/// MLE for LinearModel's s_tri parameter.
/// Finds root of d/ds_tri [LL] = 0 via brentq.
/// Returns updated s_tri in [0, 1-1e-5].
pub fn mle_linear(x: &[f64], w: &[f64], x_min: f64, x_max: f64) -> f64 {
    let width = x_max - x_min;
    let a_prime = 2.0 / (width * width);
    let b_prime = -1.0 / width - a_prime * x_min;
    let eps = 1e-5_f64;

    let score = |s_tri: f64| -> f64 {
        let s_uni = 1.0 - s_tri;
        let a = 2.0 * s_tri / (width * width);
        let b = s_uni / width - a * x_min;
        x.iter().zip(w.iter())
            .filter(|(&xi, _)| xi >= x_min && xi <= x_max)
            .map(|(&xi, &wi)| wi * (a_prime * xi + b_prime) / (a * xi + b + eps))
            .sum::<f64>()
    };

    let s0 = score(0.0);
    let s1 = score(1.0 - eps);

    if s0 * s1 <= 0.0 {
        let mut conv = SimpleConvergency { eps: 1e-8, max_iter: 100 };
        if let Ok(v) = find_root_brent(0.0, 1.0 - eps, &score, &mut conv) {
            return v.clamp(0.0, 1.0 - eps);
        }
    }
    if s0 >= s1 { 0.0 } else { 1.0 - eps }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64).collect()
    }

    fn integrate_trapezoid(y: &[f64], x: &[f64]) -> f64 {
        x.windows(2).zip(y.windows(2))
            .map(|(xs, ys)| 0.5 * (ys[0] + ys[1]) * (xs[1] - xs[0]))
            .sum()
    }

    #[test]
    fn test_uniform_normalizes() {
        let x = linspace(0.0, 100.0, 1000);
        let p = predict_uniform(&x, 0.0, 100.0);
        let area = integrate_trapezoid(&p, &x);
        assert!((area - 1.0).abs() < 1e-3, "area = {}", area);
    }

    #[test]
    fn test_linear_normalizes() {
        let x = linspace(0.0, 100.0, 1000);
        let p = predict_linear(&x, 0.0, 100.0, 0.4);
        let area = integrate_trapezoid(&p, &x);
        assert!((area - 1.0).abs() < 1e-3, "area = {}", area);
    }
}
