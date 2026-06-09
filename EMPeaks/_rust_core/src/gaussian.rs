const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const SQRT2: f64 = std::f64::consts::SQRT_2;

pub fn predict_inplace(x: &[f64], mu: f64, sigma: f64, out: &mut [f64]) {
    let z = (TWO_PI * sigma * sigma).sqrt();
    let inv = 1.0 / (SQRT2 * sigma);
    for i in 0..x.len() {
        let t = (x[i] - mu) * inv;
        out[i] = (-t * t).exp() / z;
    }
}

pub fn predict(x: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    predict_inplace(x, mu, sigma, &mut out);
    out
}

pub fn mle(
    x: &[f64], w: &[f64],
    mu_curr: f64, sigma_curr: f64,
    fix_mu: bool, fix_sigma: bool,
) -> (f64, f64) {
    if fix_mu && fix_sigma { return (mu_curr, sigma_curr); }
    if w.iter().sum::<f64>() < 1e-300 { return (mu_curr, sigma_curr); }

    let eps = 1e-100;
    let sum_w: f64 = w.iter().sum::<f64>() + eps;

    let mu = if fix_mu {
        mu_curr
    } else {
        x.iter().zip(w.iter()).map(|(&xi, &wi)| wi * xi).sum::<f64>() / sum_w
    };

    let sigma = if fix_sigma {
        sigma_curr
    } else {
        let sigma2: f64 = x.iter().zip(w.iter())
            .map(|(&xi, &wi)| wi * (xi - mu) * (xi - mu))
            .sum::<f64>() / sum_w;
        (if sigma2 > 0.0 { sigma2.sqrt() } else { 1e-5_f64 }).max(1e-5)
    };

    (mu, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn test_predict_sums_to_one() {
        let x = linspace(-10.0, 10.0, 1000);
        let dx = x[1] - x[0];
        let p = predict(&x, 0.0, 1.0);
        let area: f64 = p.iter().sum::<f64>() * dx;
        assert!((area - 1.0).abs() < 1e-4, "area = {}", area);
    }

    #[test]
    fn test_mle_recovers_params() {
        let x = linspace(-5.0, 5.0, 500);
        let mu_true = 1.0_f64;
        let sigma_true = 0.8_f64;
        let w = predict(&x, mu_true, sigma_true);
        let (mu_est, sigma_est) = mle(&x, &w, 0.0, 1.0, false, false);
        assert!((mu_est - mu_true).abs() < 1e-3, "mu_est = {}", mu_est);
        assert!((sigma_est - sigma_true).abs() < 1e-3, "sigma_est = {}", sigma_est);
    }
}
