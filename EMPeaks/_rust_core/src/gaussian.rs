use ndarray::ArrayView1;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const SQRT2: f64 = std::f64::consts::SQRT_2;

pub fn predict(x: ArrayView1<f64>, mu: f64, sigma: f64) -> Vec<f64> {
    let z = (TWO_PI * sigma * sigma).sqrt();
    let inv = 1.0 / (SQRT2 * sigma);
    x.iter().map(|&xi| {
        let t = (xi - mu) * inv;
        (-t * t).exp() / z
    }).collect()
}

/// Weighted MLE: returns (mu, sigma).
/// Mirrors Python: mu = sum(w*x)/sum(w), sigma = sqrt(sum(w*(x-mu)^2)/sum(w))
pub fn mle(x: ArrayView1<f64>, w: ArrayView1<f64>) -> (f64, f64) {
    let eps = 1e-100;
    let sum_w: f64 = w.iter().sum::<f64>() + eps;
    let mu: f64 = x.iter().zip(w.iter()).map(|(&xi, &wi)| wi * xi).sum::<f64>() / sum_w;
    let sigma2: f64 = x.iter().zip(w.iter())
        .map(|(&xi, &wi)| wi * (xi - mu) * (xi - mu))
        .sum::<f64>() / sum_w;
    let sigma = if sigma2 > 0.0 { sigma2.sqrt() } else { 1e-5_f64 };
    (mu, sigma.max(1e-5))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_sums_to_one() {
        let x = ndarray::Array1::linspace(-10.0, 10.0, 1000);
        let dx = x[1] - x[0];
        let p = predict(x.view(), 0.0, 1.0);
        let area: f64 = p.iter().sum::<f64>() * dx;
        assert!((area - 1.0).abs() < 1e-4, "area = {}", area);
    }

    #[test]
    fn test_mle_recovers_params() {
        let x = ndarray::Array1::linspace(-5.0, 5.0, 500);
        let mu_true = 1.0_f64;
        let sigma_true = 0.8_f64;
        let w = ndarray::Array1::from_vec(predict(x.view(), mu_true, sigma_true));
        let (mu_est, sigma_est) = mle(x.view(), w.view());
        assert!((mu_est - mu_true).abs() < 1e-3, "mu_est = {}", mu_est);
        assert!((sigma_est - sigma_true).abs() < 1e-3, "sigma_est = {}", sigma_est);
    }
}
