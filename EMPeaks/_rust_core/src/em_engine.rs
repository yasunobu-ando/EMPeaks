use ndarray::{Array1, Array2, ArrayView1};
use crate::gaussian;

const EPSILON_PREDICT: f64 = 1e-20;
const EPSILON_LOG: f64 = 1e-200;

fn total_predict(x: ArrayView1<f64>, mu: &[f64], sigma: &[f64], pi: &[f64]) -> Vec<f64> {
    let n = x.len();
    let k_all = pi.len();
    let mut total = vec![0.0_f64; n];
    for k in 0..k_all {
        let p_k = gaussian::predict(x, mu[k], sigma[k]);
        for i in 0..n {
            total[i] += pi[k] * p_k[i];
        }
    }
    total
}

fn log_likelihood(x: ArrayView1<f64>, intensity: ArrayView1<f64>,
                  mu: &[f64], sigma: &[f64], pi: &[f64]) -> f64 {
    let total = total_predict(x, mu, sigma, pi);
    intensity.iter().zip(total.iter())
        .map(|(&ii, &ti)| ii * (ti + EPSILON_LOG).ln())
        .sum()
}

fn e_step(x: ArrayView1<f64>, mu: &[f64], sigma: &[f64], pi: &[f64]) -> Array2<f64> {
    let n = x.len();
    let k_all = pi.len();
    let total = total_predict(x, mu, sigma, pi);

    let mut gamma = Array2::<f64>::zeros((k_all, n));
    for k in 0..k_all {
        let p_k = gaussian::predict(x, mu[k], sigma[k]);
        for i in 0..n {
            gamma[[k, i]] = pi[k] * p_k[i] / (total[i] + EPSILON_PREDICT);
        }
    }
    gamma
}

fn m_step(
    x: ArrayView1<f64>,
    intensity: ArrayView1<f64>,
    gamma: &Array2<f64>,
    dirichlet_alpha: &[f64],
    mu: &mut Vec<f64>,
    sigma: &mut Vec<f64>,
    pi: &mut Vec<f64>,
) {
    let k_all = pi.len();
    let n = x.len();

    let mut n_k = vec![0.0_f64; k_all];
    for k in 0..k_all {
        for i in 0..n {
            n_k[k] += intensity[i] * gamma[[k, i]];
        }
        n_k[k] += dirichlet_alpha[k] - 1.0;
    }

    let n_k_sum: f64 = n_k.iter().sum();
    for k in 0..k_all {
        pi[k] = (n_k[k] / n_k_sum).max(0.0);
    }
    let pi_sum: f64 = pi.iter().sum();
    pi.iter_mut().for_each(|v| *v /= pi_sum);

    // MLE for each Gaussian component
    let w_buf: Vec<f64> = vec![0.0; n];
    let _ = w_buf;
    for k in 0..k_all {
        let w: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| intensity[i] * gamma[[k, i]])
        );
        let (new_mu, new_sigma) = gaussian::mle(x, w.view());
        mu[k] = new_mu;
        sigma[k] = new_sigma;
    }
}

pub fn run_em_loop(
    x: ArrayView1<f64>,
    intensity: ArrayView1<f64>,
    mu: &mut Vec<f64>,
    sigma: &mut Vec<f64>,
    pi: &mut Vec<f64>,
    dirichlet_alpha: &[f64],
    max_iter: usize,
    r_eps: f64,
) -> (usize, Vec<f64>, Vec<f64>) {
    let mut ll_0 = log_likelihood(x, intensity, mu, sigma, pi);
    let mut ll_hist = vec![ll_0];
    let mut res_hist = vec![0.0_f64];
    let mut total_iter = max_iter;

    for it in 0..max_iter {
        let gamma = e_step(x, mu, sigma, pi);
        m_step(x, intensity, &gamma, dirichlet_alpha, mu, sigma, pi);

        let ll = log_likelihood(x, intensity, mu, sigma, pi);
        let residual = (ll - ll_0) / ll_0.abs();
        ll_hist.push(ll);
        res_hist.push(residual);

        if residual < 0.0 {
            // 対数尤度が下降 — Python 側に通知して処理させる
            total_iter = it + 1;
            break;
        }
        if residual < r_eps {
            total_iter = it + 1;
            break;
        }
        ll_0 = ll;
    }
    (total_iter, ll_hist, res_hist)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_data() -> (Array1<f64>, Array1<f64>) {
        let x = Array1::linspace(-5.0, 5.0, 200);
        let intensity: Array1<f64> = x.mapv(|xi: f64| {
            let t0 = (-(xi * xi)).exp();
            let t1 = 0.5 * (-(xi - 2.0_f64) * (xi - 2.0_f64)).exp();
            t0 + t1
        });
        let sum = intensity.sum();
        (x, intensity / sum)
    }

    #[test]
    fn test_em_converges() {
        let (x, intensity) = make_data();
        let mut mu    = vec![-1.0, 1.0];
        let mut sigma = vec![1.0, 1.0];
        let mut pi    = vec![0.5, 0.5];
        let da        = vec![1.0, 1.0];

        let (iters, ll_hist, _) = run_em_loop(
            x.view(), intensity.view(),
            &mut mu, &mut sigma, &mut pi,
            &da, 3000, 1e-9,
        );
        assert!(iters < 3000, "did not converge within 3000 iters");
        assert!(ll_hist.last().unwrap() > ll_hist.first().unwrap(),
                "LL did not increase");
    }
}
