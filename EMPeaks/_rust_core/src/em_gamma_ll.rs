// Shared E-step (gamma) and log-likelihood computation for all EM models.
//
// predict は呼び出し側がモデル固有関数で事前計算して渡す。
// これにより E-step と LL の両方で同一 predict 結果を再利用できる（predict cache）。

const EPSILON_PREDICT: f64 = 1e-20;
const EPSILON_LOG: f64 = 1e-200;

/// E-step（gamma）と対数尤度を1パスで計算。
///
/// # 引数
/// - `intensity`: 観測強度 [n]
/// - `predictions`: モデル固有の predict 結果 [k][n]（呼び出し側が事前計算）
/// - `pi`: 混合比 [k]
///
/// # 返り値
/// `(gamma[k][n], LL)`
/// E-step（gamma）と対数尤度を1パスで計算し、バッファを再利用する。
///
/// # 引数
/// - `intensity`: 観測強度 [n]
/// - `predictions`: モデル固有の predict 結果 [k][n]（呼び出し側が事前計算）
/// - `pi`: 混合比 [k]
/// - `mixture`: 計算用バッファ [n]
/// - `gamma`: 結果格納バッファ [k][n]
///
/// # 返り値
/// `LL` (f64)
pub fn compute_gamma_and_ll_inplace(
    intensity: &[f64],
    predictions: &[Vec<f64>],
    pi: &[f64],
    mixture: &mut [f64],
    gamma: &mut [Vec<f64>],
) -> f64 {
    let n = intensity.len();
    let k_all = pi.len();

    // mixture[n] = Σk pi[k] * pred[k][n]
    for i in 0..n {
        mixture[i] = 0.0;
        for k in 0..k_all {
            mixture[i] += pi[k] * predictions[k][i];
        }
    }

    // gamma[k][n] = pi[k] * pred[k][n] / mixture[n]
    for k in 0..k_all {
        let pik = pi[k];
        let pred_k = &predictions[k];
        let gamma_k = &mut gamma[k];
        for i in 0..n {
            gamma_k[i] = pik * pred_k[i] / (mixture[i] + EPSILON_PREDICT);
        }
    }

    // LL = Σn intensity[n] * log(mixture[n])
    let mut ll = 0.0;
    for i in 0..n {
        ll += intensity[i] * (mixture[i] + EPSILON_LOG).ln();
    }

    ll
}

pub fn compute_gamma_and_ll(
    intensity: &[f64],
    predictions: &[Vec<f64>],
    pi: &[f64],
) -> (Vec<Vec<f64>>, f64) {
    let n = intensity.len();
    let k_all = pi.len();
    let mut mixture = vec![0.0; n];
    let mut gamma = vec![vec![0.0; n]; k_all];
    let ll = compute_gamma_and_ll_inplace(intensity, predictions, pi, &mut mixture, &mut gamma);
    (gamma, ll)
}

/// pi の M-step（全モデル共通）
///
/// Dirichlet 正則化付き混合比更新。
pub fn update_pi(
    pi: &mut [f64],
    intensity: &[f64],
    gamma: &[Vec<f64>],
    dirichlet_alpha: &[f64],
) {
    let k_all = pi.len();
    let mut n_k: Vec<f64> = (0..k_all)
        .map(|k| {
            intensity
                .iter()
                .zip(gamma[k].iter())
                .map(|(&i, &g)| i * g)
                .sum::<f64>()
                + dirichlet_alpha[k]
                - 1.0
        })
        .collect();
    let n_k_sum: f64 = n_k.iter().sum();
    n_k.iter_mut().for_each(|v| *v = (*v / n_k_sum).max(0.0));
    let pi_sum: f64 = n_k.iter().sum();
    pi.iter_mut()
        .zip(n_k.iter())
        .for_each(|(p, &v)| *p = v / pi_sum);
}
