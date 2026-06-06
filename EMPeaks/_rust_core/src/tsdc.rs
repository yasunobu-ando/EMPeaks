use xsf::{expn, expi};
use roots::{find_root_brent, SimpleConvergency};
use rayon::prelude::*;

const KB: f64 = 8.61733034e-5;
const INV_KB: f64 = 1.0 / KB;

fn trapezoid(y: &[f64], x: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(xs, ys)| 0.5 * (ys[0] + ys[1]) * (xs[1] - xs[0]))
        .sum()
}

pub fn get_tau0(tp: f64, ea: f64, beta: f64) -> f64 {
    ((KB * tp * tp) / (beta * ea)) * (-ea / (KB * tp)).exp()
}

pub fn get_tp(ea: f64, tau0: f64, beta: f64) -> Result<f64, String> {
    let f = |tp: f64| -> f64 {
        ((KB * tp * tp) / (beta * ea)).ln() - ea / (KB * tp) - tau0.ln()
    };
    let mut convergency = SimpleConvergency { eps: 1e-8, max_iter: 100 };
    match find_root_brent(1.0, 2000.0, &f, &mut convergency) {
        Ok(v) => Ok(v),
        Err(e) => Err(format!("brentq failed: {:?}", e)),
    }
}

pub fn predict_inplace(t: &[f64], ea: f64, tau0: f64, beta: f64, out: &mut [f64]) {
    out.par_iter_mut().zip(t.par_iter()).for_each(|(out_i, &ti)| {
        let arg = ea * INV_KB / ti;
        let e2 = expn(2, arg);
        let val = (-ea / KB / ti - 1.0 / (beta * tau0) * ti * e2).exp();
        *out_i = val / (beta * tau0);
    });

    let z = trapezoid(out, t);
    let z_safe = z + 1.0e-100;
    out.par_iter_mut().for_each(|p| *p /= z_safe);
}

// Rayon parallel: predict TSDC PDF on T array.
// xsf::expn is significantly slower per-element than scipy's vectorized C,
// so parallelising across T points recovers that gap.
pub fn predict_tsdc(t: &[f64], ea: f64, tau0: f64, beta: f64) -> Vec<f64> {
    let mut out = vec![0.0; t.len()];
    predict_inplace(t, ea, tau0, beta, &mut out);
    out
}

fn ll_tsdc(t: &[f64], intensity: &[f64], ea: f64, tau0: f64, beta: f64) -> f64 {
    let pred = predict_tsdc(t, ea, tau0, beta);
    intensity
        .iter()
        .zip(pred.iter())
        .map(|(&ii, &p)| ii * (p + 1e-200).ln())
        .sum()
}

// Rayon parallel: Σ_i y_i * t_i * E_2(ea/(kB*ti))
fn f_sum(ea: f64, t: &[f64], y: &[f64]) -> f64 {
    t.par_iter().zip(y.par_iter()).map(|(&ti, &yi)| {
        yi * ti * expn(2, ea * INV_KB / ti)
    }).sum()
}

// Rayon parallel: root function for Ea MLE
fn f_g_diff(ea: f64, t: &[f64], y: &[f64], beta: f64) -> f64 {
    let eps = 1e-10;
    let a: f64 = t.par_iter().zip(y.par_iter()).map(|(&ti, &yi)| yi / (KB * ti)).sum::<f64>() + eps;
    let b: f64 = y.par_iter().sum::<f64>() + eps;
    let g_sum: f64 = t.par_iter().zip(y.par_iter()).map(|(&ti, &yi)| {
        yi * INV_KB * (-expi(-ea * INV_KB / ti))
    }).sum();

    let fsum = f_sum(ea, t, y);
    fsum.log10() - (beta * b).log10() - g_sum.log10() + (beta * a).log10()
}

pub fn mle_tsdc_find_root(
    t: &[f64],
    intensity: &[f64],
    ea: &mut f64,
    tau0: &mut f64,
    tp: &mut f64,
    ea_min: f64,
    ea_max: f64,
    beta: f64,
) -> Result<(), String> {
    let f = |e: f64| -> f64 {
        f_g_diff(e, t, intensity, beta)
    };
    
    let mut convergency = SimpleConvergency { eps: 1e-8, max_iter: 100 };
    match find_root_brent(ea_min, ea_max, &f, &mut convergency) {
        Ok(root_ea) => {
            *ea = root_ea;
            *tau0 = f_sum(root_ea, t, intensity) / (beta * intensity.iter().sum::<f64>());
            *tp = get_tp(root_ea, *tau0, beta)?;
            Ok(())
        },
        Err(e) => Err(format!("brentq failed: {:?}", e)),
    }
}

pub fn mle_tsdc_lbfgsb(
    t: &[f64],
    intensity: &[f64],
    ea: &mut f64,
    tau0: &mut f64,
    tp: &mut f64,
    ea_min: f64,
    ea_max: f64,
    t_min: f64,
    t_max: f64,
    beta: f64,
) -> Result<(), String> {
    let sum_w: f64 = intensity.iter().sum();
    if sum_w < 1e-300 {
        return Ok(());
    }

    let d1: f64 = sum_w;
    let d2: f64 = t.iter().zip(intensity.iter()).map(|(&ti, &yi)| yi / ti * INV_KB).sum();

    let bounds = vec![
        (ea_min, ea_max),
        (t_min, t_max),
    ];
    let init = vec![*ea, *tp];

    let t_s = t.to_vec();
    let w_s = intensity.to_vec();

    if let Ok(state) = lbfgsb::lbfgsb(init, &bounds, |p, g| {
        let ea_c = p[0];
        let tp_c = p[1];
        let tau0_c = get_tau0(tp_c, ea_c, beta);
        
        let beta_tau = beta * tau0_c;
        let ktp = KB * tp_c;
        
        let d3: f64 = t_s.par_iter().zip(w_s.par_iter()).map(|(&ti, &yi)| yi * ti * expn(2, ea_c * INV_KB / ti)).sum();
        let d3_p: f64 = t_s.par_iter().zip(w_s.par_iter()).map(|(&ti, &yi)| -yi * INV_KB * expn(1, ea_c * INV_KB / ti)).sum();
        
        let ll_e = (d1 - d3 / beta_tau) * (1.0 / ea_c + 1.0 / ktp) - d2 - d3_p / beta_tau;
        let ll_t = (-d1 + d3 / beta_tau) * (2.0 / tp_c + ea_c * INV_KB / (tp_c * tp_c));
        
        g[0] = -ll_e;
        g[1] = -ll_t;

        let ll = ll_tsdc(&t_s, &w_s, ea_c, tau0_c, beta);
        Ok(-ll)
    }) {
        *ea = state.x()[0].clamp(ea_min, ea_max);
        *tp = state.x()[1].clamp(t_min, t_max);
        *tau0 = get_tau0(*tp, *ea, beta);
        Ok(())
    } else {
        Err("L-BFGS-B failed".to_string())
    }
}
