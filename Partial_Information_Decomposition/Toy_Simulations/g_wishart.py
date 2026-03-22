import numpy as np
from scipy.special import digamma
from g_wishart_bias_corr import logdet_wishart_bias, safe_logdet, inv_sqrtm_spd, sqrtm_spd, make_random_m7_cov, build_m7_whitened_from_sample_cov
def logdet_wishart_bias(df: int, d: int) -> float:
    """
    Exact finite-sample bias for log|S| when S is the unbiased sample covariance.
    """
    if df <= d - 1:
        raise ValueError(f"Need df > d-1. Got df={df}, d={d}.")
    return np.sum([digamma((df - i + 1) / 2.0) for i in range(1,d+1)]) + d * np.log(2.0 / df)

def logdet_m7_structural_bias(df: int, n0: int, n1: int, n2: int) -> float:
    """
    Structural bias correction for the M7 model based on the Normalizing Term
    ratio of Cliques and Separators (Kay & Ince 2018).
    
    Bias_M7 = Bias(X0, Y) + Bias(X1, Y) - Bias(Y)
    """
    bias_c0 = logdet_wishart_bias(df, n0 + n2)
    bias_c1 = logdet_wishart_bias(df, n1 + n2)
    bias_sep = logdet_wishart_bias(df, n2)
    return bias_c0 + bias_c1 - bias_sep

# ... (Include your existing safe_logdet, inv_sqrtm_spd, sqrtm_spd, make_random_m7_cov, and build_m7_whitened_from_sample_cov here) ...

def simulate_m7_bias_comparison(
    n_samples: int,
    n0: int,
    n1: int,
    n2: int,
    n_trials: int = 1000,
    q_scale: float = 0.25,
    r_scale: float = 0.25,
    seed: int | None = None,
):
    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")

    d = n0 + n1 + n2
    df = n_samples - 1
    rng = np.random.default_rng(seed)

    # True population covariance satisfying M7
    true_cov = make_random_m7_cov(n0, n1, n2, q_scale, r_scale, seed=rng.integers(0, 2**32 - 1))
    true_logdet_full = safe_logdet(true_cov)

    # Whitened true covariance (Target for M7 Whitened Estimator)
    # Based on paper notation where marginals are forced to I
    i0, i1, i2 = slice(0, n0), slice(n0, n0 + n1), slice(n0 + n1, d)
    W0_t, W1_t, W2_t = inv_sqrtm_spd(true_cov[i0,i0]), inv_sqrtm_spd(true_cov[i1,i1]), inv_sqrtm_spd(true_cov[i2,i2])
    Q_t, R_t = W0_t @ true_cov[i0,i2] @ W2_t, W1_t @ true_cov[i1,i2] @ W2_t
    true_cov_m7_white = np.block([[np.eye(n0), Q_t@R_t.T, Q_t], [R_t@Q_t.T, np.eye(n1), R_t], [Q_t.T, R_t.T, np.eye(n2)]])
    true_logdet_m7_white = safe_logdet(true_cov_m7_white)

    # CALCULATE BIAS TERMS
    bias_wishart_full = logdet_wishart_bias(df, d)
    bias_m7_structural = logdet_m7_structural_bias(df, n0, n1, n2)

    logdets_full, logdets_m7_white, logdets_m7_orig = [], [], []

    for _ in range(n_trials):
        data = rng.multivariate_normal(mean=np.zeros(d), cov=true_cov, size=n_samples)
        S = np.cov(data, rowvar=False, bias=False)
        try:
            ld_full = safe_logdet(S)
            m7w, m7o, _, _ = build_m7_whitened_from_sample_cov(S, n0, n1, n2)
            logdets_full.append(ld_full)
            logdets_m7_white.append(safe_logdet(m7w))
            logdets_m7_orig.append(safe_logdet(m7o))
        except np.linalg.LinAlgError:
            continue

    # Averages and Empirical Biases
    avg_full, avg_m7w, avg_m7o = np.mean(logdets_full), np.mean(logdets_m7_white), np.mean(logdets_m7_orig)
    
    return {
        "settings": {"n_samples": n_samples, "df": df, "dims": {"n0": n0, "n1": n1, "n2": n2, "d": d}},
        "true": {"full": true_logdet_full, "m7_white": true_logdet_m7_white},
        "bias_terms": {"full_wishart": bias_wishart_full, "m7_structural": bias_m7_structural},
        "results": {
            "full": {"avg": avg_full, "emp_bias": avg_full - true_logdet_full, "corrected_bias": (avg_full - bias_wishart_full) - true_logdet_full},
            "m7_white": {"avg": avg_m7w, "emp_bias": avg_m7w - true_logdet_m7_white, "corrected_bias": (avg_m7w - bias_m7_structural) - true_logdet_m7_white},
            "m7_orig": {"avg": avg_m7o, "emp_bias": avg_m7o - true_logdet_full, "corrected_bias": (avg_m7o - bias_m7_structural) - true_logdet_full}
        }
    }

def print_summary(res):
    s, t, b, r = res["settings"], res["true"], res["bias_terms"], res["results"]
    print(f"=== M7 Structural Bias Verification (n={s['n_samples']}, p={s['dims']['d']}) ===")
    print(f"M8 (Full) Bias Term: {b['full_wishart']:.6f} | M7 Structural Bias Term: {b['m7_structural']:.6f}")
    print("-" * 72)
    print(f"1) Full Sample S: Error with M8 Correction: {r['full']['corrected_bias']:.6f}")
    print(f"2) M7 Whitened:   Error with M7 Correction: {r['m7_white']['corrected_bias']:.6f}")
    print(f"3) M7 Original:   Error with M7 Correction: {r['m7_orig']['corrected_bias']:.6f}")
    print("-" * 72)

if __name__ == "__main__":
    res = simulate_m7_bias_comparison(n_samples=500, n0=100, n1=100, n2=100, n_trials=2000, seed=12345)
    print_summary(res)