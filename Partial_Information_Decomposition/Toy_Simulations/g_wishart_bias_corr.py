import numpy as np
from scipy.special import digamma
import sys
import os
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *

def logdet_wishart_bias(df: int, d: int) -> float:
    """
    Exact finite-sample bias for log|S| when S is the unbiased sample covariance
    from Gaussian data and (df) * S ~ Wishart_d(Sigma, df).

    Returns
    -------
    bias : float
        E[log|S|] - log|Sigma|
    """
    if df <= d - 1:
        raise ValueError(f"Need df > d-1. Got df={df}, d={d}.")
    return np.sum([digamma((df - i + 1) / 2.0) for i in range(1,d+1)]) + d * np.log(2.0 / df)


def digamma_func(df: int, d: int) -> float:
    """
    Helper function to compute the sum of digamma terms in the Wishart bias formula.
    """
    return np.sum([digamma((df - i + 1) / 2.0) for i in range(1,d+1)])

def safe_logdet(A: np.ndarray) -> float:
    """
    Compute log determinant and raise if matrix is not positive definite.
    """
    sign, ld = np.linalg.slogdet(A)
    if sign <= 0:
        eigmin = np.min(np.linalg.eigvalsh(0.5 * (A + A.T)))
        raise np.linalg.LinAlgError(
            f"Matrix not positive definite in logdet. sign={sign}, min_eig={eigmin:.3e}"
        )
    return ld






def make_random_true_cov(
    n0: int,
    n1: int,
    n2: int,
    q_scale: float = 0.25,
    r_scale: float = 0.25,
    p_scale: float = 0.25,
    seed: int | None = None,
) -> np.ndarray:
    """
    Construct a generic positive-definite Gaussian M7 covariance.

    Block order is [X0, X1, Y].

    M7 population structure:
        Sigma_01 = Sigma_02 @ Sigma_22^{-1} @ Sigma_21
    and here Sigma_22 = I, so:
        P = Q @ R.T
    """
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((n0, n2))
    B = rng.standard_normal((n1, n2))
    C =  rng.standard_normal((n0, n1))

    A_norm = np.linalg.norm(A, ord=2)
    B_norm = np.linalg.norm(B, ord=2)
    C_norm = np.linalg.norm(C, ord=2)   
    if A_norm == 0 or B_norm == 0 or C_norm == 0:
        raise RuntimeError("Unexpected zero spectral norm in random construction.")

    Q = q_scale * A / A_norm
    R = r_scale * B / B_norm
    P = p_scale * C / C_norm

    true_cov = np.block([
        [np.eye(n0), P,          Q],
        [P.T,        np.eye(n1), R],
        [Q.T,        R.T,        np.eye(n2)]
    ])

    eigvals = np.linalg.eigvalsh(true_cov)
    if np.min(eigvals) <= 1e-10:
        raise ValueError(
            f"Constructed covariance not sufficiently PD. min eig={np.min(eigvals):.3e}"
        )

    # # Check precision-matrix M7 condition: K_{X0,X1} = 0
    # K = np.linalg.inv(true_cov)
    # K01 = K[:n0, n0:n0+n1]
    # if not np.allclose(K01, 0, atol=1e-10):
    #     raise ValueError("Constructed covariance does not satisfy the M7 precision condition.")

    return true_cov


def sample_data_from_cov(true_cov: np.ndarray, n_samples: int, seed: int | None = None) -> np.ndarray:
    """
    Sample multivariate Gaussian data from the specified covariance.
    and return it's covariance matrix. This is a helper function for the M7 bias simulation.
    """
    rng = np.random.default_rng(seed)
    d = true_cov.shape[0]
    mean = np.zeros(d)
    data =  rng.multivariate_normal(mean, true_cov, size=n_samples)
    return np.cov(data, rowvar=False, bias=False) # Unbiased estimator with N-1 in the denominator


def simulate_m7_m8_bias_comparison(
    n_samples: int,
    n0: int,
    n1: int,
    n2: int,
    n_trials: int = 1000,
    q_scale: float = 0.25,
    r_scale: float = 0.25,
    p_scale: float = 0.25,
    seed: int | None = None,
):
    """
    Compare logdet bias for:
      1) Full sample covariance S                     (exact Wishart case)
      2) Whitened paper-style M7 estimator           (not Wishart)
      3) Original-scale paper-style M7 estimator     (not Wishart)

    Returns
    -------
    dict with empirical and corrected bias summaries.
    """
    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")

    d = n0 + n1 + n2
    df = n_samples - 1

    if df <= d - 1:
        raise ValueError(
            f"Need df > d-1 for stable logdet expectation. Got n_samples={n_samples}, df={df}, d={d}."
        )

    rng = np.random.default_rng(seed)

    # True population covariance satisfying M7
    true_cov = make_random_true_cov(
        n0=n0,
        n1=n1,
        n2=n2,
        q_scale=q_scale,
        r_scale=r_scale,
        p_scale=p_scale,

        seed=rng.integers(0, 2**32 - 1),
    )

    true_cov_torch = torch.from_numpy(true_cov).to(torch.float64)
    true_cov_dict = create_cov_matrix(Sigma=true_cov_torch, dims=[n0, n1, n2])


    sigma00 = true_cov_dict['cov_x0']
    sigma11 = true_cov_dict['cov_x1']
    sigma22 = true_cov_dict['cov_x2']
    sigma01 = true_cov_dict['cross_x0_x1']
    sigma02 = true_cov_dict['cross_x0_x2']
    sigma12 = true_cov_dict['cross_x1_x2']
    
    P = whiten_block(sigma00, sigma01, sigma11).numpy()
    Q = whiten_block(sigma00, sigma02, sigma22).numpy()
    R = whiten_block(sigma11, sigma12, sigma22).numpy()


    true_cov_white = np.block([
        [np.eye(n0),         P, Q],
        [P.T,  np.eye(n1),        R],
        [Q.T,           R.T,          np.eye(n2)],
    ])

    

    P_m7 = Q @ R.T
    m7_true_cov = np.block([
        [np.eye(n0),        P_m7, Q],
        [P_m7.T,  np.eye(n1),        R],
        [Q.T,           R.T,          np.eye(n2)],
    ])


    m8_true_logdet_full = safe_logdet(true_cov)
    m8_true_logdet_white = safe_logdet(true_cov_white)
    m7_true_logdet_full= safe_logdet(m7_true_cov)

    #Bias correction for Wishart case:
    wishart_bias_correction_full = logdet_wishart_bias(df=df, d=d)

    # 1. Calculate Marginal Biases
    bias_x0 = logdet_wishart_bias(df=df, d=n0)
    bias_x1 = logdet_wishart_bias(df=df, d=n1)
    bias_y  = logdet_wishart_bias(df=df, d=n2)

    # Bias correction for Chrodal Graphs: 
    bias_c0 = logdet_wishart_bias(df=df, d=n0+n2)
    bias_c1 = logdet_wishart_bias(df=df, d=n1+n2)
    bias_sep = logdet_wishart_bias(df=df, d=n2)
    bias_m7_structural = bias_c0 + bias_c1 - bias_sep
    bias_m7_naive = wishart_bias_correction_full - (bias_x0 + bias_x1 + bias_y)
    bias_m7_structural = bias_m7_structural - (bias_x0 + bias_x1 + bias_y)


    logdets_m8 = []
    logdets_m7_naive = []

    fail_m8 = 0
    fail_m7_naive = 0

    for _ in range(n_trials):


        #Sample data and get sample covariance
        S = sample_data_from_cov(true_cov=true_cov, n_samples=n_samples, seed=rng.integers(0, 2**32 - 1))
        S_torch = torch.from_numpy(S).to(torch.float64)
        S_dict = create_cov_matrix(Sigma=S_torch, dims=[n0, n1, n2])

        #Calculate M8 model logdets
        m8_logdet_raw = safe_logdet(S)


        #Calculate M7 model logdets
        Q_m7 = whiten_block(S_dict['cov_x0'], S_dict['cross_x0_x2'], S_dict['cov_x2']).numpy()
        R_m7 = whiten_block(S_dict['cov_x1'], S_dict['cross_x1_x2'], S_dict['cov_x2']).numpy()
        P = Q_m7 @ R_m7.T
        m7_white = np.block([
            [np.eye(n0),         P, Q_m7],
            [P.T,  np.eye(n1),        R_m7],
            [Q_m7.T,           R_m7.T,          np.eye(n2)],
        ])

        log_det_m7_white_raw = safe_logdet(m7_white)

            
        logdets_m8.append(m8_logdet_raw)
        logdets_m7_naive.append(log_det_m7_white_raw)



    logdets_m8 = np.asarray(logdets_m8)
    logdets_m7_naive = np.asarray(logdets_m7_naive)
    logdets_m7_structural = np.asarray(logdets_m7_naive)

    avg_m8 = np.mean(logdets_m8)
    avg_m7_naive = np.mean(logdets_m7_naive)

    emp_bias_m8 = avg_m8 - m8_true_logdet_full
    emp_bias_m8_white = avg_m8 - m8_true_logdet_white
    emp_bias_m7_naive = avg_m7_naive - m7_true_logdet_full

    avg_m8_corrected = avg_m8- wishart_bias_correction_full
    avg_m7_naive_corrected = avg_m7_naive - bias_m7_naive
    avg_m7_structural_corrected = avg_m7_naive - bias_m7_structural

    corr_bias_m8 = avg_m8_corrected - m8_true_logdet_full
    corr_bias_m8_white = avg_m8_corrected - m8_true_logdet_white   
    corr_bias_m7_naive = avg_m7_naive_corrected - m7_true_logdet_full
    corr_bias_m7_structural = avg_m7_structural_corrected - m7_true_logdet_full

    return {
        "settings": {
            "n_samples": n_samples,
            "df": df,
            "n_trials": n_trials,
            "dims": {"n0": n0, "n1": n1, "n2": n2, "d": d},
            "q_scale": q_scale,
            "r_scale": r_scale,
            "p_scale": p_scale,
            "seed": seed,
        },
        "True_M8": {
            "True_log_det": m8_true_logdet_full,
            "true_M8_after_whitening": m8_true_logdet_white,
            "Error_TrueM8_-_WhitenM8 ": m8_true_logdet_full - m8_true_logdet_white,
            "true_logdet_m7_white": m7_true_logdet_full,
        },
        "wishart_bias_terms": {
            "m8_bias": wishart_bias_correction_full,
            "m7_whitened_bias": bias_m7_naive,
            "m7_structure_bias": bias_m7_structural,
        },
        "Sample_M8": {
            "Raw_mean_logdet": avg_m8,
            "Empirical_bias_true_cov": emp_bias_m8,
            "Empirical_bias_whitened_true_cov": emp_bias_m8_white,
            "avg_corrected_logdet": avg_m8_corrected,
            "Corrected_bias": corr_bias_m8,
            "Corrected_bias_whitened": corr_bias_m8_white,
            "std_logdet": np.std(logdets_m8, ddof=1),
        },
        "M7_Naive_(Normal Wishart)": {
            "Raw_mean_logdet": avg_m7_naive,
            "Empirical_bias_true_cov": emp_bias_m7_naive,
            "Empirical_bias_whitened_true_cov": emp_bias_m7_naive,
            "avg_corrected_logdet": avg_m7_naive_corrected,
            "Corrected_bias": corr_bias_m7_naive,
            "std_logdet": np.std(logdets_m7_naive, ddof=1),
        },
        "M7_Structural_(Chrodal Graph)": {
            "avg_corrected_logdet": avg_m7_structural_corrected,
            "Corrected_bias": corr_bias_m7_structural,
            "std_logdet": np.std(logdets_m7_structural, ddof=1),
        },

    }


def print_m7_bias_summary(results: dict) -> None:
    s = results["settings"]
    true_logdet = results["True_M8"]
    m8_sample = results["Sample_M8"]
    m7_naive = results["M7_Naive_(Normal Wishart)"]
    m7_struc = results["M7_Structural_(Chrodal Graph)"]

    print("=" * 72)
    print("M7 Log-Det Bias Comparison")
    print("=" * 72)
    print(
        f"n_samples={s['n_samples']}, df={s['df']}, "
        f"(n0,n1,n2)=({s['dims']['n0']},{s['dims']['n1']},{s['dims']['n2']}), "
        f"d={s['dims']['d']}, n_trials={s['n_trials']}"
    )
    print(f"q_scale={s['q_scale']}, r_scale={s['r_scale']}, p_scale={s['p_scale']}, seed={s['seed']}")
    print()
    print("Bias Correction Terms:")
    print(f"  M8 Bias                     = {results['wishart_bias_terms']['m8_bias']:.6f}")
    print(f"  M7 Whitened Bias            = {results['wishart_bias_terms']['m7_whitened_bias']:.6f}")
    print(f"  M7 Structural Bias          = {results['wishart_bias_terms']['m7_structure_bias']:.6f}")
    print()

    print("True M8 Log Det_erminant values:")
    print(f"  True mean logdet            = {true_logdet['True_log_det']:.8f}")
    print(f"  True mean logdet after whitening = {true_logdet['true_M8_after_whitening']:.8f}")
    print(f"  Raw mean logdet            = {m8_sample['Raw_mean_logdet']:.8f}")
    print(f"  Empirical bias (true cov)  = {m8_sample['Empirical_bias_true_cov']:.6f}")
    print(f"  Empirical bias (whitened)  = {m8_sample['Empirical_bias_whitened_true_cov']:.6f}")
    print()
    print("M8 Sample Log Determinant results:")
    print(f"  Correct M8 log det         = {m8_sample['avg_corrected_logdet']:.8f}")
    print(f"  After Correction bias (true cov)  = {m8_sample['Corrected_bias']:.6f}")
    print(f"  After Correction bias (whitened)  = {m8_sample['Corrected_bias_whitened']:.6f}")
    print(f"  Standard deviation of logdet  = {m8_sample['std_logdet']:.6f}")
    print("=" * 72)

    print("M7 True Log Determinant values:")
    print(f"  True mean logdet            = {true_logdet['true_logdet_m7_white']:.8f}")
    print(f"  Raw mean logdet            = {m7_naive['Raw_mean_logdet']:.8f}")
    print(f"  Empirical bias            = {m7_naive['Empirical_bias_true_cov']:.6f}")
    print()

    print("M7 (Naive and Structural bias corrections) Sample Log Determinant results:")
    print(f"  Naive (Wishart) Corrected mean logdet      = {m7_naive['avg_corrected_logdet']:.8f}")
    print(f"  Naive (Wishart) After Correction bias (true cov)  = {m7_naive['Corrected_bias']:.6f}")
    print(f"  Structural Corrected mean logdet      = {m7_struc['avg_corrected_logdet']:.8f}")
    print(f"  Structural After Correction bias (true cov)  = {m7_struc['Corrected_bias']:.6f}")




if __name__ == "__main__":
    results = simulate_m7_m8_bias_comparison(
        n_samples=1000,
        n0=100,
        n1=100,
        n2=200,
        n_trials=2000,
        q_scale=0.35,
        r_scale=0.30,
        p_scale=0.25,
        seed=12345,
    )

    print_m7_bias_summary(results)