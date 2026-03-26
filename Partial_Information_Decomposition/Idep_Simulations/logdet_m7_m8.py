import numpy as np
from scipy.special import digamma
import sys
import os
from pathlib import Path
from Simulation_utils import *
from M7_M8_models import make_random_true_cov
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *




def simulate_m7_m8_log_det(
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
      2) Whitened paper-style m7_whiten estimator           (not Wishart)
      3) Original-scale paper-style m7_whiten estimator     (not Wishart)

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

    # True population covariance satisfying m7_whiten
    m8_true_cov,m7_true_cov = make_random_true_cov(
        n0=n0,
        n1=n1,
        n2=n2,
        q_scale=q_scale,
        r_scale=r_scale,
        p_scale=p_scale,

        seed=rng.integers(0, 2**32 - 1),
    )

    true_cov_torch = torch.from_numpy(m8_true_cov).to(torch.float64)
    true_cov_dict = create_cov_matrix(Sigma=true_cov_torch, dims=[n0, n1, n2])




    #True logdet values 
    m8_true_logdet_full = safe_logdet(m8_true_cov)
    m7_true_cov_logdet = safe_logdet(m7_true_cov)


    #Bias correction for Wishart case:
    wishart_bias_correction_full = logdet_wishart_bias(df=df, d=d)

    # 1. Calculate Marginal Biases
    bias_x0 = logdet_wishart_bias(df=df, d=n0)
    bias_x1 = logdet_wishart_bias(df=df, d=n1)
    bias_y  = logdet_wishart_bias(df=df, d=n2)

    # Bias correction for Chrodal Graphs: 
    bias_02 = logdet_wishart_bias(df=df, d=n0+n2)
    bias_12 = logdet_wishart_bias(df=df, d=n1+n2)
    bias_2 = logdet_wishart_bias(df=df, d=n2)
    bias_m7_whiten_structural = bias_02 + bias_12 - bias_2

    bias_m7_structural = bias_m7_whiten_structural #This is the structural bias for the original-scale m7_whiten estimator, which has the same structural bias as the whitened version.
    bias_m7_whiten_naive = wishart_bias_correction_full - (bias_x0 + bias_x1 + bias_y)
    bias_m7_whiten_structural = bias_m7_whiten_structural - (bias_x0 + bias_x1 + bias_y)


    logdets_m8 = []
    logdets_m7_whiten_naive = []
    log_dets_m7_org_structural = []

    for i in range(n_trials):
        if (i+1) % 100 == 0:
            print(f"Trial {i+1}/{n_trials}...")

        #Sample data and get sample covariance
        S = sample_data_from_cov(true_cov=m8_true_cov, n_samples=n_samples, seed=rng.integers(0, 2**32 - 1))
        S_torch = torch.from_numpy(S).to(torch.float64)
        S_dict = create_cov_matrix(Sigma=S_torch, dims=[n0, n1, n2])




        #Calculate m7 not whiten model logdets
        
        Q_m7 = S_dict['cross_x0_x2']
        R_m7 = S_dict['cross_x1_x2']
        P_m7 = Q_m7 @ np.linalg.inv(S_dict['cov_x2']) @ R_m7.T
        m7_org = np.block([
            [S_dict['cov_x0'].numpy(), P_m7.numpy(), S_dict['cross_x0_x2'].numpy()],
            [P_m7.numpy().T, S_dict['cov_x1'].numpy(), S_dict['cross_x1_x2'].numpy()],
            [S_dict['cross_x0_x2'].numpy().T, S_dict['cross_x1_x2'].numpy().T, S_dict['cov_x2'].numpy()]
        ])


        #Calculate m7_whiten model logdets
        Q_m7_whiten = whiten_block(S_dict['cov_x0'], S_dict['cross_x0_x2'], S_dict['cov_x2']).numpy()
        R_m7_whiten = whiten_block(S_dict['cov_x1'], S_dict['cross_x1_x2'], S_dict['cov_x2']).numpy()
        P = Q_m7_whiten @ R_m7_whiten.T
        m7_whiten_white = np.block([
            [np.eye(n0),         P, Q_m7_whiten],
            [P.T,  np.eye(n1),        R_m7_whiten],
            [Q_m7_whiten.T,           R_m7_whiten.T,          np.eye(n2)],
        ])


        m8_logdet_raw = safe_logdet(S)
        m7_org_logdet_raw = safe_logdet(m7_org)
        log_det_m7_whiten_white_raw = safe_logdet(m7_whiten_white)

            
        logdets_m8.append(m8_logdet_raw)
        log_dets_m7_org_structural.append(m7_org_logdet_raw)
        logdets_m7_whiten_naive.append(log_det_m7_whiten_white_raw)



    logdets_m8 = np.asarray(logdets_m8)
    org_log_dets_m7_org_structural = np.asarray(log_dets_m7_org_structural)
    logdets_m7_whiten_naive = np.asarray(logdets_m7_whiten_naive)
    logdets_m7_whiten_structural = np.asarray(logdets_m7_whiten_naive)

    avg_m8 = np.mean(logdets_m8)
    avg_m7_org_structural = np.mean(org_log_dets_m7_org_structural)
    avg_m7_whiten_naive = np.mean(logdets_m7_whiten_naive)

    emp_bias_m8 = avg_m8 - m8_true_logdet_full

    emp_bias_m7_whiten_naive = avg_m7_whiten_naive - m7_true_cov_logdet

    avg_m8_corrected = avg_m8- wishart_bias_correction_full
    org_avg_m7_structural_corrected = avg_m7_org_structural - bias_m7_structural
    avg_m7_whiten_naive_corrected = avg_m7_whiten_naive - bias_m7_whiten_naive
    avg_m7_whiten_structural_corrected = avg_m7_whiten_naive - bias_m7_whiten_structural

    corr_bias_m8 = avg_m8_corrected - m8_true_logdet_full
    corr_bias_m7_whiten_naive = avg_m7_whiten_naive_corrected - m7_true_cov_logdet
    corr_bias_m7_whiten_structural = avg_m7_whiten_structural_corrected - m7_true_cov_logdet

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
            "true_logdet_m7_whiten_white": m7_whiten_true_logdet_full,
            "true_logdet_m7_org_structural": org_m7_true_cov_logdet,
        },
        "wishart_bias_terms": {
            "m8_bias": wishart_bias_correction_full,
            "m7_whiten_bias": bias_m7_whiten_naive,
            "m7_whiten_structure_bias": bias_m7_whiten_structural,
            "m7_org_structure_bias": emp_bias_m7_org_structural,

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
        "m7_whiten_Naive_(Normal Wishart)": {
            "Raw_mean_logdet": avg_m7_whiten_naive,
            "Empirical_bias_true_cov": emp_bias_m7_whiten_naive,
            "Empirical_bias_whitened_true_cov": emp_bias_m7_whiten_naive,
            "avg_corrected_logdet": avg_m7_whiten_naive_corrected,
            "Corrected_bias": corr_bias_m7_whiten_naive,
            "std_logdet": np.std(logdets_m7_whiten_naive, ddof=1),
        },
        "m7_whiten_Structural_(Chrodal Graph)": {
            "avg_corrected_logdet": avg_m7_whiten_structural_corrected,
            "Corrected_bias": corr_bias_m7_whiten_structural,
            "std_logdet": np.std(logdets_m7_whiten_structural, ddof=1),
        },
        "m7_org_Structural_(Chrodal Graph)": {
            "Raw_mean_logdet": avg_m7_org_structural,
            "Empirical_bias_true_cov": emp_bias_m7_org_structural,
            "avg_corrected_logdet": org_avg_m7_structural_corrected,
            "Corrected_bias": org_m7_structural_corr_bias,
            "std_logdet": np.std(org_log_dets_m7_org_structural, ddof=1),
        },


    }


def print_m7_whiten_bias_summary(results: dict) -> None:
    s = results["settings"]
    true_logdet = results["True_M8"]
    m8_sample = results["Sample_M8"]
    m7_whiten_naive = results["m7_whiten_Naive_(Normal Wishart)"]
    m7_whiten_struc = results["m7_whiten_Structural_(Chrodal Graph)"]

    print("=" * 72)
    print("m7_whiten Log-Det Bias Comparison")
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
    print(f"  m7_whiten Bias            = {results['wishart_bias_terms']['m7_whiten_bias']:.6f}")
    print(f"  m7_whiten Structural Bias          = {results['wishart_bias_terms']['m7_whiten_structure_bias']:.6f}")
    print(f" m7_org Structural Bias          = {results['wishart_bias_terms']['m7_org_structure_bias']:.6f}")
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

    print("m7_whiten True Log Determinant values:")
    print(f"  True mean logdet            = {true_logdet['true_logdet_m7_whiten_white']:.8f}")
    print(f"  Raw mean logdet            = {m7_whiten_naive['Raw_mean_logdet']:.8f}")
    print(f"  Empirical bias            = {m7_whiten_naive['Empirical_bias_true_cov']:.6f}")
    print()

    print("m7_whiten (Naive and Structural bias corrections) Sample Log Determinant results:")
    print(f"  Naive (Wishart) Corrected mean logdet      = {m7_whiten_naive['avg_corrected_logdet']:.8f}")
    print(f"  Naive (Wishart) After Correction bias (true cov)  = {m7_whiten_naive['Corrected_bias']:.6f}")
    print(f"  Structural Corrected mean logdet      = {m7_whiten_struc['avg_corrected_logdet']:.8f}")
    print(f"  Structural After Correction bias (true cov)  = {m7_whiten_struc['Corrected_bias']:.6f}")

    print("=" * 72)
    print("m7_org (Structural bias correction) Sample Log Determinant results:")
    print(f"  True mean logdet            = {true_logdet['true_logdet_m7_org_structural']:.8f}")
    print(f"  Raw mean logdet            = {results['m7_org_Structural_(Chrodal Graph)']['Raw_mean_logdet']:.8f}")
    print(f"  Empirical bias            = {results['m7_org_Structural_(Chrodal Graph)']['Empirical_bias_true_cov']:.6f}")
    print(f"  Structural Corrected mean logdet      = {results['m7_org_Structural_(Chrodal Graph)']['avg_corrected_logdet']:.8f}")
    print(f"  Structural After Correction bias (true cov)  = {results['m7_org_Structural_(Chrodal Graph)']['Corrected_bias']:.6f}")
    print("=" * 72)




if __name__ == "__main__":
    print("Running m7_whiten and M8 Simulation logdet bias comparison simulation...")
    results = simulate_m7_m8_log_det(
        n_samples=100,
        n0=10,
        n1=10,
        n2=5,
        n_trials=200,
        q_scale=0.0,
        r_scale=0.0,
        p_scale=0.0,
        seed=12345,
    )

    print_m7_whiten_bias_summary(results)