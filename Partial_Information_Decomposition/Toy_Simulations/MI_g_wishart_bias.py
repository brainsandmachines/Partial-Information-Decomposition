
"""
compare_mutual_information.py

Simulation comparing mutual information estimates under the same
conditions as the logdet simulation. This script assumes all required
functions are available from:

    from g_wishart_bias_corr import *

No functions are redefined here — everything is imported and reused.
"""

import torch
import numpy as np
import argparse

# Import all existing utilities from the user's module
from g_wishart_bias_corr import *


def run_MI_simulation(
    n_samples=100,
    n0=3,
    n1=3,
    n2=2,
    q_scale=0.35,
    r_scale=0.30,
    p_scale=0.25,
    n_trials=2000,
    seed=0,
):  
    """
    Run MI simulation under the same covariance construction used
    in the logdet experiments.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    Sigma = make_random_true_cov(
    n0=n0,
    n1=n1,
    n2=n2,
    q_scale=q_scale,
    r_scale=r_scale,
    p_scale=p_scale,
        )

    
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
    
    P_true = whiten_block(sigma00, sigma01, sigma11).numpy()
    Q_true = whiten_block(sigma00, sigma02, sigma22).numpy()
    R_true = whiten_block(sigma11, sigma12, sigma22).numpy()


    true_cov_white = np.block([
        [np.eye(n0),         P_true, Q_true],
        [P_true.T,  np.eye(n1),        R_true],
        [Q_true.T,           R_true.T,          np.eye(n2)],
    ])

    nume8_true = safe_logdet(np.eye(n1) - (P_true.T @ P_true))
    deno8_true = safe_logdet(true_cov_white)
    m8_MI_true = 0.5*(nume8_true-deno8_true)          


    print(f"True MI under M8 construction: {m8_MI_true:.6f}")


    P_m7 = Q_true @ R_true.T
    m7_true_cov = np.block([
        [np.eye(n0),        P_m7, Q_true],
        [P_m7.T,  np.eye(n1),        R_true],
        [Q_true.T,           R_true.T,          np.eye(n2)],
    ])

    
    nume7_true = safe_logdet(np.eye(n1) - (P_m7.T @ P_m7))
    deno7_true = safe_logdet(np.eye(n2) - (Q_true.T @ Q_true)) + safe_logdet(np.eye(n2) - (R_true.T @ R_true))
    m7_MI_true = 0.5*(nume7_true-deno7_true)

    print(f"True MI under M7 construction: {m7_MI_true:.6f}")

    d = n0 + n1 + n2


    # Wishart bias corrections
    df = n_samples - 1


# A. DEFINE BIAS TERMS
    # Marginal Biases (Fixed by whitening)
    b_x0 = logdet_wishart_bias(df, n0)
    b_x1 = logdet_wishart_bias(df, n1)
    b_y  = logdet_wishart_bias(df, n2)
    
    # M8 (Saturated) Biases
    b_pred_m8 = logdet_wishart_bias(df, n0 + n1)
    b_joint_m8 = logdet_wishart_bias(df, d)
    
    # M7 (Structural) Biases
    b_c0 = logdet_wishart_bias(df, n0 + n2) # Clique 0
    b_c1 = logdet_wishart_bias(df, n1 + n2) # Clique 1
    b_sep = b_y                             # Separator
    
    # Final MI Bias Corrections (Whitened Scale)
    # M8 MI Bias = 0.5 * ( (B_pred - B_marginals) - (B_joint - B_marginals) )
    bias_mi_m8 = 0.5 * ( (b_pred_m8 - (b_x0 + b_x1)) - (b_joint_m8 - (b_x0 + b_x1 + b_y)) )
    
    # M7 MI Bias = 0.5 * ( (B_pred_struct - B_marginals) - (B_joint_struct - B_marginals) )
    # Note: b_joint_m7 = b_c0 + b_c1 - b_sep
    bias_mi_m7 = 0.5 * ( (b_x0 + b_x1 + 2*b_y - b_c0 - b_c1) ) # Corrected sign

    mi_m8_list = []
    mi_m7_naive_list = []

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}", end="\r")
        # Build true covariance exactly the same way

        # Sample data
        Z = sample_data_from_cov(Sigma, n_samples)

        # Sample covariance
        Z_torch = torch.from_numpy(Z).to(torch.float64)
        Z_dict = create_cov_matrix(Sigma=Z_torch, dims=[n0, n1, n2])


        Q_m8_m7 = whiten_block(Z_dict['cov_x0'], Z_dict['cross_x0_x2'], Z_dict['cov_x2']).numpy()
        R_m8_m7 = whiten_block(Z_dict['cov_x1'], Z_dict['cross_x1_x2'], Z_dict['cov_x2']).numpy()
        P = whiten_block(Z_dict['cov_x0'], Z_dict['cross_x0_x1'], Z_dict['cov_x1']).numpy()

        #M8 true MI
        m8_white = np.block([
            [np.eye(n0),         P, Q_m8_m7],
            [P.T,  np.eye(n1),        R_m8_m7],
            [Q_m8_m7.T,           R_m8_m7.T,          np.eye(n2)],
        ])
        nume8 = safe_logdet(np.eye(n1) - (P.T @ P))
        deno8 = safe_logdet(m8_white)
        m8_raw = 0.5*(nume8-deno8) 

        #M7 true MI
        P_m7 = Q_m8_m7 @ R_m8_m7.T
        m7_white = np.block([
            [np.eye(n0),         P_m7, Q_m8_m7],
            [P_m7.T,  np.eye(n1),        R_m8_m7],
            [Q_m8_m7.T,           R_m8_m7.T,          np.eye(n2)],
        ])

        nume7 = safe_logdet(np.eye(n1) - (P_m7.T @ P_m7))
        deno7 = safe_logdet(np.eye(n2) - (Q_m8_m7.T @ Q_m8_m7)) + safe_logdet(np.eye(n2) - (R_m8_m7.T @ R_m8_m7))
        m7_raw = 0.5*(nume7-deno7)

        mi_m8_list.append(m8_raw)
        mi_m7_naive_list.append(m7_raw)

    
    mi_m8 = np.asarray(mi_m8_list)
    mi_m7_naive = np.asarray(mi_m7_naive_list)
    mi_m7_structural = np.asarray(mi_m7_naive_list)     

    
    avg_m8 = np.mean(mi_m8)
    avg_m7_naive = np.mean(mi_m7_naive)


    emp_bias_m8 = avg_m8 - m8_MI_true
    emp_bias_m7_naive = avg_m7_naive - m7_MI_true

    avg_m8_corrected = avg_m8- bias_mi_m8
    avg_m7_naive_corrected = avg_m7_naive - bias_mi_m8
    avg_m7_structural_corrected = avg_m7_naive - bias_mi_m7

    corr_bias_m8 = avg_m8_corrected - m8_MI_true 
    corr_bias_m7_naive = avg_m7_naive_corrected - m7_MI_true
    corr_bias_m7_structural = avg_m7_structural_corrected - m7_MI_true





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
            "true_mi": {
                "mi_m8": m8_MI_true,
                "mi_m7_white": m7_MI_true,
            },
            "wishart_bias_terms": {
                "m8_bias": bias_mi_m8,
                "m7_whitened_bias": bias_mi_m8,
                "m7_structure_bias": bias_mi_m7,
            },
            "Sample_M8": {
                "Raw_mean_mi": avg_m8,
                "Empirical_bias_true_cov": emp_bias_m8,
                "avg_corrected_mi": avg_m8_corrected,
                "Corrected_bias": corr_bias_m8,
                "std_mi": np.std(m8_raw, ddof=1),
            },
            "M7_Naive_(Normal Wishart)": {
                "Raw_mean_mi": avg_m7_naive,
                "Empirical_bias_true_cov": emp_bias_m7_naive,
                "Empirical_bias_whitened_true_cov": emp_bias_m7_naive,
                "avg_corrected_mi": avg_m7_naive_corrected,
                "Corrected_bias": corr_bias_m7_naive,
                "std_mi": np.std(m7_raw, ddof=1),
            },
            "M7_Structural_(Chrodal Graph)": {
                "avg_corrected_mi": avg_m7_structural_corrected,
                "Corrected_bias": corr_bias_m7_structural,
                "std_mi": np.std(m7_raw, ddof=1),
            },

        }


def print_m7_m8_bias_summary(results: dict) -> None:
    s = results["settings"]
    true_mi = results["true_mi"]
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

    print("True M8 MI values:")
    print(f"  True mean MI            = {true_mi['mi_m8']:.8f}")
    print(f"  Raw mean MI            = {m8_sample['Raw_mean_mi']:.8f}")
    print(f"  Empirical bias (true cov)  = {m8_sample['Empirical_bias_true_cov']:.6f}")
    print()
    print("M8 Sample MI results:")
    print(f"  Correct M8 MI         = {m8_sample['avg_corrected_mi']:.8f}")
    print(f"  After Correction bias (true cov)  = {m8_sample['Corrected_bias']:.6f}")
    print("=" * 72)

    print("M7 True MI values:")
    print(f"  True mean MI            = {true_mi['mi_m7_white']:.8f}")
    print(f"  Raw mean MI            = {m7_naive['Raw_mean_mi']:.8f}")
    print(f"  Empirical bias            = {m7_naive['Empirical_bias_true_cov']:.6f}")
    print()

    print("M7 (Naive and Structural bias corrections) Sample MI results:")
    print(f"  Naive (Wishart) Corrected mean MI      = {m7_naive['avg_corrected_mi']:.8f}")
    print(f"  Naive (Wishart) After Correction bias (true cov)  = {m7_naive['Corrected_bias']:.6f}")
    print(f"  Structural Corrected mean MI      = {m7_struc['avg_corrected_mi']:.8f}")
    print(f"  Structural After Correction bias (true cov)  = {m7_struc['Corrected_bias']:.6f}")


def main():

    print("\n===============================")
    print("Mutual Information Simulation")
    print("===============================\n")
    results_mi = run_MI_simulation(
        n_samples=1000,
        n0=50,
        n1=50,
        n2=50,
        n_trials=1000,
        q_scale=0.0,
        r_scale=0.0,
        p_scale=0.0,
        seed=12345,
    )




    print_m7_m8_bias_summary(results_mi)



if __name__ == "__main__":
    main()
