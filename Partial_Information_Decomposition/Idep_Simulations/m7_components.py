import numpy as np
import pandas as pd
import torch

from Partial_Information_Decomposition.Idep_Simulations.logdet_m7_m8 import *


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _build_whitened_blocks_from_cov(S, n0, n1, n2):
    """
    Given a covariance matrix S in block order [X0, X1, Y],
    return the whitened blocks P, Q, R using the same helpers as the main code.
    """
    S_torch = torch.from_numpy(S).to(torch.float64)
    S_dict = create_cov_matrix(Sigma=S_torch, dims=[n0, n1, n2])

    P = whiten_block(S_dict["cov_x0"], S_dict["cross_x0_x1"], S_dict["cov_x1"]).numpy()
    Q = whiten_block(S_dict["cov_x0"], S_dict["cross_x0_x2"], S_dict["cov_x2"]).numpy()
    R = whiten_block(S_dict["cov_x1"], S_dict["cross_x1_x2"], S_dict["cov_x2"]).numpy()
    return P, Q, R, S_dict


def _safe_trial_value(fn, fail_value=None):
    """
    Run a computation that may fail due to non-PD matrices.
    """
    try:
        return fn()
    except np.linalg.LinAlgError:
        return fail_value


def _m7_deno_from_blocks(S_dict):
    """
    Direct block formula for the M7 denominator:
        log|I - Q'Q| + log|I - R'R|
      = log|Sigma_0Y| + log|Sigma_1Y| - log|Sigma_00| - log|Sigma_11| - 2 log|Sigma_YY|
    """
    sigma00 = S_dict["cov_x0"].numpy()
    sigma11 = S_dict["cov_x1"].numpy()
    sigma22 = S_dict["cov_x2"].numpy()
    sigma02 = S_dict["cross_x0_x2"].numpy()
    sigma12 = S_dict["cross_x1_x2"].numpy()

    sigma0y = np.block([
        [sigma00,      sigma02],
        [sigma02.T,    sigma22],
    ])
    sigma1y = np.block([
        [sigma11,      sigma12],
        [sigma12.T,    sigma22],
    ])

    return (
        safe_logdet(sigma0y)
        + safe_logdet(sigma1y)
        - safe_logdet(sigma00)
        - safe_logdet(sigma11)
        - 2.0 * safe_logdet(sigma22)
    )


def _m7_deno_from_qr(Q, R):
    """
    Denominator as written in the whitened M7 formula.
    """
    n2 = Q.shape[1]
    return safe_logdet(np.eye(n2) - (Q.T @ Q)) + safe_logdet(np.eye(n2) - (R.T @ R))


# -----------------------------------------------------------------------------
# Main simulation
# -----------------------------------------------------------------------------

def run_m7_component_diagnostic(
    n_samples=100,
    n0=3,
    n1=3,
    n2=2,
    q_scale=0.35,
    r_scale=0.30,
    p_scale=0.25,
    n_trials=2000,
    seed=0,
    true_model="M7",   # "M7" or "M8"
    progress_every=100,
):
    """
    Diagnose the M7 MI estimator by separating:
      1) numerator bias
      2) denominator bias
      3) total MI bias

    true_model="M7":
        data-generating covariance satisfies population M7, i.e. P = Q R^T

    true_model="M8":
        data-generating covariance is generic saturated model
        (so the M7 estimator is a misspecified projection)
    """
    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")

    true_model = true_model.upper()
    if true_model not in {"M7", "M8"}:
        raise ValueError("true_model must be 'M7' or 'M8'.")

    d = n0 + n1 + n2
    df = n_samples - 1
    if df <= d - 1:
        raise ValueError(f"Need df > d-1. Got df={df}, d={d}.")

    master_rng = np.random.default_rng(seed)
    cov_seed = int(master_rng.integers(0, 2**32 - 1))

    # ---------------------------------------------------------------------
    # 1) Generate population covariance
    # ---------------------------------------------------------------------
    true_cov = make_random_true_cov(
        n0=n0,
        n1=n1,
        n2=n2,
        q_scale=q_scale,
        r_scale=r_scale,
        p_scale=p_scale,
        seed=cov_seed,
        m7_whiten_structural=(true_model == "M7"),
    )

    # Population whitened blocks from the true covariance
    P_true, Q_true, R_true, true_cov_dict = _build_whitened_blocks_from_cov(true_cov, n0, n1, n2)

    # Saturated / M8 population MI
    true_cov_white = np.block([
        [np.eye(n0),     P_true,       Q_true],
        [P_true.T,       np.eye(n1),   R_true],
        [Q_true.T,       R_true.T,     np.eye(n2)],
    ])

    nume8_true = safe_logdet(np.eye(n1) - (P_true.T @ P_true))
    deno8_true = safe_logdet(true_cov_white)
    mi8_true = 0.5 * (nume8_true - deno8_true)

    # M7 population object implied by the population Q_true, R_true
    P_m7_true = Q_true @ R_true.T
    m7_white_true = np.block([
        [np.eye(n0),         P_m7_true,    Q_true],
        [P_m7_true.T,        np.eye(n1),   R_true],
        [Q_true.T,           R_true.T,     np.eye(n2)],
    ])

    nume7_true = safe_logdet(np.eye(n1) - (P_m7_true.T @ P_m7_true))
    deno7_true_qr = _m7_deno_from_qr(Q_true, R_true)
    deno7_true_blocks = _m7_deno_from_blocks(true_cov_dict)
    logdet_m7_white_true = safe_logdet(m7_white_true)
    mi7_true = 0.5 * (nume7_true - deno7_true_qr)

    # These should both be essentially zero if the determinant identity holds numerically
    true_identity_gap_qr = logdet_m7_white_true - deno7_true_qr
    true_identity_gap_blocks = logdet_m7_white_true - deno7_true_blocks
    true_deno_formula_gap = deno7_true_qr - deno7_true_blocks

    # ---------------------------------------------------------------------
    # 2) Theoretical denominator bias on whitened scale
    # ---------------------------------------------------------------------
    b_x0 = logdet_wishart_bias(df, n0)
    b_x1 = logdet_wishart_bias(df, n1)
    b_y = logdet_wishart_bias(df, n2)
    b_c0 = logdet_wishart_bias(df, n0 + n2)
    b_c1 = logdet_wishart_bias(df, n1 + n2)

    # Bias of log|I - Q'Q| + log|I - R'R|
    # = bias of whitened M7 logdet
    bias_deno_theory = b_c0 + b_c1 - b_x0 - b_x1 - 2.0 * b_y

    # If one assumes numerator bias = 0, this is the MI correction
    bias_mi_deno_only_theory = -0.5 * bias_deno_theory

    # ---------------------------------------------------------------------
    # 3) Monte Carlo
    # ---------------------------------------------------------------------
    rows = []
    fail_count = 0

    for i in range(n_trials):
        if progress_every and ((i + 1) % progress_every == 0 or i == 0 or i + 1 == n_trials):
            print(f"Trial {i+1}/{n_trials}...", end="\r")

        # FIX: use a fresh trial seed on each iteration
        trial_seed = int(master_rng.integers(0, 2**32 - 1))
        S = sample_data_from_cov(
            true_cov=true_cov,
            n_samples=n_samples,
            seed=trial_seed,
        )

        def compute_row():
            P_hat, Q_hat, R_hat, S_dict = _build_whitened_blocks_from_cov(S, n0, n1, n2)

            # M8 sample MI
            m8_white = np.block([
                [np.eye(n0),     P_hat,       Q_hat],
                [P_hat.T,        np.eye(n1),  R_hat],
                [Q_hat.T,        R_hat.T,     np.eye(n2)],
            ])
            nume8_hat = safe_logdet(np.eye(n1) - (P_hat.T @ P_hat))
            deno8_hat = safe_logdet(m8_white)
            mi8_hat = 0.5 * (nume8_hat - deno8_hat)

            # M7 sample MI
            P_m7_hat = Q_hat @ R_hat.T
            m7_white = np.block([
                [np.eye(n0),         P_m7_hat,   Q_hat],
                [P_m7_hat.T,         np.eye(n1), R_hat],
                [Q_hat.T,            R_hat.T,    np.eye(n2)],
            ])

            nume7_hat = safe_logdet(np.eye(n1) - (P_m7_hat.T @ P_m7_hat))
            deno7_qr_hat = _m7_deno_from_qr(Q_hat, R_hat)
            deno7_blocks_hat = _m7_deno_from_blocks(S_dict)
            logdet_m7_white_hat = safe_logdet(m7_white)
            mi7_hat = 0.5 * (nume7_hat - deno7_qr_hat)

            return {
                "trial": i,
                "trial_seed": trial_seed,
                "trace_S": float(np.trace(S)),
                "S00": float(S[0, 0]),
                "S01": float(S[0, 1]) if S.shape[0] > 1 else np.nan,

                "m8_numerator_hat": nume8_hat,
                "m8_denominator_hat": deno8_hat,
                "m8_mi_hat": mi8_hat,

                "m7_numerator_hat": nume7_hat,
                "m7_denominator_qr_hat": deno7_qr_hat,
                "m7_denominator_blocks_hat": deno7_blocks_hat,
                "m7_deno_formula_gap_hat": deno7_qr_hat - deno7_blocks_hat,
                "m7_logdet_white_hat": logdet_m7_white_hat,
                "m7_identity_gap_qr_hat": logdet_m7_white_hat - deno7_qr_hat,
                "m7_identity_gap_blocks_hat": logdet_m7_white_hat - deno7_blocks_hat,
                "m7_mi_hat": mi7_hat,
            }

        row = _safe_trial_value(compute_row, fail_value=None)
        if row is None:
            fail_count += 1
            continue
        rows.append(row)

    print(" " * 80, end="\r")

    df_trials = pd.DataFrame(rows)
    if len(df_trials) == 0:
        raise RuntimeError("All trials failed.")

    # ---------------------------------------------------------------------
    # 4) Empirical biases
    # ---------------------------------------------------------------------
    mean_m8_nume = df_trials["m8_numerator_hat"].mean()
    mean_m8_deno = df_trials["m8_denominator_hat"].mean()
    mean_m8_mi = df_trials["m8_mi_hat"].mean()

    mean_m7_nume = df_trials["m7_numerator_hat"].mean()
    mean_m7_deno_qr = df_trials["m7_denominator_qr_hat"].mean()
    mean_m7_deno_blocks = df_trials["m7_denominator_blocks_hat"].mean()
    mean_m7_mi = df_trials["m7_mi_hat"].mean()

    bias_m8_nume = mean_m8_nume - nume8_true
    bias_m8_deno = mean_m8_deno - deno8_true
    bias_m8_mi = mean_m8_mi - mi8_true

    bias_m7_nume = mean_m7_nume - nume7_true
    bias_m7_deno_qr = mean_m7_deno_qr - deno7_true_qr
    bias_m7_deno_blocks = mean_m7_deno_blocks - deno7_true_blocks
    bias_m7_mi = mean_m7_mi - mi7_true

    # This should equal bias_m7_mi up to Monte Carlo error
    reconstructed_m7_mi_bias = 0.5 * (bias_m7_nume - bias_m7_deno_qr)

    # Apply denominator-only correction
    df_trials["m7_mi_deno_only_corrected"] = (
        df_trials["m7_mi_hat"] - bias_mi_deno_only_theory
    )
    mean_m7_mi_deno_only_corrected = df_trials["m7_mi_deno_only_corrected"].mean()
    bias_m7_mi_deno_only_corrected = mean_m7_mi_deno_only_corrected - mi7_true

    summary = {
        "settings": {
            "true_model": true_model,
            "n_samples": n_samples,
            "df": df,
            "dims": {"n0": n0, "n1": n1, "n2": n2, "d": d},
            "q_scale": q_scale,
            "r_scale": r_scale,
            "p_scale": p_scale,
            "n_trials_requested": n_trials,
            "n_trials_used": int(len(df_trials)),
            "n_fail": int(fail_count),
            "seed": seed,
            "cov_seed": cov_seed,
        },

        "population_truth": {
            "m8_numerator_true": nume8_true,
            "m8_denominator_true": deno8_true,
            "m8_mi_true": mi8_true,

            "m7_numerator_true": nume7_true,
            "m7_denominator_true_qr": deno7_true_qr,
            "m7_denominator_true_blocks": deno7_true_blocks,
            "m7_logdet_white_true": logdet_m7_white_true,
            "m7_identity_gap_true_qr": true_identity_gap_qr,
            "m7_identity_gap_true_blocks": true_identity_gap_blocks,
            "m7_deno_formula_gap_true": true_deno_formula_gap,
            "m7_mi_true": mi7_true,
        },

        "theory": {
            "m7_deno_bias_theory": bias_deno_theory,
            "m7_mi_bias_deno_only_theory": bias_mi_deno_only_theory,
        },

        "empirical": {
            "m8_numerator_bias": bias_m8_nume,
            "m8_denominator_bias": bias_m8_deno,
            "m8_mi_bias": bias_m8_mi,

            "m7_numerator_bias": bias_m7_nume,
            "m7_denominator_bias_qr": bias_m7_deno_qr,
            "m7_denominator_bias_blocks": bias_m7_deno_blocks,
            "m7_mi_bias": bias_m7_mi,
            "m7_mi_bias_reconstructed_from_parts": reconstructed_m7_mi_bias,

            "m7_mean_deno_formula_gap_hat": df_trials["m7_deno_formula_gap_hat"].mean(),
            "m7_std_deno_formula_gap_hat": df_trials["m7_deno_formula_gap_hat"].std(ddof=1),
            "m7_mean_identity_gap_qr_hat": df_trials["m7_identity_gap_qr_hat"].mean(),
            "m7_std_identity_gap_qr_hat": df_trials["m7_identity_gap_qr_hat"].std(ddof=1),
            "m7_mean_identity_gap_blocks_hat": df_trials["m7_identity_gap_blocks_hat"].mean(),
            "m7_std_identity_gap_blocks_hat": df_trials["m7_identity_gap_blocks_hat"].std(ddof=1),

            "m7_mi_deno_only_corrected_mean": mean_m7_mi_deno_only_corrected,
            "m7_mi_deno_only_corrected_bias": bias_m7_mi_deno_only_corrected,
        },

        "stds": {
            "m8_mi_std": df_trials["m8_mi_hat"].std(ddof=1),
            "m7_mi_std": df_trials["m7_mi_hat"].std(ddof=1),
            "m7_numerator_std": df_trials["m7_numerator_hat"].std(ddof=1),
            "m7_denominator_qr_std": df_trials["m7_denominator_qr_hat"].std(ddof=1),
            "m7_denominator_blocks_std": df_trials["m7_denominator_blocks_hat"].std(ddof=1),
            "trace_S_std": df_trials["trace_S"].std(ddof=1),
            "S00_std": df_trials["S00"].std(ddof=1),
            "S01_std": df_trials["S01"].std(ddof=1),
        },
    }

    return summary, df_trials


# -----------------------------------------------------------------------------
# Sweep helper
# -----------------------------------------------------------------------------

def sweep_over_sample_size(
    n_samples_grid,
    n0=3,
    n1=3,
    n2=2,
    q_scale=0.35,
    r_scale=0.30,
    p_scale=0.25,
    n_trials=1000,
    seed=0,
    true_model="M7",
):
    """
    Run the component diagnostic over multiple sample sizes.
    """
    rows = []
    for j, N in enumerate(n_samples_grid):
        summary, _ = run_m7_component_diagnostic(
            n_samples=N,
            n0=n0,
            n1=n1,
            n2=n2,
            q_scale=q_scale,
            r_scale=r_scale,
            p_scale=p_scale,
            n_trials=n_trials,
            seed=seed + j,
            true_model=true_model,
        )

        emp = summary["empirical"]
        theory = summary["theory"]
        s = summary["settings"]

        rows.append({
            "true_model": s["true_model"],
            "N": N,
            "df": s["df"],
            "n0": n0,
            "n1": n1,
            "n2": n2,
            "q_scale": q_scale,
            "r_scale": r_scale,
            "p_scale": p_scale,

            "m7_numerator_bias_emp": emp["m7_numerator_bias"],
            "m7_denominator_bias_qr_emp": emp["m7_denominator_bias_qr"],
            "m7_denominator_bias_blocks_emp": emp["m7_denominator_bias_blocks"],
            "m7_mi_bias_emp": emp["m7_mi_bias"],
            "m7_mi_bias_from_parts_emp": emp["m7_mi_bias_reconstructed_from_parts"],

            "m7_deno_bias_theory": theory["m7_deno_bias_theory"],
            "m7_mi_bias_deno_only_theory": theory["m7_mi_bias_deno_only_theory"],
            "m7_mi_deno_only_corrected_bias": emp["m7_mi_deno_only_corrected_bias"],

            "m7_deno_formula_gap_mean": emp["m7_mean_deno_formula_gap_hat"],
            "m7_deno_formula_gap_std": emp["m7_std_deno_formula_gap_hat"],
            "m7_identity_gap_qr_mean": emp["m7_mean_identity_gap_qr_hat"],
            "m7_identity_gap_qr_std": emp["m7_std_identity_gap_qr_hat"],
            "m7_identity_gap_blocks_mean": emp["m7_mean_identity_gap_blocks_hat"],
            "m7_identity_gap_blocks_std": emp["m7_std_identity_gap_blocks_hat"],
            "m7_numerator_std": summary["stds"]["m7_numerator_std"],
            "m7_denominator_qr_std": summary["stds"]["m7_denominator_qr_std"],
            "m7_denominator_blocks_std": summary["stds"]["m7_denominator_blocks_std"],
            "m7_mi_std": summary["stds"]["m7_mi_std"],
            "trace_S_std": summary["stds"]["trace_S_std"],
            "S00_std": summary["stds"]["S00_std"],
            "S01_std": summary["stds"]["S01_std"],

            "n_trials_used": s["n_trials_used"],
            "n_fail": s["n_fail"],
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Printer
# -----------------------------------------------------------------------------

def print_diagnostic_summary(summary):
    s = summary["settings"]
    pop = summary["population_truth"]
    th = summary["theory"]
    emp = summary["empirical"]
    stds = summary["stds"]

    print("\n" + "=" * 72)
    print("M7 component diagnostic")
    print("=" * 72)
    print(
        f"true_model={s['true_model']}, N={s['n_samples']}, df={s['df']}, "
        f"(n0,n1,n2)=({s['dims']['n0']},{s['dims']['n1']},{s['dims']['n2']}), "
        f"trials used={s['n_trials_used']}, failed={s['n_fail']}"
    )
    print(
        f"q_scale={s['q_scale']}, r_scale={s['r_scale']}, p_scale={s['p_scale']}, "
        f"seed={s['seed']}, cov_seed={s['cov_seed']}"
    )
    print()

    print("Population truth:")
    print(f"  M7 numerator true               = {pop['m7_numerator_true']:.8f}")
    print(f"  M7 denominator true (QR)        = {pop['m7_denominator_true_qr']:.8f}")
    print(f"  M7 denominator true (blocks)    = {pop['m7_denominator_true_blocks']:.8f}")
    print(f"  M7 logdet(whitened) true        = {pop['m7_logdet_white_true']:.8f}")
    print(f"  M7 identity gap true (QR)       = {pop['m7_identity_gap_true_qr']:.3e}")
    print(f"  M7 identity gap true (blocks)   = {pop['m7_identity_gap_true_blocks']:.3e}")
    print(f"  M7 deno formula gap true        = {pop['m7_deno_formula_gap_true']:.3e}")
    print(f"  M7 MI true                      = {pop['m7_mi_true']:.8f}")
    print()

    print("Empirical biases:")
    print(f"  numerator bias                  = {emp['m7_numerator_bias']:.8f}")
    print(f"  denominator bias (QR)           = {emp['m7_denominator_bias_qr']:.8f}")
    print(f"  denominator bias (blocks)       = {emp['m7_denominator_bias_blocks']:.8f}")
    print(f"  MI bias                         = {emp['m7_mi_bias']:.8f}")
    print(f"  MI bias from parts              = {emp['m7_mi_bias_reconstructed_from_parts']:.8f}")
    print()

    print("Theory:")
    print(f"  denominator bias theory         = {th['m7_deno_bias_theory']:.8f}")
    print(f"  MI bias (deno only) theory      = {th['m7_mi_bias_deno_only_theory']:.8f}")
    print()

    print("After denominator-only correction:")
    print(f"  corrected mean bias (M7_MI)             = {emp['m7_mi_deno_only_corrected_bias']:.8f}")
    print()

    print("Sanity checks:")
    print(f"  mean deno formula gap           = {emp['m7_mean_deno_formula_gap_hat']:.3e}")
    print(f"  std deno formula gap            = {emp['m7_std_deno_formula_gap_hat']:.3e}")
    print(f"  mean identity gap (QR)          = {emp['m7_mean_identity_gap_qr_hat']:.3e}")
    print(f"  std identity gap (QR)           = {emp['m7_std_identity_gap_qr_hat']:.3e}")
    print(f"  mean identity gap (blocks)      = {emp['m7_mean_identity_gap_blocks_hat']:.3e}")
    print(f"  std identity gap (blocks)       = {emp['m7_std_identity_gap_blocks_hat']:.3e}")
    print()

    print("Monte Carlo spread:")
    print(f"  std trace(S)                    = {stds['trace_S_std']:.8f}")
    print(f"  std S[0,0]                      = {stds['S00_std']:.8f}")
    print(f"  std S[0,1]                      = {stds['S01_std']:.8f}")
    print(f"  std numerator                   = {stds['m7_numerator_std']:.8f}")
    print(f"  std denominator (QR)            = {stds['m7_denominator_qr_std']:.8f}")
    print(f"  std denominator (blocks)        = {stds['m7_denominator_blocks_std']:.8f}")
    print(f"  std MI                          = {stds['m7_mi_std']:.8f}")
    print("=" * 72)


if __name__ == "__main__":
    summary, trials = run_m7_component_diagnostic(
        n_samples=500,
        n0=10,
        n1=10,
        n2=5,
        q_scale=0.25,
        r_scale=0.25,
        p_scale=0.25,
        n_trials=1000,
        seed=12345,
        true_model="M7",
    )
    print_diagnostic_summary(summary)

    sweep_df = sweep_over_sample_size(
        n_samples_grid=[50, 100, 200, 500, 1000],
        n0=10,
        n1=10,
        n2=5,
        q_scale=0.25,
        r_scale=0.25,
        p_scale=0.25,
        n_trials=500,
        seed=12345,
        true_model="M7",
    )
    print("\nSweep results:")
    print(sweep_df)
