

import pathlib
from py_compile import main
import torch
import numpy as np
import argparse
import yaml
from functools import partial
# Import all existing utilities from the user's module

from Simulation_utils import *
from wrapper_M7_M8_models import simulation
from Simulation_utils import *
from logdet_m7_m8 import  sort_m7_m8_results
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from Partial_Information_Decomposition.resampling_wrapper import bias_resampling
from Partial_Information_Decomposition.numertaor_m7_bias import  bias_m7_nume_second_order


# -----------------------------------------------------------------------------
# Second-order M7 non-whitened numerator correction
# -----------------------------------------------------------------------------

def _ensure_2d_sigma(Sigma: torch.Tensor) -> torch.Tensor:
    """Accept (d,d) or (1,d,d) and return (d,d)."""
    if Sigma.ndim == 3:
        if Sigma.shape[0] != 1:
            raise ValueError("Analytic second-order bias expects a single covariance matrix.")
        Sigma = Sigma.squeeze(0)
    if Sigma.ndim != 2:
        raise ValueError(f"Expected Sigma to be 2D, got shape {tuple(Sigma.shape)}")
    return Sigma.to(torch.float64)


def _commutation_matrix(m: int, n: int, *, device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    """K_{m,n} such that vec(A.T) = K vec(A) for A in R^{m x n}."""
    K = torch.zeros((m * n, m * n), dtype=dtype, device=device)
    for i in range(m):
        for j in range(n):
            K[j * m + i, i * n + j] = 1.0
    return K


def _stable_pd_inverse(A: torch.Tensor, jitter: float = 1e-10) -> torch.Tensor:
    """Invert a symmetric PD matrix with a tiny safety jitter if needed."""
    A = 0.5 * (A + A.mT)
    eigmin = torch.min(torch.linalg.eigvalsh(A)).item()
    if eigmin <= jitter:
        A = A + (jitter - eigmin + jitter) * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return torch.linalg.inv(A)


def _m7_sample_whitened_qrp_from_sigma(Sigma: torch.Tensor, n0: int, n1: int, n2: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    From a single sample covariance S in raw block coordinates [X0, X1, T], construct the
    induced M7 cross-block on the raw scale and then whiten it on the predictor/target scale.

    Returns
    -------
    Q_hat : (n0, n2)
    R_hat : (n1, n2)
    P_hat : (n0, n1) = Q_hat @ R_hat.T
    """
    S_dict = create_cov_matrix(Sigma=Sigma, dims=[n0, n1, n2])

    Q_hat = whiten_block(S_dict["cov_x0"], S_dict["cross_x0_x2"], S_dict["cov_x2"]).to(torch.float64)
    R_hat = whiten_block(S_dict["cov_x1"], S_dict["cross_x1_x2"], S_dict["cov_x2"]).to(torch.float64)
    P_hat = Q_hat @ R_hat.mT
    return Q_hat, R_hat, P_hat


def m7_hidden_logdet_second_order_bias(config: dict) -> float:
    r"""
    Plug-in O(1/N) bias for the hidden hard term in the non-whitened M7 joint numerator:

        log | joint_x0_x1^(M7) |
      = log |S00| + log |S11| + log | I - P^T P |

    where P is the sample-whitened induced M7 cross-block.

    This uses the second-order formula derived in the chat for

        f(P) = log | I - P^T P |.

    The covariance plug-in uses the leading-order core M7 expression

        Sigma_Delta ≈ (1/df) [
            Sigma1 ⊗ QQ^T
            + RR^T ⊗ Sigma0
            + (R ⊗ Q)(I + K_d)(R^T ⊗ Q^T)
        ],

    with
        Sigma0 = I - QQ^T,
        Sigma1 = I - RR^T,
        P = QR^T.

    Notes
    -----
    * This is a plug-in analytic approximation.
    * It corrects only the hidden hard term log|I - P^T P|.
    * The easy Wishart logdet pieces must still be added separately.
    """
    n0 = int(config["n0"])
    n1 = int(config["n1"])
    n2 = int(config["n2"])
    df = int(config["n_samples"]) - 1

    Sigma = _ensure_2d_sigma(config["Sigma"])
    device = Sigma.device
    dtype = Sigma.dtype

    Q, R, P = _m7_sample_whitened_qrp_from_sigma(Sigma, n0, n1, n2)

    I0 = torch.eye(n0, dtype=dtype, device=device)
    I1 = torch.eye(n1, dtype=dtype, device=device)
    Id = torch.eye(n2, dtype=dtype, device=device)

    Sigma0 = 0.5 * ((I0 - Q @ Q.mT) + (I0 - Q @ Q.mT).mT)
    Sigma1 = 0.5 * ((I1 - R @ R.mT) + (I1 - R @ R.mT).mT)

    M = 0.5 * ((I1 - P.mT @ P) + (I1 - P.mT @ P).mT)
    M_inv = _stable_pd_inverse(M)

    # Leading covariance of vec(Delta_star) from the second-order derivation.
    Kd = _commutation_matrix(n2, n2, device=device, dtype=dtype)
    Sigma_delta = (
        torch.kron(Sigma1.contiguous(), (Q @ Q.mT).contiguous())
        + torch.kron((R @ R.mT).contiguous(), Sigma0.contiguous())
        + torch.kron(R.contiguous(), Q.contiguous()) @ (torch.eye(n2 * n2, dtype=dtype, device=device) + Kd) @ torch.kron(R.mT.contiguous(), Q.mT.contiguous())
    ) / float(df)

    # J_P for vec(P^T Delta + Delta^T P)
    K01 = _commutation_matrix(n0, n1, device=device, dtype=dtype)
    J_P = torch.kron(torch.eye(n1, dtype=dtype, device=device), P.mT.contiguous()) + torch.kron(P.mT.contiguous(), torch.eye(n1, dtype=dtype, device=device)) @ K01

    term1 = torch.trace(torch.kron(M_inv.contiguous(), torch.eye(n0, dtype=dtype, device=device)) @ Sigma_delta)
    term2 = torch.trace(torch.kron(M_inv.contiguous(), M_inv.contiguous()) @ J_P @ Sigma_delta @ J_P.mT)

    bias_hard_logdet = -(term1 + 0.5 * term2)
    return float(bias_hard_logdet.item())


# -----------------------------------------------------------------------------
# Bias wrappers matching the simulation API used in mi_bias_calc
# -----------------------------------------------------------------------------

def calculate_bias_second_order(
    config: dict,
    *,
    mi: bool = False,
    nume: bool = False,
    nume_joint: bool = False,
    nume_target: bool = False,
    deno: bool = False,
    bias_correction: bool = True,
) -> dict:
    """
    Analytic bias wrapper for the non-whitened M7/M8 simulation.

    For M8 we keep the standard Wishart corrections.
    For M7 we decompose the joint numerator into easy Wishart pieces plus the hard hidden term:

        log |joint_x0_x1^(M7)|
      = log |S00| + log |S11| + log |I - P^T P|.

    The hidden hard term uses the second-order plug-in correction from the derivation in chat.
    """
    if not bias_correction:
        return {config["st"]: 0.0}

    model = config["model"]
    n0 = int(config["n0"])
    n1 = int(config["n1"])
    n2 = int(config["n2"])
    df = int(config["n_samples"]) - 1

    # Standard Wishart logdet biases for unbiased sample covariance terms.
    bias00 = logdet_wishart_bias(df, n0)
    bias11 = logdet_wishart_bias(df, n1)
    bias2 = logdet_wishart_bias(df, n2)
    bias01 = logdet_wishart_bias(df, n0 + n1)
    bias02 = logdet_wishart_bias(df, n0 + n2)
    bias12 = logdet_wishart_bias(df, n1 + n2)
    bias012 = logdet_wishart_bias(df, n0 + n1 + n2)

    if model == "M8":
        bias_nume_joint = 0.5 * bias01
        bias_nume_target = 0.5 * bias2
        bias_nume = bias_nume_joint + bias_nume_target
        bias_deno = 0.5 * bias012
        bias_mi = bias_nume - bias_deno
    elif model == "M7":
        bias_hard = m7_hidden_logdet_second_order_bias(config)
        bias_nume_joint = 0.5 * (bias00 + bias11 + bias_hard)
        bias_nume_target = 0.5 * bias2
        bias_nume = bias_nume_joint + bias_nume_target
        # Exact determinant factorization of the M7 denominator:
        # log|S_m7| = log|S_{0T}| + log|S_{1T}| - log|S_T|
        bias_deno = 0.5 * (bias02 + bias12 - bias2)
        bias_mi = bias_nume - bias_deno
    else:
        raise ValueError(f"Unknown model {model!r}. Expected 'M7' or 'M8'.")

    if mi:
        return {"mi": float(bias_mi)}
    if nume:
        return {"nume": float(bias_nume)}
    if nume_joint:
        return {"nume_joint": float(bias_nume_joint)}
    if nume_target:
        return {"nume_target": float(bias_nume_target)}
    if deno:
        return {"deno": float(bias_deno)}
    raise ValueError("One of mi/nume/nume_joint/nume_target/deno must be True.")


# -----------------------------------------------------------------------------
# Simulation code mirroring mi_m7_m8_notwhiten.py
# -----------------------------------------------------------------------------

def simulate_m7_m8_mi(
    data: list,
    sim_config: dict,
    rng: torch.Generator | None = None,
):
    """Run the M7/M8 MI simulation with the new second-order M7 numerator correction."""
    n_samples = sim_config["n_samples"]
    n0 = sim_config["n0"]
    n1 = sim_config["n1"]
    n2 = sim_config["n2"]
    n_trials = sim_config["n_trials"]
    device = sim_config["device"]

    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")

    d = n0 + n1 + n2
    df = n_samples - 1
    if df <= d - 1:
        raise ValueError(
            f"Need df > d-1 for stable logdet expectation. Got n_samples={n_samples}, df={df}, d={d}."
        )

    m8_true_cov, m7_true_cov = data

    # Ground-truth values for reporting.
    m8_true_cov_dict = create_cov_matrix(Sigma=m8_true_cov, dims=[n0, n1, n2])
    deno8_true = 0.5 * safe_logdet(m8_true_cov)
    nume8_joint_true = 0.5 * safe_logdet(m8_true_cov_dict["joint_x0_x1"])
    nume8_target_true = 0.5 * safe_logdet(m8_true_cov_dict["cov_x2"])
    nume8_true = nume8_joint_true + nume8_target_true
    mi_m8_true = nume8_true - deno8_true

    m7_true_cov_dict = create_cov_matrix(Sigma=m7_true_cov, dims=[n0, n1, n2])
    deno7_true = 0.5 * safe_logdet(m7_true_cov)
    nume7_joint_true = 0.5 * safe_logdet(m7_true_cov_dict["joint_x0_x1"])
    nume7_target_true = 0.5 * safe_logdet(m7_true_cov_dict["cov_x2"])
    nume7_true = nume7_joint_true + nume7_target_true
    mi_m7_true = nume7_true - deno7_true

    stats_keys = ["mi", "nume", "nume_joint", "nume_target", "deno"]
    m8_dict_values = {k: [] for k in stats_keys}
    m7_dict_values = {k: [] for k in stats_keys}
    m8_corrected = {k: [] for k in stats_keys}
    m7_corrected = {k: [] for k in stats_keys}

    for i in range(n_trials):
        print(f"Trial {i + 1}/{n_trials}", end="\r")

        S, rv_list = sample_data_from_cov(sim_config, true_cov=m8_true_cov, rng=rng)

        trial_cfg = sim_config.copy()
        trial_cfg["Sigma"] = S.unsqueeze(0)
        trial_cfg["model"] = "M8_M7"
        raw_results,_ = mi_calculation_not_whiten(config=trial_cfg)

        # M8 raw statistics
        m8_raw = raw_results["M8"]
        for key in stats_keys:
            m8_dict_values[key].append(float(m8_raw["mi" if key == "mi" else key]))

        # M7 raw statistics
        m7_raw = raw_results["M7"]
        for key in stats_keys:
            m7_dict_values[key].append(float(m7_raw["mi" if key == "mi" else key]))

        # Bias correction configs
        cfg_m8 = sim_config.copy()
        cfg_m7 = sim_config.copy()
        cfg_m8["model"] = "M8"
        cfg_m7["model"] = "M7"
        cfg_m8["Sigma"] = S
        cfg_m7["Sigma"] = S
        cfg_m8["rvs_list"] = rv_list
        cfg_m7["rvs_list"] = rv_list
        cfg_m8["calc_statistic_func"] = mi_calculation_not_whiten
        cfg_m7["calc_statistic_func"] = mi_calculation_not_whiten

        bias_m8 = mi_bias_calc(cfg_m8)
        bias_m7 = mi_bias_calc(cfg_m7)

        for key in stats_keys:
            raw_key = "mi" if key == "mi" else key
            m8_corrected[key].append(float(m8_raw[raw_key]) - float(bias_m8[key]))
            m7_corrected[key].append(float(m7_raw[raw_key]) - float(bias_m7[key]))

    # Convert to tensors for summaries
    def _tensorize(dct):
        return {k: torch.tensor(v, dtype=torch.float64) for k, v in dct.items()}

    m8_samples = _tensorize(m8_dict_values)
    m7_samples = _tensorize(m7_dict_values)
    m8_corr = _tensorize(m8_corrected)
    m7_corr = _tensorize(m7_corrected)

    def _make_summary(samples, corrected, ground_truths):
        out = {}
        for key, gt in ground_truths.items():
            avg = torch.mean(samples[key])
            out[key] = {
                "sample": samples[key],
                "avg": avg,
                "corrected_avg": torch.mean(corrected[key]),
                "std": torch.std(samples[key]),
                "emp_bias": avg - gt,
                "ground_truth": gt,
            }
        return out

    m8_summary = _make_summary(
        m8_samples,
        m8_corr,
        {
            "mi": mi_m8_true,
            "nume": nume8_true,
            "nume_joint": nume8_joint_true,
            "nume_target": nume8_target_true,
            "deno": deno8_true,
        },
    )
    m7_summary = _make_summary(
        m7_samples,
        m7_corr,
        {
            "mi": mi_m7_true,
            "nume": nume7_true,
            "nume_joint": nume7_joint_true,
            "nume_target": nume7_target_true,
            "deno": deno7_true,
        },
    )

    return {
        "M8_mi": m8_summary["mi"],
        "M8_nume": m8_summary["nume"],
        "M8_joint": m8_summary["nume_joint"],
        "M8_target": m8_summary["nume_target"],
        "M8_deno": m8_summary["deno"],
        "M7_mi": m7_summary["mi"],
        "M7_nume": m7_summary["nume"],
        "M7_joint": m7_summary["nume_joint"],
        "M7_target": m7_summary["nume_target"],
        "M7_deno": m7_summary["deno"],
    }


def sort_m7_m8_results(results_list):
    """Keep the same output structure as the original file for plotting wrappers."""
    m7_lists = {"mi": [], "nume": [], "joint": [], "target": [], "deno": []}
    m8_lists = {"mi": [], "nume": [], "joint": [], "target": [], "deno": []}

    for res in results_list:
        N = res["N"]
        p = res["p"]
        m7_lists["mi"].append({"N": N, "p": p, "mean": res["M7_mi_mean"], "std": res["M7_mi_std"], "ground_truth": res["M7_mi_ground_truth"]})
        m7_lists["nume"].append({"N": N, "p": p, "mean": res["M7_nume_mean"], "std": res["M7_nume_std"], "ground_truth": res["M7_nume_ground_truth"]})
        m7_lists["joint"].append({"N": N, "p": p, "mean": res["M7_joint_mean"], "std": res["M7_joint_std"], "ground_truth": res["M7_joint_ground_truth"]})
        m7_lists["target"].append({"N": N, "p": p, "mean": res["M7_target_mean"], "std": res["M7_target_std"], "ground_truth": res["M7_target_ground_truth"]})
        m7_lists["deno"].append({"N": N, "p": p, "mean": res["M7_deno_mean"], "std": res["M7_deno_std"], "ground_truth": res["M7_deno_ground_truth"]})

        m8_lists["mi"].append({"N": N, "p": p, "mean": res["M8_mi_mean"], "std": res["M8_mi_std"], "ground_truth": res["M8_mi_ground_truth"]})
        m8_lists["nume"].append({"N": N, "p": p, "mean": res["M8_nume_mean"], "std": res["M8_nume_std"], "ground_truth": res["M8_nume_ground_truth"]})
        m8_lists["joint"].append({"N": N, "p": p, "mean": res["M8_joint_mean"], "std": res["M8_joint_std"], "ground_truth": res["M8_joint_ground_truth"]})
        m8_lists["target"].append({"N": N, "p": p, "mean": res["M8_target_mean"], "std": res["M8_target_std"], "ground_truth": res["M8_target_ground_truth"]})
        m8_lists["deno"].append({"N": N, "p": p, "mean": res["M8_deno_mean"], "std": res["M8_deno_std"], "ground_truth": res["M8_deno_ground_truth"]})

    return [m7_lists["mi"], m7_lists["nume"], m7_lists["joint"], m7_lists["target"], m7_lists["deno"]], [m8_lists["mi"], m8_lists["nume"], m8_lists["joint"], m8_lists["target"], m8_lists["deno"]]


def simulation_wrapper(config: dict) -> dict:
    """Wire the simulation with the new analytic second-order M7 non-whitened bias correction."""
    seed = config["seed"]
    sim_func = simulate_m7_m8_mi

    m8_mi_func = partial(calculate_bias_second_order, mi=True)
    m8_nume_func = partial(calculate_bias_second_order, nume=True)
    m8_nume_joint_func = partial(calculate_bias_second_order, nume_joint=True)
    m8_nume_target_func = partial(calculate_bias_second_order, nume_target=True)
    m8_deno_func = partial(calculate_bias_second_order, deno=True)

    m7_mi_func = partial(calculate_bias_second_order, mi=True)
    m7_nume_func = partial(calculate_bias_second_order, nume=True)
    m7_nume_joint_func = partial(calculate_bias_second_order, nume_joint=True)
    m7_nume_target_func = partial(calculate_bias_second_order, nume_target=True)
    m7_deno_func = partial(calculate_bias_second_order, deno=True)

    bias_corr_func = {
        "M8": {
            "mi": m8_mi_func,
            "nume": m8_nume_func,
            "nume_joint": m8_nume_joint_func,
            "nume_target": m8_nume_target_func,
            "deno": m8_deno_func,
        },
        "M7": {
            "mi": m7_mi_func,
            "nume": m7_nume_func,
            "nume_joint": m7_nume_joint_func,
            "nume_target": m7_nume_target_func,
            "deno": m7_deno_func,
        },
    }

    corr_value_func = corrected_statistic
    functions_dict = {
        "s_simulation": sim_func,
        "bias_correction": bias_corr_func,
        "corrected_statistic": corr_value_func,
    }
    return simulation(config, functions_dict, seed=seed)


if __name__ == "__main__":
    print("Running M7/M8 non-whitened MI simulation with second-order M7 numerator correction...")

    # This main block mirrors the original file but is optional.
    exp_name = "MI>0_m7_m8_notwhiten_second_order_bigtest"
    yaml_file = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/configs/sim.yaml"


    with open(yaml_file, "r") as f:
        super_config = yaml.safe_load(f)

    config = super_config["Mutual_Information_Simulation"]
    n_p_config = super_config["N_P_variations"]
    config["p_values"] = n_p_config["p_values"]
    config["N_values"] = n_p_config["N_values"]
    config["simulation_func"] = simulation_wrapper

    results = N_P_variation_simulation(config)
    m7_results_list, m8_results_list = sort_m7_m8_results(results)

    save_path = pathlib.Path(f"/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Partial_Information_Decomposition/Idep_Simulations/figures/Delta_method/{exp_name}_figures")
    save_path.mkdir(parents=True, exist_ok=True)

    for title_prefix, result_group in [("M7", m7_results_list), ("M8", m8_results_list)]:
        names = ["Mutual Information", "numerator", "numerator joint", "numerator target", "denominator"]
        slugs = ["mi", "nume", "joint", "target", "deno"]
        for name, slug, result in zip(names, slugs, result_group):
            plot_heatmap_mean_std(result, title=f"{title_prefix} {name} - {exp_name}", save_path=save_path)

    with open(save_path / f"{exp_name}_config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "simulation": "M7/M8 non-whitened MI with second-order M7 numerator correction",
                "seed": config["seed"],
                "N_samples_values": config["N_values"],
                "p_values": config["p_values"],
                "p_scale": config["p_scale"],
                "q_scale": config["q_scale"],
                "r_scale": config["r_scale"],
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print("Finished simulation.")
