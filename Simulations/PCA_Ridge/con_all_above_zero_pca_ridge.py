"""Compare RAW, PCA, and Ridge-CV PID routes on the concatenated all-above-zero example."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Partial_Information_Decomposition.PID_calc import pid_calc
from Partial_Information_Decomposition.mi_functions import calculate_mi_raw
from pipeline.pipeline_phases.report_results import print_pid_mi
from Simulations.PCA_Ridge.pid_feature_middleware import calculate_covariance_cmi, pca_sources, pca_target, ridge_sources_on_target
from Simulations.Theoretical_Examples.Covariance.save_results import save_sample_simulation_results_table
from Simulations.Theoretical_Examples.RVs_Story.regular_examples.All_above_zero import con_all_above_zero_weighted


if __name__ == "__main__":
    n_samples, n_train, n_components, p = 10000, 9000, 30, 70
    n_trials, base_seed, noise_std = 2, 0, 1.0
    redundant_weight, shared_noise_weight = 1.0, 1.0
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if n_components % 3 != 0:
        raise ValueError("n_components must be divisible by the three concatenated blocks.")
    pid_method, bias_correction, cmi_tolerance = "tilde", False, 1e-3
    component_keys = ("red", "unq1", "unq2", "syn", "bi_mi_1", "bi_mi_2", "tri_mi")
    cmi_keys = ("cmi_x2_given_x1", "cmi_x1_given_x2")
    route_multipliers = {"RAW": p, "PCA": n_components / 3, "RIDGE CV": n_components / 3}

    shared_redundant_variance = redundant_weight**2 * (
        1 + (shared_noise_weight * noise_std) ** 2
    )
    redundant_source_variance = shared_redundant_variance + noise_std**2
    covariance_x1 = torch.tensor([
        [1 + noise_std**2, 0.0, 0.0],
        [0.0, noise_std**2, 0.0],
        [0.0, 0.0, redundant_source_variance],
    ], dtype=torch.float64)  # construction scalars -> (3, 3), ordered [U1, U2, R]
    covariance_x2 = torch.tensor([
        [noise_std**2, 0.0, 0.0],
        [0.0, 1 + noise_std**2, 0.0],
        [0.0, 0.0, redundant_source_variance],
    ], dtype=torch.float64)  # construction scalars -> (3, 3), ordered [U1, U2, R]
    covariance_target = torch.tensor([
        [1 + noise_std**2, 0.0, 0.0],
        [0.0, 1 + noise_std**2, 0.0],
        [0.0, 0.0, 1 + noise_std**2],
    ], dtype=torch.float64)  # construction scalars -> (3, 3), ordered [U1, U2, R]
    covariance_x1_x2 = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, shared_redundant_variance],
    ], dtype=torch.float64)  # construction scalars -> (3, 3)
    covariance_x1_target = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, redundant_weight],
    ], dtype=torch.float64)  # construction scalars -> (3, 3)
    covariance_x2_target = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, redundant_weight],
    ], dtype=torch.float64)  # construction scalars -> (3, 3)

    covariance_x1_row = torch.cat(
        (covariance_x1, covariance_x1_x2, covariance_x1_target), dim=1,
    )  # three (3, 3) blocks -> (3, 9)
    covariance_x2_row = torch.cat(
        (covariance_x1_x2, covariance_x2, covariance_x2_target), dim=1,
    )  # three (3, 3) blocks -> (3, 9)
    covariance_target_row = torch.cat(
        (covariance_x1_target, covariance_x2_target, covariance_target), dim=1,
    )  # three (3, 3) blocks -> (3, 9)
    concatenated_covariance = torch.cat(
        (covariance_x1_row, covariance_x2_row, covariance_target_row), dim=0,
    )  # three (3, 9) block rows -> (9, 9), ordered [X1, X2, T]

    ground_truth_nats = calculate_mi_raw(
        torch.device("cpu"), concatenated_covariance, [3, 3, 3],
    )
    ground_truth_per_block = {
        "bi_mi_1": ground_truth_nats["bi_mi_1_t"] / np.log(2),
        "bi_mi_2": ground_truth_nats["bi_mi_2_t"] / np.log(2),
        "tri_mi": ground_truth_nats["tri_mi"] / np.log(2),
    }
    covariance_order = torch.tensor([6, 7, 8, 0, 1, 2, 3, 4, 5])  # scalar indices -> (9,)
    pid_dtype = torch.float32 if pid_method in ("thin", "thin_pid") else torch.float64
    pid_covariance = concatenated_covariance[covariance_order][:, covariance_order].to(
        pid_dtype,
    )  # [X1, X2, T] with three dimensions each -> [T, X1, X2] with three dimensions each
    dummy = torch.zeros((2, 3), dtype=pid_dtype)  # scalar placeholder -> (2, 3)
    ground_truth_pid, _ = pid_calc(
        config={"bias_correction": False, "n_samples": 2},
        sources=[dummy, dummy],
        target=[dummy],
        covariance=pid_covariance,
        method=pid_method,
    )
    ground_truth_per_block.update(ground_truth_pid)
    ground_truth_cmi_per_block = calculate_covariance_cmi(
        concatenated_covariance, [3, 3, 3],
    )

    trial_values = {route: {key: [] for key in component_keys} for route in route_multipliers}
    trial_cmi_values = {route: {key: [] for key in cmi_keys} for route in route_multipliers}
    for trial in range(n_trials):
        trial_seed = base_seed + trial
        print(f"\n{'#' * 18} TRIAL {trial + 1}/{n_trials} — seed={trial_seed} {'#' * 18}")
        raw_source1, raw_source2, raw_target = con_all_above_zero_weighted(
            np.random.default_rng(trial_seed),
            n_samples,
            p,
            noise_std,
            redundant_weight=redundant_weight,
            shared_noise_weight=shared_noise_weight,
        )  # generator inputs -> three (N, 3 * p) arrays
        shared_mask = np.zeros(n_samples, dtype=bool)  # n_samples rows -> (n_samples,) train/test mask
        shared_mask[np.random.default_rng(trial_seed + 1).permutation(n_samples)[n_train:]] = True
        _, _, target_pca = pca_target(
            raw_source1, raw_source2, raw_target, shared_mask, n_components, random_state=trial_seed + 1,
        )
        raw_inputs = raw_source1[shared_mask], raw_source2[shared_mask], raw_target[shared_mask]  # three (N, 3 * p) arrays -> three (n_test, 3 * p) arrays
        pca_inputs = pca_sources(
            raw_source1, raw_source2, target_pca, shared_mask, n_components, n_components, random_state=trial_seed + 1,
        )
        ridge_inputs = ridge_sources_on_target(
            raw_source1, raw_source2, target_pca, shared_mask,
            alphas=np.logspace(-2, 2, 9), inner_cv=5, random_state=trial_seed + 1,
        )
        for route, (source1, source2, target) in {"RAW": raw_inputs, "PCA": pca_inputs, "RIDGE CV": ridge_inputs}.items():
            source1 = torch.as_tensor(source1, dtype=torch.float64)  # (n_test, D1) -> (n_test, D1)
            source2 = torch.as_tensor(source2, dtype=torch.float64)  # (n_test, D2) -> (n_test, D2)
            target = torch.as_tensor(target, dtype=torch.float64)  # (n_test, Dt) -> (n_test, Dt)
            joint_samples = torch.cat((source1, source2, target), dim=1)  # three (n_test, D*) arrays -> (n_test, D1 + D2 + Dt)
            sample_covariance = torch.cov(joint_samples.T)  # (n_test, D1 + D2 + Dt) -> (D1 + D2 + Dt, D1 + D2 + Dt)
            sample_cmi = calculate_covariance_cmi(
                sample_covariance,
                [source1.shape[1], source2.shape[1], target.shape[1]],
                n_samples=target.shape[0],
            )
            pid, mi = pid_calc(
                config={"bias_correction": bias_correction, "n_samples": target.shape[0]},
                sources=[source1, source2],
                target=[target],
                method=pid_method,
            )
            for key, value in {**pid, **mi}.items():
                trial_values[route][key].append(float(value))
            for key, value in sample_cmi.items():
                trial_cmi_values[route][key].append(value)

    plot_results = {}
    for route, multiplier in route_multipliers.items():
        theoretical = {key: value * multiplier for key, value in ground_truth_per_block.items()}
        theoretical_cmi = {key: value * multiplier for key, value in ground_truth_cmi_per_block.items()}
        mean_sampled = {key: float(np.mean(trial_values[route][key])) for key in component_keys}
        mean_cmi = {key: float(np.mean(trial_cmi_values[route][key])) for key in cmi_keys}
        theoretical["cmi_x2_given_x1_test"] = f"{'PASS' if np.isclose(theoretical['unq2'] + theoretical['syn'], theoretical_cmi['cmi_x2_given_x1'], atol=cmi_tolerance, rtol=0) else 'FAIL'} | CMI={theoretical_cmi['cmi_x2_given_x1']:.4f}"
        theoretical["cmi_x1_given_x2_test"] = f"{'PASS' if np.isclose(theoretical['unq1'] + theoretical['syn'], theoretical_cmi['cmi_x1_given_x2'], atol=cmi_tolerance, rtol=0) else 'FAIL'} | CMI={theoretical_cmi['cmi_x1_given_x2']:.4f}"
        mean_sampled["cmi_x2_given_x1_test"] = f"{'PASS' if np.allclose(np.asarray(trial_values[route]['unq2']) + np.asarray(trial_values[route]['syn']), trial_cmi_values[route]['cmi_x2_given_x1'], atol=cmi_tolerance, rtol=0) else 'FAIL'} | CMI={mean_cmi['cmi_x2_given_x1']:.4f}"
        mean_sampled["cmi_x1_given_x2_test"] = f"{'PASS' if np.allclose(np.asarray(trial_values[route]['unq1']) + np.asarray(trial_values[route]['syn']), trial_cmi_values[route]['cmi_x1_given_x2'], atol=cmi_tolerance, rtol=0) else 'FAIL'} | CMI={mean_cmi['cmi_x1_given_x2']:.4f}"
        bias = {key: mean_sampled[key] - theoretical[key] for key in component_keys}
        variance = {key: float(np.var(trial_values[route][key])) for key in component_keys}
        plot_results[route] = {
            "theoretical": theoretical,
            "mean_sampled": mean_sampled,
            "bias": bias,
            "variance": variance,
            "mse": {key: bias[key] ** 2 + variance[key] for key in component_keys},
        }
        print(f"\n{'=' * 8} CONCATENATED ALL ABOVE ZERO — {route} — MEAN OF {n_trials} TRIALS {'=' * 8}")
        print_pid_mi(
            {key: mean_sampled[key] for key in ("red", "unq1", "unq2", "syn")},
            {key: mean_sampled[key] for key in ("tri_mi", "bi_mi_1", "bi_mi_2")},
        )

    plot_path = PROJECT_ROOT / "Simulations" / "PCA_Ridge" / "results" / f"con_all_above_zero_pid_feature_comparison_{n_trials}_trials.png"
    save_sample_simulation_results_table(
        plot_results,
        {
            "n": n_samples - n_train,
            "p": 3 * p,
            "block_p": p,
            "seed": base_seed,
            "bias_correction": bias_correction,
            "n_trials": n_trials,
            "cmi_tolerance": cmi_tolerance,
        },
        plot_path,
        title="Concatenated All Above Zero PID: RAW vs PCA vs Ridge CV",
    )
    print(f"\nSaved comparison plot: {plot_path}")
