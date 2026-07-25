"""Compare RAW, PCA, and Ridge-CV PID routes on the equal-unique example."""

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
from Simulations.Theoretical_Examples.RVs_Story.regular_examples.equal_unique import equal_unique


if __name__ == "__main__":
    n_samples, n_train, n_components, p = 10000, 9000, 30, 70
    n_trials, base_seed, noise_std = 2, 0, 1.0
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    pid_method, bias_correction, cmi_tolerance = "thin", False, 1e-3
    component_keys = ("red", "unq1", "unq2", "syn", "bi_mi_1", "bi_mi_2", "tri_mi")
    cmi_keys = ("cmi_x2_given_x1", "cmi_x1_given_x2")
    route_dimensions = {"RAW": p, "PCA": n_components, "RIDGE CV": n_components}

    equal_unique_covariance = torch.tensor(
        [[1 + noise_std**2, 0.0, 1.0], [0.0, 1 + noise_std**2, 1.0], [1.0, 1.0, 2 + noise_std**2]],
        dtype=torch.float64,
    )  # construction scalars -> (3, 3), ordered [X1, X2, T]
    ground_truth_nats = calculate_mi_raw(torch.device("cpu"), equal_unique_covariance, [1, 1, 1])
    ground_truth_per_dimension = {
        "bi_mi_1": ground_truth_nats["bi_mi_1_t"] / np.log(2),
        "bi_mi_2": ground_truth_nats["bi_mi_2_t"] / np.log(2),
        "tri_mi": ground_truth_nats["tri_mi"] / np.log(2),
    }
    covariance_order = torch.tensor([2, 0, 1])
    pid_dtype = torch.float32 if pid_method in ("thin", "thin_pid") else torch.float64
    pid_covariance = equal_unique_covariance[covariance_order][:, covariance_order].to(pid_dtype)  # [X1, X2, T] -> [T, X1, X2]
    dummy = torch.zeros((2, 1), dtype=pid_dtype)  # scalar placeholder -> (2, 1)
    ground_truth_pid, _ = pid_calc(
        config={"bias_correction": False, "n_samples": 2},
        sources=[dummy, dummy],
        target=[dummy],
        covariance=pid_covariance,
        method=pid_method,
    )
    ground_truth_per_dimension.update(ground_truth_pid)
    ground_truth_cmi_per_dimension = calculate_covariance_cmi(equal_unique_covariance, [1, 1, 1])

    trial_values = {route: {key: [] for key in component_keys} for route in route_dimensions}
    trial_cmi_values = {route: {key: [] for key in cmi_keys} for route in route_dimensions}
    for trial in range(n_trials):
        trial_seed = base_seed + trial
        print(f"\n{'#' * 18} TRIAL {trial + 1}/{n_trials} — seed={trial_seed} {'#' * 18}")
        raw_source1, raw_source2, raw_target = equal_unique(
            np.random.default_rng(trial_seed), n_samples, p, noise_std,
        )  # generator inputs -> three (N, p) arrays
        shared_mask = np.zeros(n_samples, dtype=bool)  # n_samples rows -> (n_samples,) train/test mask
        shared_mask[np.random.default_rng(trial_seed + 1).permutation(n_samples)[n_train:]] = True
        _, _, target_pca = pca_target(
            raw_source1, raw_source2, raw_target, shared_mask, n_components, random_state=trial_seed + 1,
        )
        raw_inputs = raw_source1[shared_mask], raw_source2[shared_mask], raw_target[shared_mask]  # three (N, p) arrays -> three (n_test, p) arrays
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
    for route, dimensions in route_dimensions.items():
        theoretical = {key: value * dimensions for key, value in ground_truth_per_dimension.items()}
        theoretical_cmi = {key: value * dimensions for key, value in ground_truth_cmi_per_dimension.items()}
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
        print(f"\n{'=' * 12} EQUAL UNIQUE — {route} — MEAN OF {n_trials} TRIALS {'=' * 12}")
        print_pid_mi(
            {key: mean_sampled[key] for key in ("red", "unq1", "unq2", "syn")},
            {key: mean_sampled[key] for key in ("tri_mi", "bi_mi_1", "bi_mi_2")},
        )

    plot_path = PROJECT_ROOT / "Simulations" / "PCA_Ridge" / "results" / f"equal_unique_pid_feature_comparison_{n_trials}_trials.png"
    save_sample_simulation_results_table(
        plot_results,
        {"n": n_samples - n_train, "p": p, "seed": base_seed, "bias_correction": bias_correction, "n_trials": n_trials, "cmi_tolerance": cmi_tolerance},
        plot_path,
        title="Equal Unique PID: RAW vs PCA vs Ridge CV",
    )
    print(f"\nSaved comparison plot: {plot_path}")
