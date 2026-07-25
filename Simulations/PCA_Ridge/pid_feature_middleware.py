"""Compact PCA/ridge middleware for PID simulations."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Partial_Information_Decomposition.PID_calc import mi_wishart_bias, pid_calc
from Partial_Information_Decomposition.mi_functions import calculate_mi_raw
from pipeline.pipeline_phases.preprocessing_layer import ridge_train_to_test_prediction
from pipeline.pipeline_phases.report_results import print_pid_mi
from Simulations.evil_twin.covariance_example import evil_twin_example_torch
from Simulations.Theoretical_Examples.Covariance.save_results import save_sample_simulation_results_table


def pca_target(source_1: Any, source_2: Any, target: Any, shared_mask: np.ndarray, n_components_target: int, random_state: int = 0) -> tuple[Any, Any, np.ndarray]:
    """Fit target PCA on training rows and transform every aligned target row.

    Inputs: three aligned arrays, Boolean test mask, integer component count, and seed.
    Output: `(source_1, source_2, projected_target)` with all rows preserved.
    """
    target, shared_mask = np.asarray(target), np.asarray(shared_mask, dtype=bool)
    if shared_mask.ndim != 1 or target.shape[0] != shared_mask.shape[0] or not shared_mask.any() or shared_mask.all():
        raise ValueError("shared_mask must select test rows and leave training rows.")
    pca = PCA(n_components=int(n_components_target), svd_solver="randomized", random_state=random_state)
    pca.fit(target[~shared_mask])
    projected_target = pca.transform(target)  # (N, Dt) -> (N, n_components_target)
    return source_1, source_2, projected_target  # (N, D1), (N, D2), (N, Dt) -> (N, D1), (N, D2), (N, n_components_target)


def pca_sources(source_1: Any, source_2: Any, target: Any, shared_mask: np.ndarray, n_components_source_1: int, n_components_source_2: int, random_state: int = 0) -> tuple[np.ndarray, np.ndarray, Any]:
    """Fit both source PCAs on training rows and transform only held-out rows.

    Inputs: three aligned arrays, Boolean test mask, two component counts, and seed.
    Output: held-out `(projected_source_1, projected_source_2, target)` PID arrays.
    """
    source_1, source_2 = np.asarray(source_1), np.asarray(source_2)
    shared_mask = np.asarray(shared_mask, dtype=bool)
    if len({source_1.shape[0], source_2.shape[0], target.shape[0], shared_mask.shape[0]}) != 1:
        raise ValueError("Sources, target, and shared_mask must have matching rows.")
    pca_1 = PCA(n_components=int(n_components_source_1), svd_solver="randomized", random_state=random_state)
    pca_2 = PCA(n_components=int(n_components_source_2), svd_solver="randomized", random_state=random_state)
    pca_1.fit(source_1[~shared_mask])
    pca_2.fit(source_2[~shared_mask])
    projected_source_1 = pca_1.transform(source_1[shared_mask])  # (N, D1) -> (n_test, n_components_source_1)
    projected_source_2 = pca_2.transform(source_2[shared_mask])  # (N, D2) -> (n_test, n_components_source_2)
    return projected_source_1, projected_source_2, np.asarray(target)[shared_mask]  # (N, D1), (N, D2), (N, Dt) -> (n_test, C1), (n_test, C2), (n_test, Dt)


def ridge_sources_on_target(source_1: Any, source_2: Any, target: Any, shared_mask: np.ndarray, **ridge_kwargs: Any) -> tuple[Any, Any, np.ndarray]:
    """Cross-validate ridge on training rows and predict only held-out rows.

    Inputs: three aligned arrays, Boolean test mask, and pipeline ridge CV keyword arguments.
    Output: held-out `(prediction_1, prediction_2, target_test)` arrays.
    """
    source_1, source_2, target = np.asarray(source_1), np.asarray(source_2), np.asarray(target)
    shared_mask = np.asarray(shared_mask, dtype=bool)
    if len({source_1.shape[0], source_2.shape[0], target.shape[0], shared_mask.shape[0]}) != 1:
        raise ValueError("Sources, target, and shared_mask must have matching rows.")
    prediction_1, _ = ridge_train_to_test_prediction(source_1[~shared_mask], target[~shared_mask], source_1[shared_mask], target[shared_mask], **ridge_kwargs)  # train rows -> (n_test, Dt)
    prediction_2, _ = ridge_train_to_test_prediction(source_2[~shared_mask], target[~shared_mask], source_2[shared_mask], target[shared_mask], **ridge_kwargs)  # train rows -> (n_test, Dt)
    return prediction_1, prediction_2, target[shared_mask]  # (N, Dt) -> three (n_test, Dt) arrays


def calculate_covariance_cmi(covariance: Any, dims: list[int], n_samples: int | None = None) -> dict[str, float]:
    """Calculate Gaussian CMI in bits from covariance ordered as [X1, X2, T].

    Inputs: covariance tensor/array, three integer dimensions, and optional
    sample count; providing `n_samples` applies the exact Wishart MI biases.
    Output: dict containing I(T;X2|X1) and I(T;X1|X2), both in bits.
    """
    covariance = torch.as_tensor(covariance, dtype=torch.float64)  # (D, D) -> (D, D)
    if len(dims) != 3 or covariance.shape != (sum(dims), sum(dims)):
        raise ValueError("dims and covariance must describe [X1, X2, T].")
    mi = calculate_mi_raw(covariance.device, covariance, dims)
    bias = mi_wishart_bias(dims, n_samples) if n_samples is not None else {}
    tri = (mi["tri_mi"] - bias.get("bias_tri_mi", 0.0)) / np.log(2)
    bi_1 = (mi["bi_mi_1_t"] - bias.get("bias_mi_1_t", 0.0)) / np.log(2)
    bi_2 = (mi["bi_mi_2_t"] - bias.get("bias_mi_2_t", 0.0)) / np.log(2)
    return {"cmi_x2_given_x1": float(tri - bi_1), "cmi_x1_given_x2": float(tri - bi_2)}


if __name__ == "__main__":
    n_samples, n_train, n_components, p = 10000, 9000, 30, 70
    n_trials, base_seed = 2, 0
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    pid_method, bias_correction, cmi_tolerance = "tilde", False, 1e-3
    component_keys = ("red", "unq1", "unq2", "syn", "bi_mi_1", "bi_mi_2", "tri_mi")
    cmi_keys = ("cmi_x2_given_x1", "cmi_x1_given_x2")
    route_dimensions = {"RAW": p, "PCA": n_components, "RIDGE CV": n_components}
    sonic_covariance = torch.tensor(
        [[5.5, 3.0, 3.0], [3.0, 3.5, 1.0], [3.0, 1.0, 4.5]], dtype=torch.float64,
    ) / p  # (3, 3) -> (3, 3), one Sonic dimension ordered [X1, X2, T]
    ground_truth_nats = calculate_mi_raw(torch.device("cpu"), sonic_covariance, [1, 1, 1])
    ground_truth_per_dimension = {
        "bi_mi_1": ground_truth_nats["bi_mi_1_t"] / np.log(2),
        "bi_mi_2": ground_truth_nats["bi_mi_2_t"] / np.log(2),
        "tri_mi": ground_truth_nats["tri_mi"] / np.log(2),
    }
    covariance_order = torch.tensor([2, 0, 1])
    pid_covariance = sonic_covariance[covariance_order][:, covariance_order]  # [X1, X2, T] -> [T, X1, X2]
    dummy = torch.zeros((2, 1), dtype=torch.float64)  # scalar placeholder -> (2, 1)
    ground_truth_pid, _ = pid_calc(
        config={"bias_correction": False, "n_samples": 2},
        sources=[dummy, dummy],
        target=[dummy],
        covariance=pid_covariance,
        method=pid_method,
    )
    ground_truth_per_dimension.update(ground_truth_pid)
    ground_truth_cmi_per_dimension = calculate_covariance_cmi(sonic_covariance, [1, 1, 1])

    trial_values = {
        route: {key: [] for key in component_keys}
        for route in route_dimensions
    }
    trial_cmi_values = {route: {key: [] for key in cmi_keys} for route in route_dimensions}
    for trial in range(n_trials):
        trial_seed = base_seed + trial
        print(f"\n{'#' * 18} TRIAL {trial + 1}/{n_trials} — seed={trial_seed} {'#' * 18}")
        raw_source1, raw_source2, raw_target = evil_twin_example_torch(
            torch.Generator().manual_seed(trial_seed), n=n_samples, p=p,
        )["sonic"]  # dict of three (N, p) arrays -> three Sonic (N, p) arrays
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
        mse = {key: bias[key] ** 2 + variance[key] for key in component_keys}
        plot_results[route] = {
            "theoretical": theoretical,
            "mean_sampled": mean_sampled,
            "bias": bias,
            "variance": variance,
            "mse": mse,
        }
        print(f"\n{'=' * 15} SONIC — {route} — MEAN OF {n_trials} TRIALS {'=' * 15}")
        print_pid_mi(
            {key: mean_sampled[key] for key in ("red", "unq1", "unq2", "syn")},
            {key: mean_sampled[key] for key in ("tri_mi", "bi_mi_1", "bi_mi_2")},
        )
        print(f"MI comparison against exact {dimensions}-D Sonic covariance (bits):")
        print(f"  {'Quantity':<14} {'Ground truth':>14} {'Estimated':>14} {'Absolute error':>16}")
        for label, key in (("I(X1; T)", "bi_mi_1"), ("I(X2; T)", "bi_mi_2"), ("I(X1,X2; T)", "tri_mi")):
            print(f"  {label:<14} {theoretical[key]:>14.6f} {mean_sampled[key]:>14.6f} {abs(bias[key]):>16.6f}")

    plot_path = PROJECT_ROOT / "Simulations" / "PCA_Ridge" / "results" / f"sonic_pid_feature_comparison_{n_trials}_trials.png"
    save_sample_simulation_results_table(
        plot_results,
        {"n": n_samples - n_train, "seed": base_seed, "bias_correction": bias_correction, "n_trials": n_trials, "cmi_tolerance": cmi_tolerance},
        plot_path,
        title="Sonic PID: RAW vs PCA vs Ridge CV",
    )
    print(f"\nSaved comparison plot: {plot_path}")
