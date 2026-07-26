"""Shared PCA, Ridge-CV, theoretical-PID, and trial-loop helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Partial_Information_Decomposition.PID_calc import pid_calc
from Partial_Information_Decomposition.mi_functions import calculate_mi_raw
from pipeline.pipeline_phases.preprocessing_layer import ridge_train_to_test_prediction
from pipeline.pipeline_phases.report_results import print_pid_mi
from Simulations.evil_twin.covariance_example import evil_twin_example_torch
from Simulations.Theoretical_Examples.Covariance.cov_functions import change_covariance_order
from Simulations.Theoretical_Examples.Covariance.save_results import save_sample_simulation_results_table

RESULT_KEYS = ("red", "unq1", "unq2", "syn", "bi_mi_1", "bi_mi_2", "tri_mi")
COVARIANCE_PID_METHODS = ("tilde", "thin", "thin_pid")


def pca_target(
    target: Any,
    shared_mask: np.ndarray,
    n_components_target: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit target PCA and return its scores and population linear map.

    Inputs:
        target: Any array-like target with shape (N, Dt).
        shared_mask: np.ndarray Boolean mask selecting held-out rows.
        n_components_target: int number of retained target components.
        random_state: int seed for randomized PCA.

    Outputs:
        tuple[np.ndarray, np.ndarray]: projected target with shape (N, Ct) and
        PCA component matrix with shape (Ct, Dt).
    """
    target = np.asarray(target)
    shared_mask = np.asarray(shared_mask, dtype=bool)
    if shared_mask.ndim != 1 or target.shape[0] != shared_mask.shape[0] or not shared_mask.any() or shared_mask.all():
        raise ValueError("shared_mask must select test rows and leave training rows.")
    pca = PCA(n_components=int(n_components_target), svd_solver="randomized", random_state=random_state)
    target_train = target[~shared_mask]  # (N, Dt) -> (n_train, Dt)
    pca.fit(target_train)
    projected_target = pca.transform(target)  # (N, Dt) -> (N, n_components_target)
    target_map = np.asarray(pca.components_, dtype=np.float64)  # (Ct, Dt) -> (Ct, Dt)
    return projected_target, target_map  # (N, Dt), fitted PCA -> (N, Ct), (Ct, Dt)


def pca_sources(
    source_1: Any,
    source_2: Any,
    target: Any,
    shared_mask: np.ndarray,
    n_components_source_1: int,
    n_components_source_2: int,
    random_state: int = 0,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Fit source PCAs and return held-out arrays and population linear maps.

    Inputs:
        source_1: Any array-like source with shape (N, D1).
        source_2: Any array-like source with shape (N, D2).
        target: Any array-like target with shape (N, Dt).
        shared_mask: np.ndarray Boolean mask selecting held-out rows.
        n_components_source_1: int retained components for source 1.
        n_components_source_2: int retained components for source 2.
        random_state: int seed shared by both randomized PCA fits.

    Outputs:
        tuple containing the three held-out PID arrays and the two fitted PCA
        component matrices with shapes (C1, D1) and (C2, D2).
    """
    source_1, source_2, target = map(np.asarray, (source_1, source_2, target))
    shared_mask = np.asarray(shared_mask, dtype=bool)
    if len({source_1.shape[0], source_2.shape[0], target.shape[0], shared_mask.shape[0]}) != 1:
        raise ValueError("Sources, target, and shared_mask must have matching rows.")
    pca_1 = PCA(n_components=int(n_components_source_1), svd_solver="randomized", random_state=random_state)
    pca_2 = PCA(n_components=int(n_components_source_2), svd_solver="randomized", random_state=random_state)
    source_1_train = source_1[~shared_mask]  # (N, D1) -> (n_train, D1)
    source_2_train = source_2[~shared_mask]  # (N, D2) -> (n_train, D2)
    pca_1.fit(source_1_train)
    pca_2.fit(source_2_train)
    projected_source_1 = pca_1.transform(source_1[shared_mask])  # (N, D1) -> (n_test, C1)
    projected_source_2 = pca_2.transform(source_2[shared_mask])  # (N, D2) -> (n_test, C2)
    source_1_map = np.asarray(pca_1.components_, dtype=np.float64)  # (C1, D1) -> (C1, D1)
    source_2_map = np.asarray(pca_2.components_, dtype=np.float64)  # (C2, D2) -> (C2, D2)
    arrays = projected_source_1, projected_source_2, target[shared_mask]  # three (N, D*) arrays -> (n_test, C1), (n_test, C2), (n_test, Dt)
    return arrays, (source_1_map, source_2_map)  # fitted arrays/maps -> three PID arrays and two (C*, D*) maps


def _ridge_prediction_and_map(
    source_train: np.ndarray,
    target_train: np.ndarray,
    source_test: np.ndarray,
    target_test: np.ndarray,
    **ridge_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return existing Ridge-CV predictions and their effective RAW-input map.

    Inputs:
        source_train: np.ndarray with shape (n_train, Ds).
        target_train: np.ndarray with shape (n_train, Dt).
        source_test: np.ndarray with shape (n_test, Ds).
        target_test: np.ndarray with shape (n_test, Dt).
        ridge_kwargs: keyword arguments passed to ridge_train_to_test_prediction.

    Outputs:
        tuple[np.ndarray, np.ndarray]: held-out prediction with shape
        (n_test, Dt) and effective linear map with shape (Dt, Ds).
    """
    prediction, info = ridge_train_to_test_prediction(
        source_train,
        target_train,
        source_test,
        target_test,
        **ridge_kwargs,
    )  # train/test arrays -> held-out prediction and scalar diagnostics
    prediction = np.asarray(prediction).reshape(source_test.shape[0], target_train.shape[1])  # sklearn output -> (n_test, Dt)
    fitted_pipeline = make_pipeline(
        StandardScaler(),
        Ridge(alpha=info["best_alpha"], fit_intercept=True, solver="svd"),
    )
    fitted_pipeline.fit(source_train, target_train)
    reproduced_prediction = fitted_pipeline.predict(source_test).reshape(prediction.shape)  # sklearn output -> (n_test, Dt)
    if not np.allclose(reproduced_prediction, prediction, rtol=1e-12, atol=1e-12):
        raise RuntimeError("The fitted Ridge map does not reproduce the existing helper predictions.")
    scaler = fitted_pipeline.named_steps["standardscaler"]
    ridge = fitted_pipeline.named_steps["ridge"]
    coefficients = np.asarray(ridge.coef_, dtype=np.float64).reshape(target_train.shape[1], source_train.shape[1])  # sklearn coefficients -> (Dt, Ds)
    effective_map = coefficients / np.asarray(scaler.scale_)[None, :]  # standardized-input (Dt, Ds) -> RAW-input (Dt, Ds)
    return prediction, effective_map  # fitted pipeline -> (n_test, Dt), (Dt, Ds)


def ridge_sources_on_target(
    source_1: Any,
    source_2: Any,
    target: Any,
    shared_mask: np.ndarray,
    **ridge_kwargs: Any,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Fit two Ridge-CV models and return held-out arrays and linear maps.

    Inputs:
        source_1: Any array-like source with shape (N, D1).
        source_2: Any array-like source with shape (N, D2).
        target: Any array-like target with shape (N, Dt).
        shared_mask: np.ndarray Boolean mask selecting held-out rows.
        ridge_kwargs: keyword arguments for ridge_train_to_test_prediction.

    Outputs:
        tuple containing the three held-out PID arrays and two effective
        RAW-source-to-target maps with shapes (Dt, D1) and (Dt, D2).
    """
    source_1, source_2, target = map(np.asarray, (source_1, source_2, target))
    shared_mask = np.asarray(shared_mask, dtype=bool)
    if len({source_1.shape[0], source_2.shape[0], target.shape[0], shared_mask.shape[0]}) != 1:
        raise ValueError("Sources, target, and shared_mask must have matching rows.")
    prediction_1, source_1_map = _ridge_prediction_and_map(
        source_1[~shared_mask], target[~shared_mask], source_1[shared_mask], target[shared_mask], **ridge_kwargs,
    )  # (n_train, D1), (n_train, Dt), (n_test, D1) -> (n_test, Dt)
    prediction_2, source_2_map = _ridge_prediction_and_map(
        source_2[~shared_mask], target[~shared_mask], source_2[shared_mask], target[shared_mask], **ridge_kwargs,
    )  # (n_train, D2), (n_train, Dt), (n_test, D2) -> (n_test, Dt)
    arrays = prediction_1, prediction_2, target[shared_mask]  # (N, Dt) -> three (n_test, Dt) arrays
    return arrays, (source_1_map, source_2_map)  # fitted arrays/maps -> three PID arrays and two (Dt, D*) maps


def expand_independent_covariance(
    coordinate_covariance: torch.Tensor,
    n_replicas: int,
) -> torch.Tensor:
    """Expand independent coordinate covariance into grouped variable order.

    Inputs:
        coordinate_covariance: torch.Tensor with shape (D0, D0), ordered by
            coordinate-level blocks inside [X1, X2, T].
        n_replicas: int number of independent coordinates per scalar block.

    Outputs:
        torch.Tensor: covariance with shape (D0*n_replicas, D0*n_replicas),
        ordered as the grouped generated arrays [X1, X2, T].
    """
    coordinate_covariance = torch.as_tensor(coordinate_covariance, dtype=torch.float64)  # (D0, D0) -> (D0, D0)
    if coordinate_covariance.ndim != 2 or coordinate_covariance.shape[0] != coordinate_covariance.shape[1]:
        raise ValueError("coordinate_covariance must be square.")
    if n_replicas < 1:
        raise ValueError("n_replicas must be at least 1.")
    identity = torch.eye(n_replicas, dtype=coordinate_covariance.dtype, device=coordinate_covariance.device)  # scalar size -> (n_replicas, n_replicas)
    return torch.kron(coordinate_covariance, identity)  # (D0, D0), (p, p) -> (D0*p, D0*p)


def transform_population_covariance(
    covariance: torch.Tensor,
    linear_maps: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> torch.Tensor:
    """Propagate [X1, X2, T] covariance through three fitted linear maps.

    Inputs:
        covariance: torch.Tensor population covariance with shape (D, D).
        linear_maps: tuple of maps shaped (C1, D1), (C2, D2), and (Ct, Dt).

    Outputs:
        torch.Tensor: transformed covariance with shape
        (C1+C2+Ct, C1+C2+Ct), still ordered [X1, X2, T].
    """
    covariance = torch.as_tensor(covariance, dtype=torch.float64)  # (D, D) -> (D, D)
    maps = tuple(torch.as_tensor(linear_map, dtype=torch.float64) for linear_map in linear_maps)  # three (C*, D*) arrays -> three (C*, D*) tensors
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square.")
    if any(linear_map.ndim != 2 for linear_map in maps):
        raise ValueError("Every linear map must be two-dimensional.")
    if sum(linear_map.shape[1] for linear_map in maps) != covariance.shape[0]:
        raise ValueError("Linear-map input dimensions must match the covariance blocks.")
    block_map = torch.block_diag(*maps)  # three (C*, D*) maps -> (sum(C*), sum(D*))
    transformed = block_map @ covariance @ block_map.T  # (sum(C*), D), (D, D) -> (sum(C*), sum(C*))
    return (transformed + transformed.T) / 2  # (sum(C*), sum(C*)) -> symmetric (sum(C*), sum(C*))


def calculate_theoretical_pid(
    covariance: torch.Tensor,
    dims: list[int],
    method: str,
) -> dict[str, float]:
    """Calculate exact Gaussian PID and MI values from a population covariance.

    Inputs:
        covariance: torch.Tensor ordered as [X1, X2, T].
        dims: list[int] containing [dim_X1, dim_X2, dim_T].
        method: str covariance-compatible PID method; configured simulations
            use Tilde or Thin-PID.

    Outputs:
        dict[str, float]: red, unq1, unq2, syn, and the three MI values in bits.
    """
    covariance = torch.as_tensor(covariance, dtype=torch.float64)  # (D, D) -> (D, D)
    if len(dims) != 3 or covariance.shape != (sum(dims), sum(dims)):
        raise ValueError("covariance and dims must describe [X1, X2, T].")
    if method not in COVARIANCE_PID_METHODS:
        raise ValueError(f"Theoretical covariance PID supports {COVARIANCE_PID_METHODS}; received {method!r}.")
    mi_nats = calculate_mi_raw(covariance.device, covariance, dims)
    ground_truth = {
        "bi_mi_1": float(mi_nats["bi_mi_1_t"] / np.log(2)),
        "bi_mi_2": float(mi_nats["bi_mi_2_t"] / np.log(2)),
        "tri_mi": float(mi_nats["tri_mi"] / np.log(2)),
    }
    pid_dtype = torch.float32 if method in ("thin", "thin_pid") else torch.float64
    pid_covariance = change_covariance_order(covariance, [2, 0, 1], dims).to(pid_dtype)  # [X1, X2, T] -> [T, X1, X2]
    source_1 = torch.zeros((2, dims[0]), dtype=pid_dtype)  # scalar placeholder -> (2, dim_X1)
    source_2 = torch.zeros((2, dims[1]), dtype=pid_dtype)  # scalar placeholder -> (2, dim_X2)
    target = torch.zeros((2, dims[2]), dtype=pid_dtype)  # scalar placeholder -> (2, dim_T)
    pid, _ = pid_calc(
        config={"bias_correction": False, "n_samples": 2},
        sources=[source_1, source_2],
        target=[target],
        covariance=pid_covariance,
        method=method,
    )
    ground_truth.update({key: float(value) for key, value in pid.items()})
    return ground_truth


def prepare_pid_routes(
    source_1: Any,
    source_2: Any,
    target: Any,
    shared_mask: np.ndarray,
    n_components: int,
    seed: int,
    population_covariance: torch.Tensor,
    population_dims: list[int],
) -> dict[
    str,
    tuple[tuple[np.ndarray, np.ndarray, np.ndarray], torch.Tensor],
]:
    """Prepare held-out PID arrays and exact route population covariances.

    Inputs:
        source_1: Any source array with shape (N, D1).
        source_2: Any source array with shape (N, D2).
        target: Any target array with shape (N, Dt).
        shared_mask: np.ndarray Boolean mask selecting test rows.
        n_components: int components retained by target and source PCA.
        seed: int shared PCA and ridge random state.
        population_covariance: torch.Tensor full RAW covariance in [X1, X2, T].
        population_dims: list[int] containing RAW [dim_X1, dim_X2, dim_T].

    Outputs:
        dict mapping each route to its three held-out arrays and exact
        population covariance conditional on the fitted training transformation.
    """
    source_1, source_2, target = map(np.asarray, (source_1, source_2, target))  # three (N, D*) inputs -> three NumPy (N, D*) arrays
    observed_dims = [source_1.shape[1], source_2.shape[1], target.shape[1]]
    if observed_dims != population_dims:
        raise ValueError(f"Generated dimensions {observed_dims} do not match population_dims {population_dims}.")
    if torch.as_tensor(population_covariance).shape != (sum(population_dims), sum(population_dims)):
        raise ValueError("population_covariance shape does not match population_dims.")
    target_pca, target_map = pca_target(target, shared_mask, n_components, seed)  # (N, Dt) -> (N, Ct), (Ct, Dt)
    raw = source_1[shared_mask], source_2[shared_mask], target[shared_mask]  # three (N, D*) arrays -> three (n_test, D*) arrays
    pca, source_pca_maps = pca_sources(
        source_1, source_2, target_pca, shared_mask, n_components, n_components, seed,
    )  # three (N, D*) arrays -> three held-out arrays and two PCA maps
    ridge, source_ridge_maps = ridge_sources_on_target(
        source_1,
        source_2,
        target_pca,
        shared_mask,
        alphas=np.logspace(-2, 20, 100),
        inner_cv=5,
        random_state=seed,
    )  # three (N, D*) arrays -> three held-out arrays and two Ridge maps
    pca_covariance = transform_population_covariance(
        population_covariance, (*source_pca_maps, target_map),
    )  # full RAW covariance -> (3*n_components, 3*n_components)
    ridge_covariance = transform_population_covariance(
        population_covariance, (*source_ridge_maps, target_map),
    )  # full RAW covariance -> (3*n_components, 3*n_components)
    return {  # three route array triples and covariances -> route-preparation dictionary
        "RAW": (raw, torch.as_tensor(population_covariance, dtype=torch.float64)),
        "PCA": (pca, pca_covariance),
        "RIDGE CV": (ridge, ridge_covariance),
    }


def run_pid_feature_comparison(
    sample_for_seed: Callable[[int], tuple[Any, Any, Any]],
    population_covariance: torch.Tensor,
    population_dims: list[int],
    *,
    n_samples: int,
    n_train: int,
    n_components: int,
    n_trials: int,
    base_seed: int,
    pid_method: str,
    bias_correction: bool,
    experiment_name: str,
    plot_path: str | Path,
    plot_title: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run a seeded RAW/PCA/Ridge-CV PID comparison and save its table plot.

    Inputs:
        sample_for_seed: callable mapping an integer seed to (X1, X2, T).
        population_covariance: exact full RAW covariance ordered [X1, X2, T].
        population_dims: list[int] containing RAW [dim_X1, dim_X2, dim_T].
        n_samples: int total aligned samples generated per trial.
        n_train: int rows used to fit PCA and ridge.
        n_components: int retained PCA target/source dimensions.
        n_trials: int number of consecutive seeds to run.
        base_seed: int first trial seed; trial k uses base_seed + k.
        pid_method: str configured covariance method: Tilde or Thin-PID.
        bias_correction: bool passed unchanged to sampled PID calls.
        experiment_name: str label printed for each route.
        plot_path: str | Path output image path.
        plot_title: str title rendered above the result table.
        metadata: optional dict added to the plot legend configuration.

    Outputs:
        dict containing mean route-specific theoretical and sampled values,
        paired bias, paired-error variance, and paired MSE for each route.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    routes = ("RAW", "PCA", "RIDGE CV")
    raw_theoretical = calculate_theoretical_pid(population_covariance, population_dims, pid_method)
    sampled_values = {route: {key: [] for key in RESULT_KEYS} for route in routes}
    theoretical_values = {route: {key: [] for key in RESULT_KEYS} for route in routes}
    for trial in range(n_trials):
        trial_seed = base_seed + trial
        print(f"\n{'#' * 18} TRIAL {trial + 1}/{n_trials} — seed={trial_seed} {'#' * 18}")
        source_1, source_2, target = sample_for_seed(trial_seed)
        shared_mask = np.zeros(n_samples, dtype=bool)  # n_samples rows -> (n_samples,) test mask
        shared_mask[np.random.default_rng(trial_seed + 1).permutation(n_samples)[n_train:]] = True
        prepared_routes = prepare_pid_routes(
            source_1,
            source_2,
            target,
            shared_mask,
            n_components,
            trial_seed + 1,
            population_covariance,
            population_dims,
        )  # three generated arrays and RAW covariance -> three route arrays/covariances
        for route, (arrays, route_covariance) in prepared_routes.items():
            route_dims = [np.asarray(array).shape[1] for array in arrays]
            route_theoretical = (
                raw_theoretical
                if route == "RAW"
                else calculate_theoretical_pid(route_covariance, route_dims, pid_method)
            )
            for key in RESULT_KEYS:
                theoretical_values[route][key].append(float(route_theoretical[key]))
            pid_source_1, pid_source_2, pid_target = (
                torch.as_tensor(array, dtype=torch.float64) for array in arrays
            )  # three (n_test, D*) arrays -> three float64 (n_test, D*) tensors
            pid, mi = pid_calc(
                config={"bias_correction": bias_correction, "n_samples": pid_target.shape[0]},
                sources=[pid_source_1, pid_source_2],
                target=[pid_target],
                method=pid_method,
            )
            for key, value in {**pid, **mi}.items():
                sampled_values[route][key].append(float(value))

    plot_results = {}
    for route in routes:
        theoretical = {key: float(np.mean(theoretical_values[route][key])) for key in RESULT_KEYS}
        mean_sampled = {key: float(np.mean(sampled_values[route][key])) for key in RESULT_KEYS}
        errors = {
            key: np.asarray(sampled_values[route][key]) - np.asarray(theoretical_values[route][key])
            for key in RESULT_KEYS
        }  # two dictionaries of trial scalars -> paired (n_trials,) error arrays
        bias = {key: float(np.mean(errors[key])) for key in RESULT_KEYS}
        variance = {key: float(np.var(errors[key])) for key in RESULT_KEYS}
        plot_results[route] = {
            "theoretical": theoretical,
            "mean_sampled": mean_sampled,
            "bias": bias,
            "variance": variance,
            "mse": {key: float(np.mean(errors[key] ** 2)) for key in RESULT_KEYS},
        }
        print(f"\n{'=' * 10} {experiment_name} — {route} — MEAN OF {n_trials} TRIALS {'=' * 10}")
        print_pid_mi(
            {key: mean_sampled[key] for key in ("red", "unq1", "unq2", "syn")},
            {key: mean_sampled[key] for key in ("tri_mi", "bi_mi_1", "bi_mi_2")},
        )

    plot_config = {
        "n": n_samples - n_train,
        "seed": base_seed,
        "bias_correction": bias_correction,
        "n_trials": n_trials,
        **(metadata or {}),
    }
    saved_path = save_sample_simulation_results_table(
        plot_results, plot_config, plot_path, title=plot_title,
    )
    print(f"\nSaved comparison plot: {saved_path}")
    return plot_results


def build_sonic_covariance(p: int) -> torch.Tensor:
    """Build the full Sonic covariance in grouped [X1, X2, T] order.

    Inputs:
        p: int generated feature count used to scale per-coordinate variance.

    Outputs:
        torch.Tensor: float64 covariance with shape (3*p, 3*p).
    """
    if p < 1:
        raise ValueError("p must be at least 1.")
    coordinate_covariance = torch.tensor(
        [[5.5, 3.0, 3.0], [3.0, 3.5, 1.0], [3.0, 1.0, 4.5]],
        dtype=torch.float64,
    ) / p  # construction scalars -> (3, 3), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (3, 3) -> (3*p, 3*p)


if __name__ == "__main__":
    n_samples, n_train, n_components, p = 10000, 9000, 10, 70
    n_trials, base_seed = 2, 0
    pid_method, bias_correction = "tilde", False
    population_covariance = build_sonic_covariance(p)  # (3, 3) construction -> (3*p, 3*p)
    run_pid_feature_comparison(
        lambda seed: evil_twin_example_torch(
            torch.Generator().manual_seed(seed), n=n_samples, p=p,
        )["sonic"],
        population_covariance,
        [p, p, p],
        n_samples=n_samples,
        n_train=n_train,
        n_components=n_components,
        n_trials=n_trials,
        base_seed=base_seed,
        pid_method=pid_method,
        bias_correction=bias_correction,
        experiment_name="SONIC",
        plot_path=PROJECT_ROOT / "Simulations/PCA_Ridge/results" / f"sonic_pid_feature_comparison_{n_trials}_trials.png",
        plot_title="Sonic PID: RAW vs PCA vs Ridge CV",
    )
