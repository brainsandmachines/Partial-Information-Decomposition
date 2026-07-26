"""Compare RAW, PCA, and Ridge-CV PID routes on the concatenated all-above-zero example."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Simulations.PCA_Ridge.pid_feature_middleware import expand_independent_covariance, run_pid_feature_comparison
from Simulations.Theoretical_Examples.RVs_Story.regular_examples.All_above_zero import con_all_above_zero_weighted


def build_concatenated_all_above_zero_covariance(
    p: int,
    noise_std: float,
    redundant_weight: float,
    shared_noise_weight: float,
) -> torch.Tensor:
    """Build the full concatenated covariance in grouped [X1, X2, T] order.

    Inputs:
        p: int number of independent coordinates in each U1, U2, and R block.
        noise_std: float standard deviation of each independent noise term.
        redundant_weight: float weight applied to the redundant source block.
        shared_noise_weight: float weight of source-only shared noise.

    Outputs:
        torch.Tensor: float64 covariance with shape (9*p, 9*p); each random
        variable contains its grouped [U1, U2, R] blocks.
    """
    source_shared = redundant_weight**2 * (1 + (shared_noise_weight * noise_std) ** 2)
    source_redundant = source_shared + noise_std**2
    coordinate_covariance = torch.tensor([
        [1 + noise_std**2, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, noise_std**2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, source_redundant, 0, 0, source_shared, 0, 0, redundant_weight],
        [0, 0, 0, noise_std**2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1 + noise_std**2, 0, 0, 1, 0],
        [0, 0, source_shared, 0, 0, source_redundant, 0, 0, redundant_weight],
        [1, 0, 0, 0, 0, 0, 1 + noise_std**2, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1 + noise_std**2, 0],
        [0, 0, redundant_weight, 0, 0, redundant_weight, 0, 0, 1 + noise_std**2],
    ], dtype=torch.float64)  # construction scalars -> (9, 9), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (9, 9) -> (9*p, 9*p)


if __name__ == "__main__":
    n_samples, n_train, n_components, p = 10000, 9000, 30, 70
    n_trials, base_seed, noise_std = 2, 0, 1.0
    redundant_weight, shared_noise_weight = 1.0, 1.0
    pid_method, bias_correction = "tilde", True
    population_covariance = build_concatenated_all_above_zero_covariance(
        p, noise_std, redundant_weight, shared_noise_weight,
    )  # (9, 9) coordinate construction -> (9*p, 9*p)
    run_pid_feature_comparison(
        lambda seed: con_all_above_zero_weighted(
            np.random.default_rng(seed),
            n_samples,
            p,
            noise_std,
            redundant_weight=redundant_weight,
            shared_noise_weight=shared_noise_weight,
        ),
        population_covariance,
        [3 * p, 3 * p, 3 * p],
        n_samples=n_samples,
        n_train=n_train,
        n_components=n_components,
        n_trials=n_trials,
        base_seed=base_seed,
        pid_method=pid_method,
        bias_correction=bias_correction,
        experiment_name="CONCATENATED ALL ABOVE ZERO",
        plot_path=PROJECT_ROOT / "Simulations/PCA_Ridge/results" / f"con_all_above_zero_pid_feature_comparison_{n_trials}_trials.png",
        plot_title="Concatenated All Above Zero PID: RAW vs PCA vs Ridge CV",
        metadata={"p": 3 * p, "block_p": p},
    )
