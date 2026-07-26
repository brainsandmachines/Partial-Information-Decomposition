"""Compare RAW, PCA, and Ridge-CV PID routes on the all-above-zero example."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Simulations.PCA_Ridge.pid_feature_middleware import expand_independent_covariance, run_pid_feature_comparison
from Simulations.Theoretical_Examples.RVs_Story.regular_examples.All_above_zero import all_above_zero_weighted


def build_all_above_zero_covariance(
    p: int,
    noise_std: float,
    unique1_weight: float,
    unique2_weight: float,
    redundant_weight: float,
    shared_noise_weight: float,
) -> torch.Tensor:
    """Build the full all-above-zero covariance in grouped variable order.

    Inputs:
        p: int number of independent coordinates in each random variable.
        noise_std: float standard deviation of every independent noise term.
        unique1_weight: float weight of the X1-specific latent signal.
        unique2_weight: float weight of the X2-specific latent signal.
        redundant_weight: float weight of the latent signal shared with T.
        shared_noise_weight: float weight of source-only shared noise.

    Outputs:
        torch.Tensor: float64 covariance with shape (3*p, 3*p), ordered
        [X1, X2, T].
    """
    source_noise = noise_std**2 * (1 + shared_noise_weight**2)
    coordinate_covariance = torch.tensor([
        [
            unique1_weight**2 + redundant_weight**2 + source_noise,
            redundant_weight**2 + (shared_noise_weight * noise_std) ** 2,
            unique1_weight**2 + redundant_weight**2,
        ],
        [
            redundant_weight**2 + (shared_noise_weight * noise_std) ** 2,
            unique2_weight**2 + redundant_weight**2 + source_noise,
            unique2_weight**2 + redundant_weight**2,
        ],
        [
            unique1_weight**2 + redundant_weight**2,
            unique2_weight**2 + redundant_weight**2,
            unique1_weight**2 + unique2_weight**2 + redundant_weight**2 + noise_std**2,
        ],
    ], dtype=torch.float64)  # construction scalars -> (3, 3), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (3, 3) -> (3*p, 3*p)


if __name__ == "__main__":
    n_samples, n_train, n_components, p = 10000, 9000, 5, 70
    n_trials, base_seed, noise_std = 2, 0, 1.0
    unique1_weight, unique2_weight = 6.0, 7.0
    redundant_weight, shared_noise_weight = 1.0, 1.0
    pid_method, bias_correction = "tilde", True

    population_covariance = build_all_above_zero_covariance(
        p,
        noise_std,
        unique1_weight,
        unique2_weight,
        redundant_weight,
        shared_noise_weight,
    )  # (3, 3) coordinate construction -> (3*p, 3*p)
    run_pid_feature_comparison(
        lambda seed: all_above_zero_weighted(
            np.random.default_rng(seed),
            n_samples,
            p,
            noise_std,
            unique1_weight,
            unique2_weight,
            redundant_weight,
            shared_noise_weight,
        ),
        population_covariance,
        [p, p, p],
        n_samples=n_samples,
        n_train=n_train,
        n_components=n_components,
        n_trials=n_trials,
        base_seed=base_seed,
        pid_method=pid_method,
        bias_correction=bias_correction,
        experiment_name="ALL ABOVE ZERO",
        plot_path=PROJECT_ROOT / "Simulations/PCA_Ridge/results" / f"all_above_zero_pid_feature_comparison_{n_trials}_trials.png",
        plot_title="All Above Zero PID: RAW vs PCA vs Ridge CV",
        metadata={"p": p},
    )
