"""Compare RAW, PCA, and Ridge-CV PID routes when source-2 unique is zero."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Simulations.PCA_Ridge.pid_feature_middleware import expand_independent_covariance, run_pid_feature_comparison
from Simulations.Theoretical_Examples.RVs_Story.suppresion_examples.unq2_zero import unq2_zero


def build_unq2_zero_covariance(
    p: int,
    noise_std: float,
) -> torch.Tensor:
    """Build the full zero-source-2-unique covariance in [X1, X2, T].

    Inputs:
        p: int number of independent coordinates in each random variable.
        noise_std: float standard deviation of shared and target noise.

    Outputs:
        torch.Tensor: float64 covariance with shape (3*p, 3*p), ordered
        [X1, X2, T].
    """
    noise_variance = noise_std**2
    coordinate_covariance = torch.tensor([
        [2 + noise_variance, 1 + noise_variance, 2],
        [1 + noise_variance, 1 + noise_variance, 1],
        [2, 1, 2 + noise_variance],
    ], dtype=torch.float64)  # construction scalars -> (3, 3), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (3, 3) -> (3*p, 3*p)


if __name__ == "__main__":
    n_samples, n_train, n_components, p = 10000, 9000, 30, 70
    n_trials, base_seed, noise_std = 2, 0, 1.0
    pid_method, bias_correction = "tilde", True

    population_covariance = build_unq2_zero_covariance(
        p, noise_std,
    )  # (3, 3) coordinate construction -> (3*p, 3*p)
    run_pid_feature_comparison(
        lambda seed: unq2_zero(
            np.random.default_rng(seed), n_samples, p, noise_std,
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
        experiment_name="SOURCE-2 UNIQUE ZERO",
        plot_path=PROJECT_ROOT / "Simulations/PCA_Ridge/results" / f"unq2_zero_pid_feature_comparison_{n_trials}_trials.png",
        plot_title="Source-2 Unique Zero PID: RAW vs PCA vs Ridge CV",
        metadata={"p": p},
    )
