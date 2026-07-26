"""Compare RAW, PCA, and Ridge-CV PID routes on concatenated equal-unique blocks."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Simulations.PCA_Ridge.pid_feature_middleware import expand_independent_covariance, run_pid_feature_comparison


def concatenated_equal_unique(
    rng: np.random.Generator,
    n: int,
    p: int,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate equal unique information in separate concatenated target blocks.

    Inputs:
        rng: np.random.Generator controlling every generated latent and noise.
        n: int number of aligned samples.
        p: int dimension of each U1 and U2 block.
        noise_std: float standard deviation of each independent noise term.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray]: X1, X2, and T, each with
        shape (n, 2*p) and block order [U1, U2].
    """
    unique_1 = rng.standard_normal((n, p))  # scalar dimensions -> (n, p)
    unique_2 = rng.standard_normal((n, p))  # scalar dimensions -> (n, p)
    zeros = np.zeros_like(unique_1)  # (n, p) -> (n, p)
    target_signal = np.hstack((unique_1, unique_2))  # two (n, p) blocks -> (n, 2*p)
    source_1_signal = np.hstack((unique_1, zeros))  # two (n, p) blocks -> (n, 2*p)
    source_2_signal = np.hstack((zeros, unique_2))  # two (n, p) blocks -> (n, 2*p)
    target = target_signal + noise_std * rng.standard_normal(target_signal.shape)  # (n, 2*p) -> (n, 2*p)
    source_1 = source_1_signal + noise_std * rng.standard_normal(source_1_signal.shape)  # (n, 2*p) -> (n, 2*p)
    source_2 = source_2_signal + noise_std * rng.standard_normal(source_2_signal.shape)  # (n, 2*p) -> (n, 2*p)
    return source_1, source_2, target  # three (n, 2*p) arrays -> three (n, 2*p) arrays


def build_concatenated_equal_unique_covariance(
    p: int,
    noise_std: float,
) -> torch.Tensor:
    """Build the full concatenated equal-unique covariance in [X1, X2, T].

    Inputs:
        p: int number of independent coordinates in each U1 and U2 block.
        noise_std: float standard deviation of each independent noise term.

    Outputs:
        torch.Tensor: float64 covariance with shape (6*p, 6*p), with every
        random variable internally ordered [U1, U2].
    """
    total_variance = 1 + noise_std**2
    noise_variance = noise_std**2
    coordinate_covariance = torch.tensor([
        [total_variance, 0, 0, 0, 1, 0],
        [0, noise_variance, 0, 0, 0, 0],
        [0, 0, noise_variance, 0, 0, 0],
        [0, 0, 0, total_variance, 0, 1],
        [1, 0, 0, 0, total_variance, 0],
        [0, 0, 0, 1, 0, total_variance],
    ], dtype=torch.float64)  # construction scalars -> (6, 6), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (6, 6) -> (6*p, 6*p)


if __name__ == "__main__":
    n_samples, n_train, n_components, p = 10000, 9000, 30, 70
    n_trials, base_seed, noise_std = 2, 0, 1.0
    pid_method, bias_correction = "thin", True

    population_covariance = build_concatenated_equal_unique_covariance(
        p, noise_std,
    )  # (6, 6) coordinate construction -> (6*p, 6*p)
    run_pid_feature_comparison(
        lambda seed: concatenated_equal_unique(
            np.random.default_rng(seed), n_samples, p, noise_std,
        ),
        population_covariance,
        [2 * p, 2 * p, 2 * p],
        n_samples=n_samples,
        n_train=n_train,
        n_components=n_components,
        n_trials=n_trials,
        base_seed=base_seed,
        pid_method=pid_method,
        bias_correction=bias_correction,
        experiment_name="CONCATENATED EQUAL UNIQUE",
        plot_path=PROJECT_ROOT / "Simulations/PCA_Ridge/results" / f"con_equal_unique_pid_feature_comparison_{n_trials}_trials.png",
        plot_title="Concatenated Equal Unique PID: RAW vs PCA vs Ridge CV",
        metadata={"p": 2 * p, "block_p": p},
    )
