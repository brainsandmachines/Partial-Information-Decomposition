"""Gaussian suppression generators selectable by name."""

from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pid.calculation import pid_calc
from simulations.commonality import commonality_analysis


SELECTED_EXAMPLE = "full_suppression"
RUN_CONFIG = {
    "n_samples": 10000000,
    "p": 3,
    "noise_std": 1.0,
    "seed": 0,
    "bias_correction": True,
    "commonality_method": "standard",
}


def full_suppression(
    rng: np.random.Generator,
    n: int,
    p: int,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the full-suppression example.

    Inputs:
        rng: np.random.Generator controlling every random draw.
        n: int number of aligned samples.
        p: int dimension of each source and target.
        noise_std: float standard deviation multiplier for noise terms.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray] containing X1, X2, and T,
        each with shape (n, p). X2 is a suppressor with no target signal.
    """
    target_noise = noise_std * rng.standard_normal((n, p))
    x1_noise = noise_std * rng.standard_normal((n, p))
    shared_noise = noise_std * rng.standard_normal((n, p))
    target_signal = rng.standard_normal((n, p))
    observed_target = target_signal + target_noise
    x1 = observed_target + x1_noise + shared_noise
    x2 = shared_noise
    # Four (n, p) latent/noise arrays -> X1, X2, T arrays with shape (n, p).
    return x1, x2, observed_target


def unq12_zero(
    rng: np.random.Generator,
    n: int,
    p: int,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a suppression example with zero unique information in both sources.

    Inputs:
        rng: np.random.Generator controlling every random draw.
        n: int number of aligned samples.
        p: int dimension of each source and target.
        noise_std: float standard deviation multiplier for noise terms.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray] containing X1, X2, and T,
        each with shape (n, p).
    """
    redundant = rng.standard_normal((n, p))
    shared_noise = noise_std * rng.standard_normal((n, p))
    x1_noise = noise_std * rng.standard_normal((n, p))
    x2_noise = noise_std * rng.standard_normal((n, p))
    target_noise = noise_std * rng.standard_normal((n, p))
    target = redundant + target_noise
    x1 = redundant + noise_std * shared_noise + x1_noise
    x2 = redundant + noise_std * shared_noise + x2_noise
    # Five (n, p) latent/noise arrays -> X1, X2, T arrays with shape (n, p).
    return x1, x2, target


def unq2_zero(
    rng: np.random.Generator,
    n: int,
    p: int,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a suppression example with zero source-2 unique information.

    Inputs:
        rng: np.random.Generator controlling every random draw.
        n: int number of aligned samples.
        p: int dimension of each source and target.
        noise_std: float standard deviation multiplier for noise terms.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray] containing X1, X2, and T,
        each with shape (n, p).
    """
    redundant = rng.standard_normal((n, p))
    unique_x1 = rng.standard_normal((n, p))
    shared_noise = noise_std * rng.standard_normal((n, p))
    target = redundant + unique_x1 + noise_std * rng.standard_normal((n, p))
    x1 = redundant + unique_x1 + shared_noise
    x2 = redundant + shared_noise
    # Four (n, p) latent/noise arrays -> X1, X2, T arrays with shape (n, p).
    return x1, x2, target


SUPPRESSION_EXAMPLES: dict[
    str,
    Callable[[np.random.Generator, int, int, float], tuple[np.ndarray, np.ndarray, np.ndarray]],
] = {
    "full_suppression": full_suppression,
    "unq12_zero": unq12_zero,
    "unq2_zero": unq2_zero,
}


def run_suppression_example(
    example: str,
    *,
    n_samples: int,
    p: int,
    noise_std: float,
    seed: int,
    bias_correction: bool = False,
    commonality_method: str = "standard",
    commonality_alphas=None,
    n_resamples: int = 20,
    n_shuffles: int = 20,
) -> dict:
    """Run Eigen-PID and commonality on the same suppression samples.

    Inputs:
        example: str key from ``SUPPRESSION_EXAMPLES``.
        n_samples: int number of aligned observations.
        p: int dimension of each source and target.
        noise_std: float standard deviation multiplier for noise terms.
        seed: int seed controlling sample generation and PID correction.
        bias_correction: bool, whether Eigen-PID uses combined bias correction.
        commonality_method: str, ``standard`` or ``ridge_cv``.
        commonality_alphas: array-like or None, penalties for ridge_cv.
        n_resamples: int number of Eigen-PID bootstrap resamples.
        n_shuffles: int number of Eigen-PID permutations.

    Outputs:
        dict containing the example name, Eigen-PID atoms, mutual information,
        and commonality terms calculated from exactly the same samples.
    """
    if example not in SUPPRESSION_EXAMPLES:
        raise ValueError(
            f"Unknown suppression example {example!r}; choose from "
            f"{sorted(SUPPRESSION_EXAMPLES)}."
        )
    if n_samples <= 0 or p <= 0:
        raise ValueError("n_samples and p must be positive.")

    rng = np.random.default_rng(seed)
    x1, x2, target = SUPPRESSION_EXAMPLES[example](rng, n_samples, p, noise_std)
    sources = [torch.from_numpy(x1), torch.from_numpy(x2)]
    target_list = [torch.from_numpy(target)]
    # Three NumPy (n_samples, p) arrays -> two source and one target torch tensors.
    pid, mi = pid_calc(
        config={
            "bias_correction": bias_correction,
            "bias_seed": seed,
            "n_resamples": n_resamples,
            "n_shuffles": n_shuffles,
        },
        sources=sources,
        target=target_list,
    )
    commonality = commonality_analysis(
        x1,
        x2,
        target,
        method=commonality_method,
        alphas=commonality_alphas,
    )
    return {
        "example": example,
        "pid": pid,
        "mi": mi,
        "commonality": commonality,
    }


if __name__ == "__main__":
    pid_components = ['red','unq1','unq2','syn']
    results = run_suppression_example(SELECTED_EXAMPLE, **RUN_CONFIG)
    print(f"Suppression example {SELECTED_EXAMPLE!r} results:")
    for key, value in results.items():
        if key != 'example':
            print(f"{key}")
            for subkey, subvalue in value.items():
                if key == 'pid' and subkey in pid_components:
                    print(f"  {subkey}: {subvalue:.6f}")
                else:
                    print(f"  {subkey}: {subvalue}")
