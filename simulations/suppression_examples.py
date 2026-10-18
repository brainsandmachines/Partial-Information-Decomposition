"""Gaussian suppression generators selectable by name."""

from collections.abc import Callable

import numpy as np


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
