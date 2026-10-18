"""Sample generators shared by simulation scenarios."""

import numpy as np


def all_above_zero_weighted(
    rng: np.random.Generator,
    n: int,
    p: int,
    noise_std: float,
    unique1_weight: float = 5.0,
    unique2_weight: float = 5.0,
    redundant_weight: float = 1.0,
    shared_noise_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a Gaussian example in which every PID atom can be positive.

    Inputs:
        rng: np.random.Generator controlling every random draw.
        n: int number of aligned samples.
        p: int dimension of each source and target.
        noise_std: float standard deviation of independent and shared noise.
        unique1_weight: float weight of the X1-specific signal.
        unique2_weight: float weight of the X2-specific signal.
        redundant_weight: float weight of the signal shared by both sources.
        shared_noise_weight: float weight of source-only shared noise.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray] containing X1, X2, and T,
        each with shape (n, p).
    """
    unique_x1 = rng.standard_normal((n, p))
    unique_x2 = rng.standard_normal((n, p))
    redundant = rng.standard_normal((n, p))
    x1_noise = noise_std * rng.standard_normal((n, p))
    x2_noise = noise_std * rng.standard_normal((n, p))
    target_noise = noise_std * rng.standard_normal((n, p))
    shared_noise = noise_std * rng.standard_normal((n, p))
    x1 = (
        unique1_weight * unique_x1
        + redundant_weight * redundant
        + x1_noise
        + shared_noise_weight * shared_noise
    )
    x2 = (
        unique2_weight * unique_x2
        + redundant_weight * redundant
        + x2_noise
        + shared_noise_weight * shared_noise
    )
    target = (
        unique1_weight * unique_x1
        + unique2_weight * unique_x2
        + redundant_weight * redundant
        + target_noise
    )
    # Three (n, p) latent constructions -> X1, X2, T arrays with shape (n, p).
    return x1, x2, target


def con_all_above_zero_weighted(
    rng: np.random.Generator,
    n: int,
    p: int,
    noise_std: float,
    unique1_weight: float = 5.0,
    unique2_weight: float = 5.0,
    redundant_weight: float = 1.0,
    shared_noise_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate concatenated unique and redundant Gaussian blocks.

    Inputs:
        rng: np.random.Generator controlling every random draw.
        n: int number of aligned samples.
        p: int dimension of each latent block.
        noise_std: float standard deviation of independent and shared noise.
        unique1_weight: float retained compatibility argument; currently unused.
        unique2_weight: float retained compatibility argument; currently unused.
        redundant_weight: float weight of the redundant source block.
        shared_noise_weight: float weight of source-only shared noise.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray] containing X1, X2, and T,
        each with shape (n, 3*p) and block order [U1, U2, R].
    """
    del unique1_weight, unique2_weight
    unique_x1 = rng.standard_normal((n, p))
    unique_x2 = rng.standard_normal((n, p))
    redundant = rng.standard_normal((n, p))
    shared_noise = noise_std * rng.standard_normal((n, p))
    target = np.hstack((unique_x1, unique_x2, redundant))
    # Three (n, p) blocks -> target with shape (n, 3*p).
    target += noise_std * rng.standard_normal(target.shape)
    noisy_redundant = redundant + shared_noise_weight * shared_noise
    x1 = np.hstack((unique_x1, np.zeros_like(unique_x2), redundant_weight * noisy_redundant))
    # Three (n, p) blocks -> X1 with shape (n, 3*p).
    x1 += noise_std * rng.standard_normal(x1.shape)
    x2 = np.hstack((np.zeros_like(unique_x1), unique_x2, redundant_weight * noisy_redundant))
    # Three (n, p) blocks -> X2 with shape (n, 3*p).
    x2 += noise_std * rng.standard_normal(x2.shape)
    return x1, x2, target


def equal_unique(
    rng: np.random.Generator,
    n: int,
    p: int,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate independent source-specific signals with equal weights.

    Inputs:
        rng: np.random.Generator controlling every random draw.
        n: int number of aligned samples.
        p: int dimension of each source and target.
        noise_std: float standard deviation of independent noise.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray] containing X1, X2, and T,
        each with shape (n, p).
    """
    unique_x1 = rng.standard_normal((n, p))
    unique_x2 = rng.standard_normal((n, p))
    x1 = unique_x1 + noise_std * rng.standard_normal((n, p))
    x2 = unique_x2 + noise_std * rng.standard_normal((n, p))
    target = unique_x1 + unique_x2 + noise_std * rng.standard_normal((n, p))
    # Three (n, p) latent constructions -> X1, X2, T arrays with shape (n, p).
    return x1, x2, target


def concatenated_equal_unique(
    rng: np.random.Generator,
    n: int,
    p: int,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate equal unique information in separate target blocks.

    Inputs:
        rng: np.random.Generator controlling every random draw.
        n: int number of aligned samples.
        p: int dimension of each source-specific block.
        noise_std: float standard deviation of independent noise.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray] containing X1, X2, and T,
        each with shape (n, 2*p) and block order [U1, U2].
    """
    unique_x1 = rng.standard_normal((n, p))
    unique_x2 = rng.standard_normal((n, p))
    zeros = np.zeros_like(unique_x1)
    target_signal = np.hstack((unique_x1, unique_x2))
    source_1_signal = np.hstack((unique_x1, zeros))
    source_2_signal = np.hstack((zeros, unique_x2))
    # Two (n, p) blocks -> signal arrays with shape (n, 2*p).
    target = target_signal + noise_std * rng.standard_normal(target_signal.shape)
    source_1 = source_1_signal + noise_std * rng.standard_normal(source_1_signal.shape)
    source_2 = source_2_signal + noise_std * rng.standard_normal(source_2_signal.shape)
    return source_1, source_2, target
