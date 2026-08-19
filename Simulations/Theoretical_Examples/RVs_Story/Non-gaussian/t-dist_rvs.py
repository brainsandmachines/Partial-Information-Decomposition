"""Student-t non-Gaussian RV examples."""

from pathlib import Path
import sys

import numpy as np

STORY_ROOT = Path(__file__).resolve().parents[1]
if str(STORY_ROOT) not in sys.path:
    sys.path.append(str(STORY_ROOT))

from story_pid_utils import load_story_config, save_single_example


def standardized_t(rng, df, size):
    """Sample a variance-one Student-t random variable.

    Inputs:
        rng: np.random.Generator, random number generator.
        df: float, Student-t degrees of freedom; must be greater than 2.
        size: tuple[int, ...], output sample shape.

    Outputs:
        np.ndarray, samples with approximately unit variance.
    """
    if df <= 2:
        raise ValueError("df must be > 2 so that the t-distribution has finite variance.")
    return rng.standard_t(df=df, size=size) / np.sqrt(df / (df - 2))


def unq2_zero_t(rng, n, p, noise_std, df=5):
    """Generate a Student-t version of the zero-source-2-unique example.

    Inputs:
        rng: np.random.Generator, random number generator.
        n: int, number of samples.
        p: int, dimension of each random variable.
        noise_std: float, standard deviation multiplier for noise terms.
        df: float, Student-t degrees of freedom; must be greater than 2.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray], arrays (X1, X2, T) with shape (n, p).
    """
    redundant = standardized_t(rng, df, size=(n, p))
    unique_x1 = standardized_t(rng, df, size=(n, p))
    shared_noise = noise_std * standardized_t(rng, df, size=(n, p))
    target_noise = noise_std * standardized_t(rng, df, size=(n, p))

    target = redundant + unique_x1 + target_noise
    x1 = redundant + unique_x1 + shared_noise
    x2 = redundant + shared_noise
    return x1, x2, target


def unq12_zero(rng, n, p, noise_std, df=5):
    """Generate a Student-t version of the zero-both-unique example.

    Inputs:
        rng: np.random.Generator, random number generator.
        n: int, number of samples.
        p: int, dimension of each random variable.
        noise_std: float, standard deviation multiplier for noise terms.
        df: float, Student-t degrees of freedom; must be greater than 2.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray], arrays (X1, X2, T) with shape (n, p).
    """
    redundant = standardized_t(rng, df, size=(n, p))
    shared_noise = noise_std * standardized_t(rng, df, size=(n, p))
    x1_noise = noise_std * standardized_t(rng, df, size=(n, p))
    x2_noise = noise_std * standardized_t(rng, df, size=(n, p))
    target_noise = noise_std * standardized_t(rng, df, size=(n, p))

    target = redundant + target_noise
    x1 = redundant + noise_std * shared_noise + x1_noise
    x2 = redundant + noise_std * shared_noise + x2_noise
    return x1, x2, target


if __name__ == "__main__":
    config = load_story_config()
    save_single_example(config, unq12_zero, "2.0unq12_zero_t.png", truth_func=None)
