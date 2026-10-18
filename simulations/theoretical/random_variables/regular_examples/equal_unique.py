"""Regular examples with balanced unique information."""

from pathlib import Path
import sys

import numpy as np

STORY_ROOT = Path(__file__).resolve().parents[1]
if str(STORY_ROOT) not in sys.path:
    sys.path.append(str(STORY_ROOT))

from story_pid_utils import load_story_config, save_single_example, truth_pid_equal_unique


def equal_unique(rng, n, p, noise_std):
    """Generate a Gaussian example with equal unique information in both sources.

    Inputs:
        rng: np.random.Generator, random number generator.
        n: int, number of samples.
        p: int, dimension of each source and target component.
        noise_std: float, standard deviation multiplier for noise terms.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray], arrays (X1, X2, T) with shape (n, p).
    """
    unique_x1 = rng.standard_normal((n, p))
    unique_x2 = rng.standard_normal((n, p))
    x1_noise = noise_std * rng.standard_normal((n, p))
    x2_noise = noise_std * rng.standard_normal((n, p))
    target_noise = noise_std * rng.standard_normal((n, p))

    target = unique_x1 + unique_x2 + target_noise
    x1 = unique_x1 + x1_noise
    x2 = unique_x2 + x2_noise
    return x1, x2, target


def equal_unique2(rng, n, p, noise_std, snr=1):
    """Generate a higher-dimensional equal-unique example from real features.

    Inputs:
        rng: np.random.Generator, random number generator.
        n: int, number of samples.
        p: int, source block dimension.
        noise_std: float, unused compatibility argument.
        snr: float, signal-to-noise ratio used for generated target noise.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray], arrays (X1, X2, T) where X1
        and X2 have shape (n, 2*p) and T has shape (n, 2*p).
    """
    real_features = rng.standard_normal((n, 2 * p))
    betas = rng.standard_normal((2 * p, 2 * p))
    signal = real_features @ betas
    derived_noise_std = np.std(signal) / snr

    target = signal + derived_noise_std * rng.standard_normal((n, 2 * p))
    x1_noise = derived_noise_std * rng.standard_normal((n, p))
    x2_noise = derived_noise_std * rng.standard_normal((n, p))
    x1 = np.hstack([real_features[:, :p], x1_noise])
    x2 = np.hstack([x2_noise, real_features[:, p : 2 * p]])
    return x1, x2, target


if __name__ == "__main__":
    config = load_story_config()
    save_single_example(config, equal_unique, "p=1_equal_unique_zeroRed_structure.png", truth_func=truth_pid_equal_unique)
