"""Suppression example where source 2 has zero unique information."""

from pathlib import Path
import sys

import numpy as np

STORY_ROOT = Path(__file__).resolve().parents[1]
if str(STORY_ROOT) not in sys.path:
    sys.path.append(str(STORY_ROOT))

from story_pid_utils import load_story_config, save_single_example, truth_pid_suppression


def unq2_zero(rng, n, p, noise_std):
    """Generate an example with zero source-2 unique information.

    Inputs:
        rng: np.random.Generator, random number generator.
        n: int, number of samples.
        p: int, dimension of each random variable.
        noise_std: float, standard deviation multiplier for noise terms.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray], arrays (X1, X2, T) with shape (n, p).
    """
    redundant = rng.standard_normal((n, p))
    unique_x1 = rng.standard_normal((n, p))
    shared_noise = noise_std * rng.standard_normal((n, p))
    target_noise = noise_std * rng.standard_normal((n, p))

    target = redundant + unique_x1 + target_noise
    x1 = redundant + unique_x1 + shared_noise
    x2 = redundant + shared_noise
    return x1, x2, target


if __name__ == "__main__":
    config = load_story_config()
    save_single_example(config, unq2_zero, "unq2_zero.png", truth_func=truth_pid_suppression)
    