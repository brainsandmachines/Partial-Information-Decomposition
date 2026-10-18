"""Full suppression Gaussian RV example."""

from pathlib import Path
import sys

import numpy as np

STORY_ROOT = Path(__file__).resolve().parents[1]
if str(STORY_ROOT) not in sys.path:
    sys.path.append(str(STORY_ROOT))

from story_pid_utils import load_story_config, save_single_example, truth_pid_suppression


def full_suppresion(rng, n, p, noise_std):
    """Generate the full suppression example.

    Inputs:
        rng: np.random.Generator, random number generator.
        n: int, number of samples.
        p: int, dimension of each random variable.
        noise_std: float, standard deviation multiplier for noise terms.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray], arrays (X1, X2, T) with shape (n, p).
    """
    n_t = noise_std * rng.standard_normal((n, p))
    n_x1 = noise_std * rng.standard_normal((n, p))
    n_shared = noise_std * rng.standard_normal((n, p))
    target = rng.standard_normal((n, p))

    t = target + n_t
    x1 = t + n_x1 + n_shared
    x2 = n_shared
    return x1, x2, t


if __name__ == "__main__":
    config = load_story_config()
    save_single_example(config, full_suppresion, "full_suppresion.png", truth_func=truth_pid_suppression)
