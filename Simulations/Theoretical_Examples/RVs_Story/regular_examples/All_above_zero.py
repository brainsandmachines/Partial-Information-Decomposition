"""Regular examples with balanced unique information."""

from pathlib import Path
import sys

import numpy as np

STORY_ROOT = Path(__file__).resolve().parents[1]
if str(STORY_ROOT) not in sys.path:
    sys.path.append(str(STORY_ROOT))

from story_pid_utils import load_story_config, save_single_example, truth_pid_equal_unique


def all_above_zero_weighted(
    rng,
    n,
    p,
    noise_std,
    unique1_weight=5.0,
    unique2_weight=5.0,
    redundant_weight=1.0,
    shared_noise_weight=1.0,
):
    """
    Gaussian example where all PID components should be above zero,
    but unique information is emphasized.

    Components:
        U1: unique signal shared by X1 and target only
        U2: unique signal shared by X2 and target only
        R:  redundant signal shared by X1, X2, and target
        shared_noise: nuisance/suppressor shared by X1 and X2 but not target
    """

    unique_x1 = rng.standard_normal((n, p))
    unique_x2 = rng.standard_normal((n, p))
    R = rng.standard_normal((n, p))

    x1_noise = noise_std * rng.standard_normal((n, p))
    x2_noise = noise_std * rng.standard_normal((n, p))
    target_noise = noise_std * rng.standard_normal((n, p))

    shared_noise = noise_std * rng.standard_normal((n, p))

    target = (
        unique1_weight * unique_x1
        + unique2_weight * unique_x2
        + redundant_weight * R
        + target_noise
    )

    x1 = (
        unique1_weight * unique_x1
        + redundant_weight * R
        + x1_noise
        + shared_noise_weight * shared_noise
    )

    x2 = (
        unique2_weight * unique_x2
        + redundant_weight * R
        + x2_noise
        + shared_noise_weight * shared_noise
    )

    return x1, x2, target



if __name__ == "__main__":
    p_s = [10,20,30,50,100,150,200,500]
    for p in p_s:
        config = load_story_config()
        config["parameters"]["p"] = p
        print(f"Running all_above_zero_weighted with p={p}...")
        save_single_example(config, all_above_zero_weighted, f"weighted_unique_informality=5_p={p}_all_above_zero.png", truth_func=None)
