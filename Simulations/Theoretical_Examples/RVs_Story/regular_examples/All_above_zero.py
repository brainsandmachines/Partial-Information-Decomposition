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



def con_all_above_zero_weighted(
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
    Concatenated Gaussian example where all PID components should be above zero,
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

    shared_noise = noise_std * rng.standard_normal((n, p))

    target = np.hstack([unique_x1, unique_x2, R])

    target += noise_std * rng.standard_normal((target.shape[0], target.shape[1]))

    R += shared_noise_weight * shared_noise

    
    x1 = np.hstack([unique_x1, np.zeros_like(unique_x2), redundant_weight * R]) 
    x1 += noise_std * rng.standard_normal((x1.shape[0], x1.shape[1]))

    x2 = np.hstack([np.zeros_like(unique_x1), unique_x2, redundant_weight * R]) 
    x2 += noise_std * rng.standard_normal((x2.shape[0], x2.shape[1]))

    return x1, x2, target




if __name__ == "__main__":
    p_s = [100,150,200,250]
    for p in p_s:
        config = load_story_config()
        config["parameters"]["p"] = p
        print(f"Running con_all_above_zero_weighted with p={p}...")
        save_single_example(config, con_all_above_zero_weighted, f"con_p={p}_all_above_zero.png", truth_func=None)
