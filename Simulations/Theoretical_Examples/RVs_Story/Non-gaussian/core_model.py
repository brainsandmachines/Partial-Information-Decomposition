"""Compatibility wrapper for non-Gaussian RVs_Story examples."""

from pathlib import Path
import sys

STORY_ROOT = Path(__file__).resolve().parents[1]
if str(STORY_ROOT) not in sys.path:
    sys.path.append(str(STORY_ROOT))

from story_pid_utils import run_pid_story


def main_func(config, function_to_run):
    """Run a non-Gaussian RV generator without a Gaussian truth row.

    Inputs:
        config: dict, simulation and PID configuration values.
        function_to_run: callable, RV generator accepting (rng, n, p, noise_std).

    Outputs:
        dict, PID results keyed by method name.
    """
    return run_pid_story(config, function_to_run, truth_func=None)
