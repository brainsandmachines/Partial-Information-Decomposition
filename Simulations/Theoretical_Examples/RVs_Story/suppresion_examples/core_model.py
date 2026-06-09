"""Compatibility wrapper for suppression RVs_Story examples."""

from pathlib import Path
import sys

STORY_ROOT = Path(__file__).resolve().parents[1]
if str(STORY_ROOT) not in sys.path:
    sys.path.append(str(STORY_ROOT))

from story_pid_utils import run_pid_story, truth_pid_suppression


def true_mi_pid(sources, target, covariance=None):
    """Return the suppression-style true PID row.

    Inputs:
        sources: list[torch.Tensor], two source tensors ordered as [X1, X2].
        target: list[torch.Tensor], one target tensor ordered as [T].
        covariance: optional covariance input, kept for backward compatibility.

    Outputs:
        tuple[dict, dict], PID component dictionary and MI dictionary.
    """
    return truth_pid_suppression(sources, target, covariance=covariance)


def main_func(config, function_to_run):
    """Run a suppression RV generator through the shared PID story runner.

    Inputs:
        config: dict, simulation and PID configuration values.
        function_to_run: callable, RV generator accepting (rng, n, p, noise_std).

    Outputs:
        dict, PID results keyed by method name.
    """
    return run_pid_story(config, function_to_run, truth_func=truth_pid_suppression)
