"""Shared PID execution helpers for the RVs_Story examples."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import os
import sys

import numpy as np
import torch
import yaml

from story_math_utils import calculate_story_mi_values

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

STORY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STORY_ROOT.parents[2]
DEFAULT_CONFIG_PATH = STORY_ROOT.parent / "rv_config.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Partial_Information_Decomposition.PID_util import save_pid_comparison_table


def truth_pid_suppression(
    sources: list[torch.Tensor],
    target: list[torch.Tensor],
    covariance=None,
) -> tuple[dict, dict]:
    """Compute the Gaussian truth row for suppression-style examples.

    Inputs:
        sources: list[torch.Tensor], two source tensors ordered as [X1, X2].
        target: list[torch.Tensor], one target tensor ordered as [T].
        covariance: optional covariance input, currently unused and kept for compatibility.

    Outputs:
        tuple[dict, dict], PID component dictionary and MI dictionary.
    """
    mi, bias = calculate_story_mi_values(sources, target)
    mi_tri = mi["tri_mi"] - bias["bias_tri_mi"]
    mi_bi_1 = mi["bi_mi_1_t"] - bias["bias_mi_1_t"]
    mi_bi_2 = mi["bi_mi_2_t"] - bias["bias_mi_2_t"]
    pid = {"red": mi_bi_2, "unq1": mi_bi_1 - mi_bi_2, "unq2": 0, "syn": mi_tri - mi_bi_1 }
    return pid, {"tri_mi": mi_tri, "bi_mi_1": mi_bi_1, "bi_mi_2": mi_bi_2}


def truth_pid_equal_unique(
    sources: list[torch.Tensor],
    target: list[torch.Tensor],
    covariance=None,
) -> tuple[dict, dict]:
    """Compute the Gaussian truth row for equal-unique regular examples.

    Inputs:
        sources: list[torch.Tensor], two source tensors ordered as [X1, X2].
        target: list[torch.Tensor], one target tensor ordered as [T].
        covariance: optional covariance input, currently unused and kept for compatibility.

    Outputs:
        tuple[dict, dict], PID component dictionary and MI dictionary.
    """
    mi, bias = calculate_story_mi_values(sources, target)
    mi_tri = mi["tri_mi"] - bias["bias_tri_mi"]
    mi_bi_1 = mi["bi_mi_1_t"] - bias["bias_mi_1_t"]
    mi_bi_2 = mi["bi_mi_2_t"] - bias["bias_mi_2_t"]
    pid = {
        "red": 0,
        "unq1": mi_bi_1,
        "unq2": mi_bi_2,
        "syn": mi_tri - mi_bi_1 - mi_bi_2,
    }
    return pid, {"tri_mi": mi_tri, "bi_mi_1": mi_bi_1, "bi_mi_2": mi_bi_2}


def run_pid_story(
    config: dict,
    function_to_run: Callable,
    truth_func: Callable | None = None,
    methods: tuple[str, ...] = ("tilde", "delta", "flow"),
) -> dict:
    """Run one RVs_Story generator through selected PID methods.

    Inputs:
        config: dict, simulation and PID configuration values.
        function_to_run: Callable, generator accepting (rng, n_samples, p, noise_std).
        truth_func: Callable | None, optional function producing a "True Values" row.
        methods: tuple[str, ...], PID methods passed to pid_calc.

    Outputs:
        dict, result rows keyed by display name such as "Tilde" and "Delta".
    """
    from Partial_Information_Decomposition.PID_calc import pid_calc

    run_config = dict(config)
    rng = np.random.default_rng(run_config["seed"])
    x1, x2, t = function_to_run(rng, run_config["n_samples"], run_config["p"], run_config["noise_std"])
    sources = [torch.from_numpy(x1), torch.from_numpy(x2)]
    target = [torch.from_numpy(t)]
    run_config.update({"dx1": x1.shape[1], "dx2": x2.shape[1], "dt": t.shape[1]})

    results = {}
    if truth_func is not None:
        true_values = truth_func(sources, target, covariance=None)
        if config['bits']:
            pid_bits = {k: v / np.log(2) for k, v in true_values[0].items()}
            mi_bits = {k: v / np.log(2) for k, v in true_values[1].items()}
            true_values = (pid_bits, mi_bits)
        print(f"\nTrue PID values: {true_values[0]}")
        print(f"True MI values: {true_values[1]}")
        results["True Values"] = true_values

    for method in methods:
        results[method.title() if method != "flow" else "Flow"] = pid_calc(
            run_config,
            sources,
            target,
            covariance=None,
            rng=rng,
            on_rvs=None,
            method=method,
        )
        print(f"\nFinished calculating PID with {method} method")
        print("=" * 70)
    return results


def load_story_config(config_path: str | Path | None = None) -> dict:
    """Load the RVs_Story YAML configuration.

    Inputs:
        config_path: str | Path | None, optional path to a YAML config file.

    Outputs:
        dict, parsed YAML configuration.
    """
    path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_single_example(
    config: dict,
    function_to_run: Callable,
    output_name: str,
    truth_func: Callable | None = None,
) -> dict:
    """Run one example and save its PID comparison figure.

    Inputs:
        config: dict, full YAML config containing a "parameters" section.
        function_to_run: Callable, RV generator to execute.
        output_name: str, output image filename inside results_dir.
        truth_func: Callable | None, optional truth-row function.

    Outputs:
        dict, PID result dictionary returned by run_pid_story.
    """

    params = config["parameters"]
    results = run_pid_story(params, function_to_run, truth_func=truth_func)
    output_path = Path(params["results_dir"]) / output_name
    save_pid_comparison_table(results, save_path=str(output_path), config=config)
    return results
