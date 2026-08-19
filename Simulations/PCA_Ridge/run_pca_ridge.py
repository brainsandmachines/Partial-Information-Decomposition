"""Run any registered PCA–Ridge PID comparison through one command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Simulations.PCA_Ridge.pid_feature_middleware import run_pid_feature_comparison
from Simulations.PCA_Ridge.scenarios import SCENARIOS, get_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse common PCA–Ridge simulation arguments.

    Inputs:
        No inputs; values are read from the process command line.

    Outputs:
        argparse.Namespace containing the selected scenario and run parameters.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-train", type=int, default=9000)
    parser.add_argument("--p", type=int, default=70)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--n-components", type=int)
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--pid-method")
    parser.add_argument(
        "--bias-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "Simulations/PCA_Ridge/results")
    return parser.parse_args()


def run_scenario(
    scenario_name: str,
    *,
    n_samples: int = 10000,
    n_train: int = 9000,
    p: int = 70,
    noise_std: float = 1.0,
    n_components: int | None = None,
    n_trials: int = 2,
    base_seed: int = 0,
    pid_method: str | None = None,
    bias_correction: bool = True,
    output_dir: Path | str = PROJECT_ROOT / "Simulations/PCA_Ridge/results",
) -> dict[str, Any]:
    """Run one registered PCA–Ridge scenario with shared orchestration.

    Inputs:
        scenario_name: str key in the scenario registry.
        n_samples: int aligned samples generated per trial.
        n_train: int rows used to fit PCA and ridge models.
        p: int base coordinate dimension used by the selected scenario.
        noise_std: float standard deviation of scenario noise.
        n_components: optional int PCA width overriding the scenario default.
        n_trials: int number of independent trials.
        base_seed: int seed assigned to the first trial.
        pid_method: optional str PID method overriding the scenario default.
        bias_correction: bool passed to PID calculation.
        output_dir: Path or str directory receiving the comparison plot.

    Outputs:
        dict[str, Any] returned by ``run_pid_feature_comparison``.
    """

    if n_samples <= 0 or p <= 0 or n_trials <= 0:
        raise ValueError("n_samples, p, and n_trials must be positive.")
    if n_train <= 0 or n_train >= n_samples:
        raise ValueError("n_train must be positive and smaller than n_samples.")

    scenario = get_scenario(scenario_name)
    selected_components = scenario.n_components if n_components is None else n_components
    if selected_components <= 0:
        raise ValueError("n_components must be positive.")

    population_covariance = scenario.covariance_builder(
        p,
        noise_std,
        **scenario.covariance_kwargs,
    )  # scenario coordinate covariance -> (3*m*p, 3*m*p)
    rv_dimension = scenario.dimension_multiplier * p
    metadata = {"p": rv_dimension}
    if scenario.dimension_multiplier > 1:
        metadata["block_p"] = p

    output_path = Path(output_dir) / (
        f"{scenario.plot_slug}_pid_feature_comparison_{n_trials}_trials.png"
    )
    return run_pid_feature_comparison(
        lambda seed: scenario.generator(
            np.random.default_rng(seed),
            n_samples,
            p,
            noise_std,
            **scenario.generator_kwargs,
        ),
        population_covariance,
        [rv_dimension, rv_dimension, rv_dimension],
        n_samples=n_samples,
        n_train=n_train,
        n_components=selected_components,
        n_trials=n_trials,
        base_seed=base_seed,
        pid_method=scenario.pid_method if pid_method is None else pid_method,
        bias_correction=bias_correction,
        experiment_name=scenario.experiment_name,
        plot_path=output_path,
        plot_title=scenario.plot_title,
        metadata=metadata,
    )


def main() -> dict[str, Any]:
    """Run the scenario selected on the command line.

    Inputs:
        No inputs; delegates to ``parse_args``.

    Outputs:
        dict[str, Any] containing the selected simulation's comparison results.
    """

    args = parse_args()
    return run_scenario(
        args.scenario,
        n_samples=args.n_samples,
        n_train=args.n_train,
        p=args.p,
        noise_std=args.noise_std,
        n_components=args.n_components,
        n_trials=args.n_trials,
        base_seed=args.base_seed,
        pid_method=args.pid_method,
        bias_correction=args.bias_correction,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
