"""Compare native and Eigen-PID Gamma-star optimizer initializations."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Eigen_PID_Simulations.gamma_star_reporting import (
    GAMMA_STAR_START,
    NATIVE_START,
    add_paired_comparisons,
    create_result_row,
    plot_iteration_comparison,
    plot_pid_comparison,
    print_experiment_summary,
    save_figure,
    save_hyperparameters_yaml,
    save_results_csv,
    show_figures,
)
from external.Gaussian_eig_PID.benchmarks.eigen_coupling import (
    construct_eigen_coupling,
    replay_from_coupling,
)
from external.Gaussian_eig_PID.src.gaussian_eigen_pid import GaussianEigenPID
from external.gpid.src.gpid import tilde_pid as gpid_module
from library_wrappers import wrapper_utils

# Thin_PID.py uses its script-style import name when loaded as a module.
sys.modules.setdefault("wrapper_utils", wrapper_utils)
from library_wrappers.Thin_PID import load_exact_gauss_thin_pid


RANDOM_SEED = 56
EXPERIMENT_NAME = "gamma_star_initialization_comparison"
TARGET_DIMENSIONS = (1, 5, 10, 25, 30, 50, 60)
REPEATS_PER_DIMENSION = 5
CHANNEL_GAIN_SCALE = 0.75
OPTIMIZER_REGULARIZATION = 1e-7
EIGEN_NEUTRAL_LOG2_TOLERANCE = 0.0
PLOT_DPI = 300
# Set to None to run every case until convergence or its maximum iteration limit.
TIME_LIMIT_SECONDS = None

GPID_METHOD = "GPID/Tilde-PID"
THIN_PID_METHOD = "Thin-PID"
METHOD_NAMES = (GPID_METHOD, THIN_PID_METHOD)
INITIALIZATION_NAMES = (NATIVE_START, GAMMA_STAR_START)

EXPERIMENT_DIRECTORY = (
    Path(__file__).resolve().parent / "results" / EXPERIMENT_NAME
)
RESULTS_CSV_PATH = EXPERIMENT_DIRECTORY / "iteration_results.csv"
ITERATION_PLOT_PATH = EXPERIMENT_DIRECTORY / "iteration_comparison.png"
PID_PLOT_PATH = EXPERIMENT_DIRECTORY / "pid_comparison.png"
HYPERPARAMETERS_YAML_PATH = EXPERIMENT_DIRECTORY / "hyperparameters.yaml"


def load_optimizer_modules() -> dict[str, tuple[str, Any]]:
    """Load the GPID and Thin-PID modules used by the replay experiment.

    Returns:
        A mapping from display name to ``(replay_method_key, module)`` for
        GPID/Tilde-PID and Thin-PID.
    """
    thin_pid_solver = load_exact_gauss_thin_pid()
    thin_pid_module = sys.modules[thin_pid_solver.__module__]
    return {
        GPID_METHOD: ("venkatesh_tilde", gpid_module),
        THIN_PID_METHOD: ("zhao_thin", thin_pid_module),
    }


def generate_gaussian_system(
    random_generator: np.random.Generator,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one balanced Gaussian target/source covariance system.

    Args:
        random_generator: NumPy generator controlling reproducible channels.
        dimension: Shared positive dimension of target, source 1, and source 2.

    Returns:
        ``(covariance, channel_x, channel_y)`` where covariance is ordered
        ``[target, source1, source2]`` with shape ``(3d, 3d)`` and each channel
        has shape ``(d, d)``.
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive")

    # Independent scalar draws -> (source dimension, target dimension).
    channel_x = random_generator.normal(
        0.0,
        CHANNEL_GAIN_SCALE / np.sqrt(dimension),
        size=(dimension, dimension),
    )
    channel_y = random_generator.normal(
        0.0,
        CHANNEL_GAIN_SCALE / np.sqrt(dimension),
        size=(dimension, dimension),
    )
    identity = np.eye(dimension)

    # Nine (dimension, dimension) blocks -> (3 * dimension, 3 * dimension).
    covariance = np.block(
        [
            [identity, channel_x.T, channel_y.T],
            [channel_x, channel_x @ channel_x.T + identity, channel_x @ channel_y.T],
            [channel_y, channel_y @ channel_x.T, channel_y @ channel_y.T + identity],
        ]
    )
    return covariance, channel_x, channel_y


def construct_native_couplings(
    channel_x: np.ndarray,
    channel_y: np.ndarray,
    thin_pid_module: Any,
) -> dict[str, np.ndarray]:
    """Construct the native feasible optimizer coupling for each PID method.

    Args:
        channel_x: Whitened source-1 channel with shape ``(dx, dm)``.
        channel_y: Whitened source-2 channel with shape ``(dy, dm)``.
        thin_pid_module: Loaded Thin-PID module exposing projection utilities.

    Returns:
        Method-name mapping to native coupling arrays with shape ``(dx, dy)``.
    """
    # (dx, dm) @ (dm, dy) -> (dx, dy).
    gpid_candidate = channel_x @ gpid_module.pinv(channel_y)
    gpid_start = gpid_module.project(gpid_candidate)[0]

    # (dx, dm) @ (dm, dy) -> (dx, dy).
    thin_candidate = channel_x @ thin_pid_module.pinv(channel_y)
    try:
        thin_start = thin_pid_module.thin_project(thin_candidate)[0]
    except np.linalg.LinAlgError:
        thin_start = thin_pid_module.tilde_project(thin_candidate)[0]
    return {GPID_METHOD: gpid_start, THIN_PID_METHOD: thin_start}


def run_experiment() -> list[dict[str, Any]]:
    """Run all cases while printing repeat and dimension-sweep durations.

    Returns:
        Detailed result rows containing convergence diagnostics, optimizer
        updates, PID values, and method-minus-Eigen differences.
    """
    random_generator = np.random.default_rng(RANDOM_SEED)
    eigen_pid = GaussianEigenPID(
        neutral_log2_tol=EIGEN_NEUTRAL_LOG2_TOLERANCE
    )
    optimizer_modules = load_optimizer_modules()
    experiment_metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "random_seed": RANDOM_SEED,
        "repeats_per_dimension": REPEATS_PER_DIMENSION,
        "channel_gain_scale": CHANNEL_GAIN_SCALE,
        "optimizer_regularization": OPTIMIZER_REGULARIZATION,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
    }
    result_rows: list[dict[str, Any]] = []
    full_sweep_start = time.perf_counter()

    for dimension_index, dimension in enumerate(TARGET_DIMENSIONS):
        dimension_sweep_start = time.perf_counter()
        print(
            f"Starting dimension sweep {dimension_index + 1}/"
            f"{len(TARGET_DIMENSIONS)}: dimension={dimension}",
            flush=True,
        )
        for repeat_index in range(REPEATS_PER_DIMENSION):
            repeat_start = time.perf_counter()
            print(
                f"  Starting repeat {repeat_index + 1}/"
                f"{REPEATS_PER_DIMENSION}",
                flush=True,
            )
            covariance, channel_x, channel_y = generate_gaussian_system(
                random_generator, dimension
            )
            eigen_result = eigen_pid.decompose(
                covariance, dimension, dimension, dimension
            )
            eigen_coupling = construct_eigen_coupling(channel_x, channel_y)
            native_couplings = construct_native_couplings(
                channel_x,
                channel_y,
                optimizer_modules[THIN_PID_METHOD][1],
            )

            for method_name, (method_key, method_module) in optimizer_modules.items():
                initializations = (
                    (NATIVE_START, native_couplings[method_name]),
                    (GAMMA_STAR_START, eigen_coupling.coupling),
                )
                for initialization_name, starting_coupling in initializations:
                    convergence = replay_from_coupling(
                        method_key,
                        method_module,
                        channel_x,
                        channel_y,
                        starting_coupling,
                        regularization=OPTIMIZER_REGULARIZATION,
                        time_limit_seconds=TIME_LIMIT_SECONDS,
                    )
                    run_metadata = {
                        "repeat_index": repeat_index,
                        "target_dimension": dimension,
                        "source1_dimension": dimension,
                        "source2_dimension": dimension,
                        "method": method_name,
                        "initialization": initialization_name,
                    }
                    result_rows.append(
                        create_result_row(
                            experiment_metadata,
                            run_metadata,
                            eigen_result,
                            eigen_coupling,
                            convergence,
                        )
                    )

            repeat_seconds = time.perf_counter() - repeat_start
            print(
                f"  Completed repeat {repeat_index + 1}/"
                f"{REPEATS_PER_DIMENSION} in {repeat_seconds:.1f} seconds "
                f"({repeat_seconds / 60.0:.2f} minutes)",
                flush=True,
            )

        dimension_sweep_seconds = time.perf_counter() - dimension_sweep_start
        print(
            f"Completed dimension={dimension} sweep in "
            f"{dimension_sweep_seconds:.1f} seconds "
            f"({dimension_sweep_seconds / 60.0:.2f} minutes)",
            flush=True,
        )

    full_sweep_seconds = time.perf_counter() - full_sweep_start
    print(
        f"Completed full sweep in {full_sweep_seconds:.1f} seconds "
        f"({full_sweep_seconds / 3600.0:.2f} hours)",
        flush=True,
    )

    return add_paired_comparisons(result_rows)


def main() -> Path:
    """Run the experiment and save its CSV, plots, and hyperparameters.

    Returns:
        Path to the detailed experiment CSV.
    """
    result_rows = run_experiment()
    csv_path = save_results_csv(result_rows, RESULTS_CSV_PATH)
    iteration_figure = plot_iteration_comparison(
        result_rows,
        TARGET_DIMENSIONS,
        REPEATS_PER_DIMENSION,
        METHOD_NAMES,
        INITIALIZATION_NAMES,
    )
    pid_figure = plot_pid_comparison(
        result_rows,
        TARGET_DIMENSIONS,
        REPEATS_PER_DIMENSION,
        METHOD_NAMES,
        INITIALIZATION_NAMES,
    )
    iteration_plot_path = save_figure(
        iteration_figure,
        ITERATION_PLOT_PATH,
        dpi=PLOT_DPI,
    )
    pid_plot_path = save_figure(pid_figure, PID_PLOT_PATH, dpi=PLOT_DPI)
    hyperparameters = {
        "experiment_name": EXPERIMENT_NAME,
        "random_seed": RANDOM_SEED,
        "target_dimensions": list(TARGET_DIMENSIONS),
        "source1_dimensions": list(TARGET_DIMENSIONS),
        "source2_dimensions": list(TARGET_DIMENSIONS),
        "repeats_per_dimension": REPEATS_PER_DIMENSION,
        "channel_gain_scale": CHANNEL_GAIN_SCALE,
        "optimizer_regularization": OPTIMIZER_REGULARIZATION,
        "eigen_neutral_log2_tolerance": EIGEN_NEUTRAL_LOG2_TOLERANCE,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "plot_dpi": PLOT_DPI,
        "methods": list(METHOD_NAMES),
        "initializations": list(INITIALIZATION_NAMES),
        "covariance_variable_order": ["target", "source1", "source2"],
        "balanced_dimensions": True,
        "output_paths": {
            "results_csv": str(csv_path),
            "iteration_plot": str(iteration_plot_path),
            "pid_plot": str(pid_plot_path),
            "hyperparameters_yaml": str(HYPERPARAMETERS_YAML_PATH),
        },
    }
    yaml_path = save_hyperparameters_yaml(
        hyperparameters,
        HYPERPARAMETERS_YAML_PATH,
    )

    print_experiment_summary(result_rows, csv_path)
    print(f"Saved iteration plot to {iteration_plot_path}")
    print(f"Saved PID plot to {pid_plot_path}")
    print(f"Saved reproducibility hyperparameters to {yaml_path}")
    show_figures()
    return csv_path


if __name__ == "__main__":
    main()
