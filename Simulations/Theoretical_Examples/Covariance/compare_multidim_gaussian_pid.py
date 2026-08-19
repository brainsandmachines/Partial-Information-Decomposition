"""Compare population Gaussian PID methods on a flexible channel model.

This is the covariance-only counterpart of ``external/flow-pid/examples/gpid/
multi_dim.py``. It does not draw samples or train the normalizing-flow PID
estimator. Edit the constants below to change the experiment.
"""

from __future__ import annotations

import csv
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Partial_Information_Decomposition.PID_calc import pid_calc
from Simulations.evil_twin.evil_twin_pid_batch_utils import write_rows_to_csv
from Simulations.Theoretical_Examples.Covariance.cov_functions import \
    rectangular_identity

# Edit these constants to configure the population-covariance dimension sweep.
TARGET_DIM = 2
SOURCE1_DIM = 2
SOURCE2_DIM = 2
DIMENSION_TO_SWEEP = "source1"
DIMENSION_VALUES = [2]  # positive integers to sweep in order
# Crossed gains give each source stronger coordinates while retaining shared signal.
SOURCE1_GAIN = 1.0
SOURCE2_GAIN = 3.0
SOURCE1_GAIN_COORDINATE = 0
SOURCE2_GAIN_COORDINATE = 1
SOURCE1_NOISE_VARIANCE = 1.0
SOURCE2_NOISE_VARIANCE = 1.0
DEVICE = "cpu"
BIAS_CORRECTION = False
EIGEN_PID_MODE = "mathematical"
N_SAMPLES = 0
COVARIANCE_ORDER = ("T", "X1", "X2")
PLOT_FIGSIZE = (19, 9)
SPEEDUP_PLOT_FIGSIZE = (11, 6)
PLOT_DPI = 200
OUTPUT_DIR = Path('/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Simulations/Theoretical_Examples/Covariance/results_covariance/thin_pid_nonfinite_eigen_pid_solution')
CSV_PATH = OUTPUT_DIR / "dimension_sweep_results.csv"
PLOT_PATH = OUTPUT_DIR / "dimension_sweep_results.png"
SPEEDUP_PLOT_PATH = OUTPUT_DIR / "eigen_pid_speedup.png"
YAML_PATH = OUTPUT_DIR / "dimension_sweep_constants.yaml"

METHODS = (
    ("gpid_tilde", "tilde"),
    ("thin_pid", "thin"),
    ("eigen_pid", "eigen"),
)
VALUE_COLUMNS = (
    "redundancy",
    "unique_source1",
    "unique_source2",
    "synergy",
    "I(source1;target)",
    "I(source2;target)",
    "I(source1,source2;target)",
)
CSV_COLUMNS = (
    "swept_variable",
    "swept_dimension",
    "target_dim",
    "source1_dim",
    "source2_dim",
    "method",
    "minimum_eigenvalue",
    *VALUE_COLUMNS,
    "runtime_seconds",
    "max_abs_diff_vs_eigen",
)


def save_run_constants(yaml_path: str | Path) -> Path:
    """Save every uppercase script constant as a YAML run snapshot.

    Inputs:
        yaml_path: str or Path, destination YAML file inside the results folder.

    Outputs:
        Path, location of the saved YAML file. Paths are stored as strings,
        NumPy integer sweep values as Python integers, and tuples as YAML lists.
    """
    yaml_path = Path(yaml_path)
    constants = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "TARGET_DIM": TARGET_DIM,
        "SOURCE1_DIM": SOURCE1_DIM,
        "SOURCE2_DIM": SOURCE2_DIM,
        "DIMENSION_TO_SWEEP": DIMENSION_TO_SWEEP,
        "DIMENSION_VALUES": [int(value) for value in DIMENSION_VALUES],
        "SOURCE1_GAIN": SOURCE1_GAIN,
        "SOURCE2_GAIN": SOURCE2_GAIN,
        "SOURCE1_GAIN_COORDINATE": SOURCE1_GAIN_COORDINATE,
        "SOURCE2_GAIN_COORDINATE": SOURCE2_GAIN_COORDINATE,
        "SOURCE1_NOISE_VARIANCE": SOURCE1_NOISE_VARIANCE,
        "SOURCE2_NOISE_VARIANCE": SOURCE2_NOISE_VARIANCE,
        "DEVICE": DEVICE,
        "BIAS_CORRECTION": BIAS_CORRECTION,
        "EIGEN_PID_MODE": EIGEN_PID_MODE,
        "N_SAMPLES": N_SAMPLES,
        "COVARIANCE_ORDER": list(COVARIANCE_ORDER),
        "PLOT_FIGSIZE": list(PLOT_FIGSIZE),
        "SPEEDUP_PLOT_FIGSIZE": list(SPEEDUP_PLOT_FIGSIZE),
        "PLOT_DPI": PLOT_DPI,
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "CSV_PATH": str(CSV_PATH),
        "PLOT_PATH": str(PLOT_PATH),
        "SPEEDUP_PLOT_PATH": str(SPEEDUP_PLOT_PATH),
        "YAML_PATH": str(YAML_PATH),
        "METHODS": [list(method) for method in METHODS],
        "VALUE_COLUMNS": list(VALUE_COLUMNS),
        "CSV_COLUMNS": list(CSV_COLUMNS),
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(constants, handle, sort_keys=False, allow_unicode=False)
    return yaml_path


def validate_parameters(
    target_dim: int,
    source1_dim: int,
    source2_dim: int,
    source1_gain: float,
    source2_gain: float,
    source1_noise_variance: float,
    source2_noise_variance: float,
) -> None:
    """Validate dimensions, gains, and source-noise variances.

    Inputs:
        target_dim: int, number of target coordinates.
        source1_dim: int, number of source-1 coordinates.
        source2_dim: int, number of source-2 coordinates.
        source1_gain: float, gain of the first coordinate in each repeated
            two-coordinate source-1 channel block.
        source2_gain: float, gain of the second coordinate in each repeated
            two-coordinate source-2 channel block.
        source1_noise_variance: float, independent source-1 noise variance.
        source2_noise_variance: float, independent source-2 noise variance.

    Outputs:
        None. Raises ValueError for invalid parameters.
    """
    dimensions = {
        "target_dim": target_dim,
        "source1_dim": source1_dim,
        "source2_dim": source2_dim,
    }
    for name, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    finite_values = {
        "source1_gain": source1_gain,
        "source2_gain": source2_gain,
        "source1_noise_variance": source1_noise_variance,
        "source2_noise_variance": source2_noise_variance,
    }
    for name, value in finite_values.items():
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}.")

    for name, value in (
        ("source1_noise_variance", source1_noise_variance),
        ("source2_noise_variance", source2_noise_variance),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than zero, got {value!r}.")


def build_channel_matrix(
    source_dim: int,
    target_dim: int,
    special_gain: float,
    special_coordinate: int,
) -> torch.Tensor:
    """Build a rectangular channel by repeating a two-coordinate gain block.

    Inputs:
        source_dim: int, number of source coordinates (matrix rows).
        target_dim: int, number of target coordinates (matrix columns).
        special_gain: float, gain assigned to the selected coordinate of every
            repeated two-coordinate block.
        special_coordinate: int, zero-based position in the two-coordinate
            block; must be either zero or one.

    Outputs:
        torch.Tensor: float64 channel matrix with shape
        ``(source_dim, target_dim)``. Matched coordinates otherwise have unit
        gain; unmatched source coordinates receive no target signal. Truncated
        final blocks retain only their available coordinates.
    """
    if special_coordinate not in (0, 1):
        raise ValueError(
            "special_coordinate must be 0 or 1 for a two-coordinate block, "
            f"got {special_coordinate!r}."
        )
    channel = rectangular_identity(source_dim, target_dim, dtype=torch.float64)  # scalar dimensions -> (source_dim, target_dim)
    matched_dim = min(source_dim, target_dim)
    for gain_coordinate in range(special_coordinate, matched_dim, 2):
        channel[gain_coordinate, gain_coordinate] = special_gain
    return channel  # (source_dim, target_dim) -> (source_dim, target_dim)


def build_population_covariance(
    target_dim: int,
    source1_dim: int,
    source2_dim: int,
    source1_gain: float,
    source2_gain: float,
    source1_noise_variance: float,
    source2_noise_variance: float,
) -> torch.Tensor:
    """Construct the theoretical Gaussian covariance in ``[T, X1, X2]`` order.

    Inputs:
        target_dim: int, number of standard-normal target coordinates.
        source1_dim: int, number of source-1 coordinates.
        source2_dim: int, number of source-2 coordinates.
        source1_gain: float, gain of the first coordinate in each repeated
            two-coordinate source-1 channel block.
        source2_gain: float, gain of the second coordinate in each repeated
            two-coordinate source-2 channel block.
        source1_noise_variance: float, variance of independent source-1 noise.
        source2_noise_variance: float, variance of independent source-2 noise.

    Outputs:
        torch.Tensor: symmetric positive-definite float64 covariance with shape
        ``(target_dim + source1_dim + source2_dim,) * 2``.
    """
    validate_parameters(
        target_dim,
        source1_dim,
        source2_dim,
        source1_gain,
        source2_gain,
        source1_noise_variance,
        source2_noise_variance,
    )
    target_covariance = torch.eye(target_dim, dtype=torch.float64)  # scalar target_dim -> (target_dim, target_dim)
    source1_channel = build_channel_matrix(source1_dim, target_dim, source1_gain, SOURCE1_GAIN_COORDINATE)  # scalar dimensions -> (source1_dim, target_dim)
    source2_channel = build_channel_matrix(source2_dim, target_dim, source2_gain, SOURCE2_GAIN_COORDINATE)  # scalar dimensions -> (source2_dim, target_dim)
    source1_noise = source1_noise_variance * torch.eye(source1_dim, dtype=torch.float64)  # scalar variance/dimension -> (source1_dim, source1_dim)
    source2_noise = source2_noise_variance * torch.eye(source2_dim, dtype=torch.float64)  # scalar variance/dimension -> (source2_dim, source2_dim)

    target_source1 = target_covariance @ source1_channel.T  # (target_dim, target_dim) @ (target_dim, source1_dim) -> (target_dim, source1_dim)
    target_source2 = target_covariance @ source2_channel.T  # (target_dim, target_dim) @ (target_dim, source2_dim) -> (target_dim, source2_dim)
    source1_covariance = source1_channel @ target_covariance @ source1_channel.T + source1_noise  # (source1_dim, target_dim) @ (target_dim, target_dim) @ (target_dim, source1_dim) -> (source1_dim, source1_dim)
    source2_covariance = source2_channel @ target_covariance @ source2_channel.T + source2_noise  # (source2_dim, target_dim) @ (target_dim, target_dim) @ (target_dim, source2_dim) -> (source2_dim, source2_dim)
    source1_source2 = source1_channel @ target_covariance @ source2_channel.T  # (source1_dim, target_dim) @ (target_dim, target_dim) @ (target_dim, source2_dim) -> (source1_dim, source2_dim)

    top = torch.cat([target_covariance, target_source1, target_source2], dim=1)  # covariance blocks -> (target_dim, total_dim)
    middle = torch.cat([target_source1.T, source1_covariance, source1_source2], dim=1)  # covariance blocks -> (source1_dim, total_dim)
    bottom = torch.cat([target_source2.T, source1_source2.T, source2_covariance], dim=1)  # covariance blocks -> (source2_dim, total_dim)
    covariance = torch.cat([top, middle, bottom], dim=0)  # covariance block rows -> (total_dim, total_dim)

    expected_dim = target_dim + source1_dim + source2_dim
    if covariance.shape != (expected_dim, expected_dim):
        raise RuntimeError(f"Unexpected covariance shape {tuple(covariance.shape)}.")
    if not torch.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Constructed covariance is not symmetric.")
    minimum_eigenvalue = float(torch.linalg.eigvalsh(covariance).min())
    if minimum_eigenvalue <= 0.0:
        raise RuntimeError(
            "Constructed covariance is not positive definite: "
            f"minimum eigenvalue={minimum_eigenvalue:.6g}."
        )
    return covariance


def run_pid_methods(
    covariance: torch.Tensor,
    target_dim: int,
    source1_dim: int,
    source2_dim: int,
) -> dict[str, dict[str, float]]:
    """Run GPID Tilde, Thin-PID, and Eigen-PID on one population covariance.

    Inputs:
        covariance: torch.Tensor ordered ``[T, X1, X2]`` with shape
            ``(target_dim + source1_dim + source2_dim,) * 2``.
        target_dim: int, target block dimension.
        source1_dim: int, source-1 block dimension.
        source2_dim: int, source-2 block dimension.

    Outputs:
        dict[str, dict[str, float]]: method rows containing four PID atoms,
        three mutual informations, and one single-run runtime in seconds.
    """
    config = {
        "dt": target_dim,
        "dx1": source1_dim,
        "dx2": source2_dim,
        "n_samples": N_SAMPLES,
        "bias_correction": BIAS_CORRECTION,
        "device": DEVICE,
        "eigen_pid_mode": EIGEN_PID_MODE,
    }
    results: dict[str, dict[str, float]] = {}
    for label, method in METHODS:
        started = perf_counter()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            pid, mi = pid_calc(
                config=config.copy(),
                sources=None,
                target=None,
                covariance=covariance,
                method=method,
            )
        runtime_seconds = perf_counter() - started
        row = {
            "redundancy": float(pid["red"]),
            "unique_source1": float(pid["unq1"]),
            "unique_source2": float(pid["unq2"]),
            "synergy": float(pid["syn"]),
            "I(source1;target)": float(mi["bi_mi_1"]),
            "I(source2;target)": float(mi["bi_mi_2"]),
            "I(source1,source2;target)": float(mi["tri_mi"]),
            "runtime_seconds": float(runtime_seconds),
        }
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"{label} returned a non-finite value: {row}.")
        results[label] = row
    return results


def run_dimension_sweep(
    dimension_to_sweep: str,
    dimension_values: list[int],
    target_dim: int,
    source1_dim: int,
    source2_dim: int,
    source1_gain: float,
    source2_gain: float,
    source1_noise_variance: float,
    source2_noise_variance: float,
    csv_path: str | Path,
) -> list[dict[str, float | int | str]]:
    """Run and checkpoint PID methods while changing one variable dimension.

    Inputs:
        dimension_to_sweep: str, one of ``"target"``, ``"source1"``, or
            ``"source2"``.
        dimension_values: list[int], positive dimensions to evaluate in order.
        target_dim: int, fixed target dimension unless target is swept.
        source1_dim: int, fixed source-1 dimension unless source 1 is swept.
        source2_dim: int, fixed source-2 dimension unless source 2 is swept.
        source1_gain: float, gain of the first coordinate in each repeated
            two-coordinate source-1 channel block.
        source2_gain: float, gain of the second coordinate in each repeated
            two-coordinate source-2 channel block.
        source1_noise_variance: float, independent source-1 noise variance.
        source2_noise_variance: float, independent source-2 noise variance.
        csv_path: str or Path, checkpoint CSV replaced at run start and updated
            after each completed dimension.

    Outputs:
        list[dict[str, float | int | str]]: one flat comparison row per
        dimension and PID method, including covariance diagnostics and runtime.
    """
    allowed_variables = {"target", "source1", "source2"}
    if dimension_to_sweep not in allowed_variables:
        raise ValueError(
            f"dimension_to_sweep must be one of {sorted(allowed_variables)}, "
            f"got {dimension_to_sweep!r}."
        )
    if not dimension_values:
        raise ValueError("dimension_values must contain at least one dimension.")

    csv_path = Path(csv_path)
    write_rows_to_csv(csv_path, [], list(CSV_COLUMNS))
    base_dimensions = {
        "target": target_dim,
        "source1": source1_dim,
        "source2": source2_dim,
    }
    rows: list[dict[str, float | int | str]] = []
    for swept_dimension in dimension_values:
        print(f"Running dimension: {dimension_to_sweep}={swept_dimension}...", flush=True)
        dimensions = base_dimensions.copy()
        dimensions[dimension_to_sweep] = swept_dimension
        covariance = build_population_covariance(
            dimensions["target"],
            dimensions["source1"],
            dimensions["source2"],
            source1_gain,
            source2_gain,
            source1_noise_variance,
            source2_noise_variance,
        )
        method_results = run_pid_methods(
            covariance,
            dimensions["target"],
            dimensions["source1"],
            dimensions["source2"],
        )
        eigen_values = method_results["eigen_pid"]
        minimum_eigenvalue = float(torch.linalg.eigvalsh(covariance).min())
        for method, result in method_results.items():
            rows.append(
                {
                    "swept_variable": dimension_to_sweep,
                    "swept_dimension": int(swept_dimension),
                    "target_dim": int(dimensions["target"]),
                    "source1_dim": int(dimensions["source1"]),
                    "source2_dim": int(dimensions["source2"]),
                    "method": method,
                    "minimum_eigenvalue": minimum_eigenvalue,
                    **result,
                    "max_abs_diff_vs_eigen": max(
                        abs(result[column] - eigen_values[column])
                        for column in VALUE_COLUMNS
                    ),
                }
            )
        write_rows_to_csv(csv_path, rows, list(CSV_COLUMNS))
    return rows


def plot_dimension_sweep_csv(
    csv_path: str | Path,
    plot_path: str | Path,
    speedup_plot_path: str | Path,
) -> tuple[Path, Path]:
    """Plot result curves and direct Eigen-PID speedups from the sweep CSV.

    Inputs:
        csv_path: str or Path, completed dimension-sweep CSV.
        plot_path: str or Path, destination PNG file.
        speedup_plot_path: str or Path, primary destination for Eigen-PID
            runtime speedups relative to GPID Tilde and Thin-PID. Matching SVG
            and PDF files are saved beside it for use in papers.

    Outputs:
        tuple[Path, Path]: locations of the saved eight-panel result figure and
        the primary direct Eigen-PID speedup figure. The returned speedup path
        also has SVG and PDF counterparts with the same stem.
    """
    csv_path = Path(csv_path)
    plot_path = Path(plot_path)
    speedup_plot_path = Path(speedup_plot_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dimension-sweep CSV does not exist: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError(
                f"Unexpected CSV columns in {csv_path}: {reader.fieldnames}."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Dimension-sweep CSV contains no result rows: {csv_path}")

    plot_metrics = (*VALUE_COLUMNS, "runtime_seconds")
    plot_titles = {
        "redundancy": "Redundancy",
        "unique_source1": "Unique source 1",
        "unique_source2": "Unique source 2",
        "synergy": "Synergy",
        "I(source1;target)": "I(source 1; target)",
        "I(source2;target)": "I(source 2; target)",
        "I(source1,source2;target)": "I(source 1, source 2; target)",
        "runtime_seconds": "Runtime",
    }
    figure, axes = plt.subplots(2, 4, figsize=PLOT_FIGSIZE, sharex=True)  # subplot grid -> figure and axes (2, 4)
    for axis, metric in zip(axes.flat, plot_metrics):
        for method_label, _ in METHODS:
            method_rows = sorted(
                (row for row in rows if row["method"] == method_label),
                key=lambda row: int(row["swept_dimension"]),
            )
            dimensions = np.asarray(
                [int(row["swept_dimension"]) for row in method_rows],
                dtype=int,
            )  # CSV rows -> (n_dimensions,)
            values = np.asarray(
                [float(row[metric]) for row in method_rows],
                dtype=float,
            )  # CSV rows -> (n_dimensions,)
            axis.plot(dimensions, values, marker="o", linewidth=1.8, label=method_label)
        axis.set_title(plot_titles[metric])
        axis.set_xlabel("Dimensions")
        axis.set_ylabel("seconds" if metric == "runtime_seconds" else "bits")
        if metric == "runtime_seconds":
            axis.set_yscale("log")
        else:
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(True, alpha=0.3)

    fixed_dimensions = ", ".join(
        f"{name}={rows[0][f'{name}_dim']}"
        for name in ("target", "source1", "source2")
        if name != rows[0]["swept_variable"]
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(
        f"Population Gaussian PID dimension sweep ({fixed_dimensions})",
        y=0.995,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(METHODS),
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)

    runtime_by_dimension: dict[int, dict[str, float]] = {}
    for row in rows:
        swept_dimension = int(row["swept_dimension"])
        method = row["method"]
        runtime_seconds = float(row["runtime_seconds"])
        if not np.isfinite(runtime_seconds) or runtime_seconds <= 0.0:
            raise ValueError(
                "Runtime values must be finite and positive to calculate "
                f"speedup, got {runtime_seconds!r} for {method} at dimension "
                f"{swept_dimension}."
            )
        runtime_by_dimension.setdefault(swept_dimension, {})[method] = runtime_seconds

    required_methods = {"gpid_tilde", "thin_pid", "eigen_pid"}
    complete_dimensions = sorted(
        dimension
        for dimension, runtimes in runtime_by_dimension.items()
        if required_methods.issubset(runtimes)
    )
    if not complete_dimensions:
        raise ValueError(
            "The CSV has no dimension containing runtimes for gpid_tilde, "
            "thin_pid, and eigen_pid."
        )

    speedup_figure, speedup_axis = plt.subplots(figsize=SPEEDUP_PLOT_FIGSIZE)  # one speedup plot -> figure and scalar axis
    dimensions = np.asarray(complete_dimensions, dtype=int)  # completed dimension values -> (n_dimensions,)
    comparisons = (
        ("gpid_tilde", "Eigen-PID vs GPID Tilde", "#2563eb"),
        ("thin_pid", "Eigen-PID vs Thin-PID", "#d97706"),
    )
    for baseline_method, label, color in comparisons:
        speedups = np.asarray(
            [
                runtime_by_dimension[dimension][baseline_method]
                / runtime_by_dimension[dimension]["eigen_pid"]
                for dimension in complete_dimensions
            ],
            dtype=float,
        )  # method and Eigen-PID runtimes -> (n_dimensions,)
        speedup_axis.plot(
            dimensions,
            speedups,
            marker="o",
            markersize=4,
            linewidth=2.0,
            color=color,
            label=label,
        )

    speedup_axis.axhline(
        1.0,
        color="#525252",
        linestyle="--",
        linewidth=1.2,
        label="Same runtime (1x)",
    )
    speedup_axis.set_title("How much faster is Eigen-PID?")
    speedup_axis.set_xlabel("Dimensions")
    speedup_axis.set_ylabel("Eigen-PID speedup (x faster)")
    speedup_axis.set_ylim(bottom=0.0)
    speedup_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    speedup_axis.grid(True, alpha=0.3)
    speedup_axis.legend(frameon=False)
    speedup_figure.tight_layout()
    speedup_plot_path.parent.mkdir(parents=True, exist_ok=True)
    speedup_figure.savefig(
        speedup_plot_path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    for vector_plot_path in (
        speedup_plot_path.with_suffix(".svg"),
        speedup_plot_path.with_suffix(".pdf"),
    ):
        if vector_plot_path != speedup_plot_path:
            speedup_figure.savefig(vector_plot_path, bbox_inches="tight")
    plt.close(speedup_figure)
    return plot_path, speedup_plot_path


def main() -> tuple[Path, Path, Path, Path]:
    """Save constants, run the sweep, and save result and speedup plots.

    Inputs:
        None. Uses the editable module constants.

    Outputs:
        tuple[Path, Path, Path, Path]: paths to the completed CSV, result PNG,
        primary Eigen-PID speedup plot, and YAML constants snapshot. SVG and
        PDF versions of the speedup plot are saved beside the primary plot.
    """
    yaml_path = save_run_constants(YAML_PATH)
    run_dimension_sweep(
        DIMENSION_TO_SWEEP,
        DIMENSION_VALUES,
        TARGET_DIM,
        SOURCE1_DIM,
        SOURCE2_DIM,
        SOURCE1_GAIN,
        SOURCE2_GAIN,
        SOURCE1_NOISE_VARIANCE,
        SOURCE2_NOISE_VARIANCE,
        CSV_PATH,
    )
    plot_path, speedup_plot_path = plot_dimension_sweep_csv(
        CSV_PATH,
        PLOT_PATH,
        SPEEDUP_PLOT_PATH,
    )
    return CSV_PATH, plot_path, speedup_plot_path, yaml_path


if __name__ == "__main__":
    main()
