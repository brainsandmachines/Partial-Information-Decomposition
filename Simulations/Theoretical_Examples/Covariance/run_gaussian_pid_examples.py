"""Benchmark Gaussian PID methods repeatedly on theoretical covariances."""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import TypeAlias

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
from Simulations.Theoretical_Examples.Covariance.compare_multidim_gaussian_pid import (
    METHODS,
    build_population_covariance,
    write_rows_to_csv,
)

IDENTITY_TOLERANCE = 1e-8
ZERO_TOLERANCE = 1e-10
RUNTIME_REPEATS = 20
COMPONENTS = ("red", "unq1", "unq2", "syn")
PARAMETERS = (
    "target_dim",
    "source1_dim",
    "source2_dim",
    "source1_gain",
    "source2_gain",
    "source1_noise_variance",
    "source2_noise_variance",
)
ExampleConfig: TypeAlias = dict[str, str | int | float | list[int]]

EXAMPLES = (
    dict(
        name="all_atoms_positive",
        expectation="all_positive",
        sweep_variable="source1_dim",
        sweep_values=list(range(1, 1000, 50)),
        target_dim=2,
        source1_dim=2,
        source2_dim=2,
        source1_gain=float(np.sqrt(2.0)),
        source2_gain=float(np.sqrt(2.0)),
        source1_noise_variance=1.0,
        source2_noise_variance=1.0,
    ),
    dict(
        name="both_unique_zero",
        expectation="both_unique_zero",
        sweep_variable="source1_dim",
        sweep_values=list(range(1, 1000, 50)),
        target_dim=2,
        source1_dim=2,
        source2_dim=2,
        source1_gain=1.0,
        source2_gain=1.0,
        source1_noise_variance=1.0,
        source2_noise_variance=1.0,
    ),
    dict(
        name="thin_pid_failure",
        expectation="thin_nonfinite",
        sweep_variable="source1_dim",
        sweep_values=list(range(1, 1000, 50)),
        target_dim=2,
        source1_dim=2,
        source2_dim=2,
        source1_gain=1.0,
        source2_gain=3.0,
        source1_noise_variance=1.0,
        source2_noise_variance=1.0,
    ),
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results_covariance" / "Sweeps_comp"
RESULTS_CSV = OUTPUT_DIR / "theoretical_timing_runs.csv"
SUMMARY_CSV = OUTPUT_DIR / "theoretical_timing_summary.csv"
PLOT_PATH = OUTPUT_DIR / "theoretical_pid_and_runtime_sweeps.png"
YAML_PATH = OUTPUT_DIR / "hyperparameters.yaml"

RESULT_COLUMNS = (
    "example",
    "sweep_variable",
    "swept_dimension",
    *PARAMETERS,
    "runtime_repeat",
    "method",
    "status",
    "message",
    *COMPONENTS,
    "bi_mi_1",
    "bi_mi_2",
    "tri_mi",
    "identity_error_joint",
    "identity_error_x1",
    "identity_error_x2",
    "runtime_seconds",
)
SUMMARY_COLUMNS = (
    "example",
    "sweep_variable",
    "swept_dimension",
    "target_dim",
    "source1_dim",
    "source2_dim",
    "method",
    "metric",
    "mean",
    "sd",
    "n_values",
    "n_runs",
    "n_failures",
)


def one_dimensional_target_example() -> ExampleConfig:
    """Return the example whose target is one-dimensional.

    Inputs: None.
    Outputs: dict containing its name, expectation, dimensions, gains, and
        noise variances.
    """
    return dict(
        name="one_dimensional_target",
        expectation="target_dim_one",
        sweep_variable="source1_dim",
        sweep_values=list(range(1, 1000, 50)),
        target_dim=1,
        source1_dim=1,
        source2_dim=1,
        source1_gain=2.0,
        source2_gain=1.0,
        source1_noise_variance=1.0,
        source2_noise_variance=1.0,
    )


def run_method(
    covariance: torch.Tensor,
    example: ExampleConfig,
    runtime_repeat: int,
    label: str,
    method: str,
) -> dict[str, object]:
    """Time one PID method call on a theoretical covariance.

    Inputs: covariance is a ``(D,D)`` torch.Tensor; example is an
        ExampleConfig; runtime_repeat is an int trial index; label and method
        are strings naming the PID method.
    Outputs: dict containing metadata, PID/MI values, diagnostics, and status.
    """
    numeric = (*COMPONENTS, "bi_mi_1", "bi_mi_2", "tri_mi")
    row: dict[str, object] = {
        "example": example["name"],
        "sweep_variable": example["sweep_variable"],
        "swept_dimension": example["swept_dimension"],
        **{key: example[key] for key in PARAMETERS},
        "runtime_repeat": runtime_repeat,
        "method": label,
        "status": "error",
        "message": "",
        **{key: float("nan") for key in (*numeric, "identity_error_joint", "identity_error_x1", "identity_error_x2")},
        "runtime_seconds": float("nan"),
    }
    config = {
        "dt": int(example["target_dim"]),
        "dx1": int(example["source1_dim"]),
        "dx2": int(example["source2_dim"]),
        "n_samples": 0,
        "bias_correction": False,
        "device": "cpu",
        "eigen_pid_mode": "mathematical",
    }
    caught: list[warnings.WarningMessage] = []
    started = perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                pid, mi = pid_calc(config=config, covariance=covariance, method=method)
    except Exception as error:
        row["message"] = f"{type(error).__name__}: {error}"
        row["runtime_seconds"] = perf_counter() - started
        return row

    row["runtime_seconds"] = perf_counter() - started
    row["message"] = " | ".join(sorted({str(item.message) for item in caught}))
    row.update({key: float(pid[key]) for key in COMPONENTS})
    row.update({key: float(mi[key]) for key in ("bi_mi_1", "bi_mi_2", "tri_mi")})
    if not all(np.isfinite(float(row[key])) for key in numeric):
        row["status"] = "nonfinite"
        return row

    row["identity_error_joint"] = float(row["tri_mi"] - sum(float(row[key]) for key in COMPONENTS))
    row["identity_error_x1"] = float(row["bi_mi_1"] - row["red"] - row["unq1"])
    row["identity_error_x2"] = float(row["bi_mi_2"] - row["red"] - row["unq2"])
    identity_errors = (
        abs(float(row["identity_error_joint"])),
        abs(float(row["identity_error_x1"])),
        abs(float(row["identity_error_x2"])),
    )
    if max(identity_errors) > IDENTITY_TOLERANCE:
        row["status"] = "identity_error"
    elif min(float(row[key]) for key in COMPONENTS) < -IDENTITY_TOLERANCE:
        row["status"] = "negative"
    else:
        row["status"] = "ok"
    return row


def check_theoretical_example(
    example: ExampleConfig,
    rows: list[dict[str, object]],
) -> None:
    """Verify the promised theoretical behavior before saving results.

    Inputs: example is an ExampleConfig; rows contains all repeated method-run
        dictionaries for one theoretical covariance.
    Outputs: None; raises RuntimeError when an expectation is not satisfied.
    """
    result = {
        label: [row for row in rows if row["method"] == label]
        for label, _ in METHODS
    }
    eigen_rows = result["eigen_pid"]
    if any(row["status"] != "ok" for row in eigen_rows):
        raise RuntimeError(f"Eigen-PID failed for {example['name']}.")
    eigen = eigen_rows[0]
    expectation = example["expectation"]
    if expectation == "all_positive" and min(float(eigen[key]) for key in COMPONENTS) <= 0:
        raise RuntimeError("Expected every PID atom to be positive.")
    sources_cover_target = min(int(example["source1_dim"]), int(example["source2_dim"])) >= int(
        example["target_dim"]
    )
    if (
        expectation == "both_unique_zero"
        and sources_cover_target
        and max(abs(float(eigen[key])) for key in ("unq1", "unq2")) > ZERO_TOLERANCE
    ):
        raise RuntimeError("Expected both unique atoms to be zero.")
    if expectation == "thin_nonfinite" and any(
        row["status"] not in {"nonfinite", "error"} for row in result["thin_pid"]
    ):
        raise RuntimeError("Expected Thin-PID to fail numerically.")
    if expectation == "target_dim_one" and int(example["target_dim"]) != 1:
        raise RuntimeError("Expected a one-dimensional target.")


def summarize_results(
    rows: list[dict[str, object]],
    examples: list[ExampleConfig],
) -> list[dict[str, object]]:
    """Calculate PID and runtime statistics across timing repetitions.

    Inputs: rows is a list of repeated theoretical result dictionaries;
        examples is a list of ExampleConfig values.
    Outputs: list of dictionaries containing metric mean and sample SD.
    """
    output: list[dict[str, object]] = []
    for example in examples:
        for swept_dimension in example["sweep_values"]:
            for label, _ in METHODS:
                matching = [
                    row
                    for row in rows
                    if row["example"] == example["name"]
                    and row["swept_dimension"] == swept_dimension
                    and row["method"] == label
                ]
                successful = [row for row in matching if row["status"] == "ok"]
                for metric in (*COMPONENTS, "runtime_seconds"):
                    source = matching if metric == "runtime_seconds" else successful
                    values = np.asarray(
                        [float(row[metric]) for row in source if np.isfinite(float(row[metric]))]
                    )  # repeated scalar results -> (n_values,)
                    output.append(
                        {
                            **{key: matching[0][key] for key in SUMMARY_COLUMNS[:7]},
                            "metric": metric,
                            "mean": float(values.mean()) if values.size else float("nan"),
                            "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                            "n_values": int(values.size),
                            "n_runs": len(matching),
                            "n_failures": len(matching) - len(successful),
                        }
                    )
    return output


def plot_results(rows: list[dict[str, object]], examples: list[ExampleConfig], path: str | Path) -> Path:
    """Plot theoretical PID values and runtime mean ± sample SD.

    Inputs: rows is a list of summary dictionaries; examples is a list of
        ExampleConfig values; path is the str or Path image destination.
    Outputs: Path of the saved plot.
    """
    path = Path(path)
    metrics = (*COMPONENTS, "runtime_seconds")
    metric_titles = ("Redundancy", "Unique X1", "Unique X2", "Synergy", "Runtime mean ± SD")
    figure, axes = plt.subplots(4, 5, figsize=(22, 16))  # examples/metrics -> axes (4, 5)
    axes = np.asarray(axes)  # axes (4, 5) -> array (4, 5)
    colors = ("#2563eb", "#d97706", "#059669")
    for row_index, example in enumerate(examples):
        for column_index, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
            axis = axes[row_index, column_index]
            plotted_values: list[float] = []
            for color, (label, _) in zip(colors, METHODS):
                selected = sorted(
                    (
                        row for row in rows
                        if row["example"] == example["name"]
                        and row["method"] == label
                        and row["metric"] == metric
                    ),
                    key=lambda row: int(row["swept_dimension"]),
                )
                dimensions = np.asarray(
                    [int(row["swept_dimension"]) for row in selected]
                )  # summary rows -> (n_dimensions,)
                means = np.asarray([float(row["mean"]) for row in selected])  # summary rows -> (n_dimensions,)
                sd = np.asarray([float(row["sd"]) for row in selected])  # summary rows -> (n_dimensions,)
                plotted_values.extend(float(value) for value in means if np.isfinite(value))
                if metric == "runtime_seconds":
                    axis.errorbar(dimensions, means, yerr=sd, marker="o", capsize=3, color=color, label=label)
                else:
                    axis.plot(dimensions, means, marker="o", color=color, label=label)
            finite_values = np.asarray(plotted_values)  # plotted scalar values -> (n_finite_values,)
            if metric != "runtime_seconds" and finite_values.size and float(np.ptp(finite_values)) <= ZERO_TOLERANCE:
                center = float(finite_values.mean())
                margin = max(abs(center) * 0.05, 0.05)
                axis.set_ylim(center - margin, center + margin)
            title = f"{example['name']}\n{metric_title}"
            if example["expectation"] == "thin_nonfinite" and metric != "runtime_seconds":
                title += " (Thin non-finite)"
            axis.set(
                title=title,
                xlabel=str(example["sweep_variable"]),
                ylabel="seconds" if metric == "runtime_seconds" else "bits",
            )
            axis.set_xticks(example["sweep_values"][::4])
            if metric == "runtime_seconds":
                axis.set_yscale("log")
            else:
                axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(METHODS), frameon=False)
    figure.suptitle(
        f"Theoretical Gaussian PID dimension sweeps ({RUNTIME_REPEATS} runtime repetitions)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def save_hyperparameters(path: str | Path, examples: list[ExampleConfig], outputs: dict[str, str]) -> Path:
    """Save all non-path settings and output paths as YAML.

    Inputs: path is the YAML destination; examples stores all example configs;
        outputs maps output names to paths.
    Outputs: Path of the saved YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "data_source": "theoretical_covariance",
        "n_samples": 0,
        "runtime_repeats": RUNTIME_REPEATS,
        "timing_scope": "pid_calc call only",
        "methods": [list(method) for method in METHODS],
        "covariance_order": ["T", "X1", "X2"],
        "identity_tolerance": IDENTITY_TOLERANCE,
        "zero_tolerance": ZERO_TOLERANCE,
        "examples": examples,
        "summary_csv": "PID values and runtime mean plus sample SD across repetitions",
        "outputs": outputs,
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, sort_keys=False)
    return path


def main(
    results_csv_path: str | Path = RESULTS_CSV,
    summary_csv_path: str | Path = SUMMARY_CSV,
    plot_path: str | Path = PLOT_PATH,
    yaml_path: str | Path = YAML_PATH,
) -> tuple[Path, Path, Path, Path]:
    """Benchmark every theoretical covariance over repeated method calls.

    Inputs: results_csv_path, summary_csv_path, plot_path, and yaml_path are
        str or Path destinations.
    Outputs: tuple containing those four paths as Path objects.
    """
    paths = tuple(map(Path, (results_csv_path, summary_csv_path, plot_path, yaml_path)))
    examples = [*EXAMPLES, one_dimensional_target_example()]
    rows: list[dict[str, object]] = []
    write_rows_to_csv(paths[0], rows, list(RESULT_COLUMNS))
    for example in examples:
        for swept_dimension in example["sweep_values"]:
            print(f"{example['name']}: {example['sweep_variable']}={swept_dimension}", flush=True)
            current = {
                **example,
                str(example["sweep_variable"]): swept_dimension,
                "swept_dimension": swept_dimension,
            }
            covariance = build_population_covariance(
                *(int(current[key]) for key in ("target_dim", "source1_dim", "source2_dim")),
                *(float(current[key]) for key in PARAMETERS[3:]),
            )  # scalar hyperparameters -> (D, D)
            theoretical_rows: list[dict[str, object]] = []
            for label, method in METHODS:
                for runtime_repeat in range(RUNTIME_REPEATS):
                    print(f"  {label}: timing repeat {runtime_repeat + 1}/{RUNTIME_REPEATS}", flush=True)
                    theoretical_rows.append(run_method(covariance, current, runtime_repeat, label, method))
                rows.extend(theoretical_rows[-RUNTIME_REPEATS:])
                write_rows_to_csv(paths[0], rows, list(RESULT_COLUMNS))
            check_theoretical_example(current, theoretical_rows)

    summary_rows = summarize_results(rows, examples)
    write_rows_to_csv(paths[1], summary_rows, list(SUMMARY_COLUMNS))
    plot_results(summary_rows, examples, paths[2])
    outputs = dict(zip(("results_csv", "summary_csv", "plot", "yaml"), map(str, paths)))
    save_hyperparameters(paths[3], examples, outputs)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv-path", type=Path, default=RESULTS_CSV)
    parser.add_argument("--summary-csv-path", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--plot-path", type=Path, default=PLOT_PATH)
    parser.add_argument("--yaml-path", type=Path, default=YAML_PATH)
    args = parser.parse_args()
    for output_path in main(args.results_csv_path, args.summary_csv_path, args.plot_path, args.yaml_path):
        print(output_path)
