"""CSV and figure helpers for the Gamma-star initialization experiment."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.figure import Figure


NATIVE_START = "Original covariance (native start)"
GAMMA_STAR_START = "Eigen-PID Gamma* start"

ResultRow = dict[str, Any]


def create_result_row(
    experiment_metadata: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    eigen_result: Any,
    eigen_coupling: Any,
    convergence: Mapping[str, Any],
) -> ResultRow:
    """Create one CSV-ready optimizer-versus-Eigen result row.

    Args:
        experiment_metadata: Mapping with experiment-wide scalar settings.
        run_metadata: Mapping identifying the dimension, repeat, method, and
            initialization for this run.
        eigen_result: ``GaussianPIDResult`` for the simulated covariance.
        eigen_coupling: ``EigenCoupling`` constructed from the two channels.
        convergence: Mapping returned by ``replay_from_coupling``.

    Returns:
        A flat dictionary containing optimizer updates, convergence status,
        PID values, method-minus-Eigen differences, and numerical diagnostics.
    """
    stopped_by = str(convergence["stopped_by"])
    optimizer_updates = int(convergence["iterations"])
    if stopped_by in {"published_objective_change", "time_limit"}:
        # Final loop check without an update: loop counter -> update count.
        optimizer_updates -= 1
    converged = stopped_by == "published_objective_change"

    optimizer_union = float(convergence["best_unregularized_objective_bits"])
    optimizer_uix = optimizer_union - eigen_result.imy_bits
    optimizer_uiy = optimizer_union - eigen_result.imx_bits
    optimizer_redundancy = (
        eigen_result.imx_bits + eigen_result.imy_bits - optimizer_union
    )
    optimizer_synergy = eigen_result.imxy_bits - optimizer_union

    return {
        **experiment_metadata,
        **run_metadata,
        "eigen_imx_bits": eigen_result.imx_bits,
        "eigen_imy_bits": eigen_result.imy_bits,
        "eigen_imxy_bits": eigen_result.imxy_bits,
        "eigen_union_bits": eigen_result.union_info_bits,
        "eigen_uix_bits": eigen_result.uix_bits,
        "eigen_uiy_bits": eigen_result.uiy_bits,
        "eigen_redundancy_bits": eigen_result.redundancy_bits,
        "eigen_synergy_bits": eigen_result.synergy_bits,
        "optimizer_union_bits": optimizer_union,
        "optimizer_uix_bits": optimizer_uix,
        "optimizer_uiy_bits": optimizer_uiy,
        "optimizer_redundancy_bits": optimizer_redundancy,
        "optimizer_synergy_bits": optimizer_synergy,
        "optimizer_minus_eigen_union_bits": (
            optimizer_union - eigen_result.union_info_bits
        ),
        "optimizer_minus_eigen_uix_bits": optimizer_uix - eigen_result.uix_bits,
        "optimizer_minus_eigen_uiy_bits": optimizer_uiy - eigen_result.uiy_bits,
        "optimizer_minus_eigen_redundancy_bits": (
            optimizer_redundancy - eigen_result.redundancy_bits
        ),
        "optimizer_minus_eigen_synergy_bits": (
            optimizer_synergy - eigen_result.synergy_bits
        ),
        "constructed_gamma_star_union_bits": eigen_coupling.union_info_bits,
        "gamma_star_union_gap_bits": (
            eigen_coupling.union_info_bits - eigen_result.union_info_bits
        ),
        "gamma_star_maximum_singular_value": (
            eigen_coupling.maximum_singular_value
        ),
        "gamma_star_block_minimum_eigenvalue": (
            eigen_coupling.block_minimum_eigenvalue
        ),
        "gamma_star_posterior_rank": eigen_coupling.posterior_rank,
        "gamma_star_posterior_factor_residual": (
            eigen_coupling.posterior_factor_residual
        ),
        "gamma_star_channel_x_residual": eigen_coupling.channel_x_residual,
        "gamma_star_channel_y_residual": eigen_coupling.channel_y_residual,
        "gamma_star_degradation_x_spectral_excess": (
            eigen_coupling.degradation_x_spectral_excess
        ),
        "gamma_star_degradation_y_spectral_excess": (
            eigen_coupling.degradation_y_spectral_excess
        ),
        "gamma_star_minimum_information_eigenvalue": (
            eigen_coupling.minimum_information_eigenvalue
        ),
        **convergence,
        "optimizer_updates": optimizer_updates,
        "converged": converged,
        "right_censored": stopped_by == "time_limit",
    }


def add_paired_comparisons(result_rows: Sequence[Mapping[str, Any]]) -> list[ResultRow]:
    """Add native-minus-Gamma update differences to paired result rows.

    Args:
        result_rows: Result mappings with one native and one Gamma-star row
            for each dimension, repeat, and method.

    Returns:
        Copied result dictionaries containing ``paired_runs_converged`` and a
        paired update difference when both optimizers converged.
    """
    rows = [dict(row) for row in result_rows]
    paired = {
        (
            row["target_dimension"],
            row["repeat_index"],
            row["method"],
            row["initialization"],
        ): row
        for row in rows
    }

    for row in rows:
        pair_key = (
            row["target_dimension"],
            row["repeat_index"],
            row["method"],
        )
        native = paired[(*pair_key, NATIVE_START)]
        gamma_star = paired[(*pair_key, GAMMA_STAR_START)]
        pair_converged = bool(native["converged"] and gamma_star["converged"])
        row["paired_runs_converged"] = pair_converged
        row["native_minus_gamma_star_optimizer_updates"] = (
            native["optimizer_updates"] - gamma_star["optimizer_updates"]
            if pair_converged
            else ""
        )
    return rows


def save_results_csv(result_rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write all detailed experiment rows to one CSV file.

    Args:
        result_rows: Nonempty sequence of flat, consistently keyed mappings.
        path: Destination CSV path; missing parent directories are created.

    Returns:
        The resolved destination as a ``Path``.
    """
    rows = [dict(row) for row in result_rows]
    if not rows:
        raise ValueError("result_rows must not be empty")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def save_hyperparameters_yaml(
    hyperparameters: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write experiment hyperparameters to a reproducibility YAML file.

    Args:
        hyperparameters: YAML-serializable experiment settings and output paths.
        path: Destination YAML path; missing parent directories are created.

    Returns:
        The destination as a ``Path``.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(dict(hyperparameters), yaml_file, sort_keys=False)
    return output_path


def print_experiment_summary(
    result_rows: Sequence[Mapping[str, Any]],
    csv_path: str | Path,
) -> None:
    """Print convergence counts and method-versus-Eigen error summaries.

    Args:
        result_rows: Completed detailed experiment rows.
        csv_path: Location of the saved CSV displayed in the final message.

    Returns:
        ``None``; the summary is written to standard output.
    """
    rows = list(result_rows)
    dimensions = list(dict.fromkeys(int(row["target_dimension"]) for row in rows))
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    initializations = list(
        dict.fromkeys(str(row["initialization"]) for row in rows)
    )

    print("Original-covariance versus Eigen-PID Gamma* convergence")
    print(
        "dimension | method          | initialization                    | "
        "optimizer updates | converged median | stopping rule"
    )
    print("-" * 145)
    for dimension in dimensions:
        for method in methods:
            for initialization in initializations:
                group = [
                    row
                    for row in rows
                    if int(row["target_dimension"]) == dimension
                    and row["method"] == method
                    and row["initialization"] == initialization
                ]
                counts_text = ", ".join(
                    f"{row['optimizer_updates']}"
                    f"{'C' if row['converged'] else 'NC'}"
                    for row in group
                )
                converged_counts = [
                    float(row["optimizer_updates"])
                    for row in group
                    if row["converged"]
                ]
                median = np.median(converged_counts) if converged_counts else np.nan
                reasons = sorted({str(row["stopped_by"]) for row in group})
                print(
                    f"{dimension:9d} | {method:15s} | {initialization:33s} | "
                    f"{counts_text:28s} | {median:16.1f} | {', '.join(reasons)}"
                )

    print("\nMaximum converged |optimizer union - Eigen-PID union|:")
    for method in methods:
        for initialization in initializations:
            gaps = [
                abs(float(row["optimizer_minus_eigen_union_bits"]))
                for row in rows
                if row["method"] == method
                and row["initialization"] == initialization
                and row["converged"]
                and np.isfinite(row["optimizer_minus_eigen_union_bits"])
            ]
            maximum_gap = max(gaps) if gaps else np.nan
            print(f"  {method}, {initialization}: {maximum_gap:.3e} bits")

    print("Table suffixes: C = converged; NC = nonconverged lower bound.")
    print(f"Saved {len(rows)} detailed result rows to {Path(csv_path)}")


def _result_matrix(
    result_rows: Sequence[Mapping[str, Any]],
    dimensions: Sequence[int],
    repeats: int,
    method: str,
    initialization: str,
    field: str,
) -> np.ndarray:
    """Arrange one scalar result field into a dimension-by-repeat matrix.

    Args:
        result_rows: Detailed experiment rows.
        dimensions: Ordered target dimensions for matrix rows.
        repeats: Expected repeat count for matrix columns.
        method: Method label used to select rows.
        initialization: Initialization label used to select rows.
        field: Numeric or boolean result field to extract.

    Returns:
        Float array with shape ``(len(dimensions), repeats)``.
    """
    # Scalar fill value -> (number of dimensions, repeats).
    matrix = np.full((len(dimensions), repeats), np.nan, dtype=np.float64)
    dimension_indexes = {int(value): index for index, value in enumerate(dimensions)}
    for row in result_rows:
        if row["method"] != method or row["initialization"] != initialization:
            continue
        dimension_index = dimension_indexes[int(row["target_dimension"])]
        repeat_index = int(row["repeat_index"])
        matrix[dimension_index, repeat_index] = float(row[field])
    return matrix


def _masked_summary(
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute median, minimum, and maximum over valid repeats.

    Args:
        values: Float array shaped ``(dimensions, repeats)``.
        valid_mask: Boolean array of the same shape; ``True`` values are used.

    Returns:
        Three float arrays shaped ``(dimensions,)`` containing median,
        minimum, and maximum values, with ``NaN`` for empty dimensions.
    """
    masked = np.ma.array(values.astype(np.float64), mask=~valid_mask)
    # (number of dimensions, repeats) -> (number of dimensions,).
    median = np.ma.median(masked, axis=1).filled(np.nan)
    minimum = np.ma.min(masked, axis=1).filled(np.nan)
    maximum = np.ma.max(masked, axis=1).filled(np.nan)
    return median, minimum, maximum


def plot_iteration_comparison(
    result_rows: Sequence[Mapping[str, Any]],
    dimensions: Sequence[int],
    repeats: int,
    methods: Sequence[str],
    initializations: Sequence[str],
) -> Figure:
    """Plot raw optimizer update counts for both initializations.

    Args:
        result_rows: Detailed experiment rows.
        dimensions: Ordered balanced dimensions on the x-axis.
        repeats: Expected number of repeats per dimension.
        methods: Ordered method labels, normally GPID and Thin-PID.
        initializations: Native and Gamma-star initialization labels.

    Returns:
        A Matplotlib figure with one raw-update panel per method.
    """
    # One plot specification -> axes with shape (number of methods, 1).
    figure, axes = plt.subplots(
        len(methods), 1, figsize=(8, 4 * len(methods)), sharex=True, squeeze=False
    )
    dimension_values = np.asarray(dimensions, dtype=np.int64)
    initialization_colors = {
        NATIVE_START: "tab:red",
        GAMMA_STAR_START: "tab:green",
    }

    for method_index, method in enumerate(methods):
        raw_axis = axes[method_index, 0]
        for initialization in initializations:
            updates = _result_matrix(
                result_rows, dimensions, repeats, method, initialization,
                "optimizer_updates",
            )
            convergence_values = _result_matrix(
                result_rows, dimensions, repeats, method, initialization,
                "converged",
            )
            converged = np.isfinite(convergence_values) & convergence_values.astype(
                bool
            )
            median, minimum, maximum = _masked_summary(updates, converged)
            raw_axis.plot(
                dimension_values, median, marker="o",
                color=initialization_colors[initialization], label=initialization,
            )
            raw_axis.fill_between(
                dimension_values, minimum, maximum,
                color=initialization_colors[initialization], alpha=0.12,
            )

            # (number of dimensions,) -> (number of dimensions, repeats).
            repeated_dimensions = np.broadcast_to(
                dimension_values[:, None], updates.shape
            )
            if np.any(~converged):
                raw_axis.scatter(
                    repeated_dimensions[~converged], updates[~converged],
                    marker="x", s=45,
                    color=initialization_colors[initialization],
                    label=f"{initialization}: nonconverged lower bound",
                )

        raw_axis.set_title(f"{method}: completed optimizer updates")
        raw_axis.set_ylabel("Optimizer updates")
        raw_axis.set_xticks(dimension_values)
        raw_axis.grid(alpha=0.3)
        raw_axis.legend()

    axes[-1, 0].set_xlabel("Balanced target/source dimension")
    figure.suptitle(
        r"Raw optimizer updates by initialization",
        fontsize=14,
    )
    figure.tight_layout()
    return figure


def plot_pid_comparison(
    result_rows: Sequence[Mapping[str, Any]],
    dimensions: Sequence[int],
    repeats: int,
    methods: Sequence[str],
    initializations: Sequence[str],
) -> Figure:
    """Plot raw PID atoms and native-minus-Gamma-star atom differences.

    Args:
        result_rows: Detailed experiment rows.
        dimensions: Ordered balanced dimensions on the x-axis.
        repeats: Expected number of repeats per dimension.
        methods: Ordered method labels, normally GPID and Thin-PID.
        initializations: Native and Gamma-star initialization labels.

    Returns:
        A Matplotlib figure with raw and paired-difference rows per method for
        unique-X, unique-Y, redundancy, and synergy.
    """
    component_specs = (
        ("Unique X", "optimizer_uix_bits", "eigen_uix_bits"),
        ("Unique Y", "optimizer_uiy_bits", "eigen_uiy_bits"),
        ("Redundancy", "optimizer_redundancy_bits", "eigen_redundancy_bits"),
        ("Synergy", "optimizer_synergy_bits", "eigen_synergy_bits"),
    )
    # One plot specification -> axes with shape (2 * methods, PID components).
    figure, axes = plt.subplots(
        2 * len(methods),
        len(component_specs),
        figsize=(18, 4 * len(methods)),
        sharex=True,
        squeeze=False,
    )
    dimension_values = np.asarray(dimensions, dtype=np.int64)
    colors = {NATIVE_START: "tab:red", GAMMA_STAR_START: "tab:green"}
    method_colors = {
        method: f"C{method_index}"
        for method_index, method in enumerate(methods)
    }

    for method_index, method in enumerate(methods):
        convergence_by_initialization: dict[str, np.ndarray] = {}
        for initialization in initializations:
            convergence_values = _result_matrix(
                result_rows, dimensions, repeats, method, initialization,
                "converged",
            )
            convergence_by_initialization[initialization] = (
                np.isfinite(convergence_values)
                & convergence_values.astype(bool)
            )

        for component_index, (
            component_label,
            optimizer_field,
            eigen_field,
        ) in enumerate(component_specs):
            raw_axis = axes[2 * method_index, component_index]
            difference_axis = axes[2 * method_index + 1, component_index]
            eigen_values = _result_matrix(
                result_rows,
                dimensions,
                repeats,
                method,
                initializations[0],
                eigen_field,
            )
            eigen_valid = np.isfinite(eigen_values)
            eigen_median, eigen_minimum, eigen_maximum = _masked_summary(
                eigen_values,
                eigen_valid,
            )
            raw_axis.plot(
                dimension_values,
                eigen_median,
                marker="o",
                color="black",
                label="Eigen-PID",
            )
            raw_axis.fill_between(
                dimension_values,
                eigen_minimum,
                eigen_maximum,
                color="black",
                alpha=0.08,
            )

            optimizer_values: dict[str, np.ndarray] = {}
            for initialization in initializations:
                component_values = _result_matrix(
                    result_rows,
                    dimensions,
                    repeats,
                    method,
                    initialization,
                    optimizer_field,
                )
                optimizer_values[initialization] = component_values
                valid = (
                    convergence_by_initialization[initialization]
                    & np.isfinite(component_values)
                )
                median, minimum, maximum = _masked_summary(
                    component_values,
                    valid,
                )
                raw_axis.plot(
                    dimension_values,
                    median,
                    marker="o",
                    color=colors[initialization],
                    label=initialization,
                )
                raw_axis.fill_between(
                    dimension_values,
                    minimum,
                    maximum,
                    color=colors[initialization],
                    alpha=0.12,
                )

            # (number of dimensions, repeats) -> (number of dimensions, repeats).
            component_difference = (
                optimizer_values[NATIVE_START]
                - optimizer_values[GAMMA_STAR_START]
            )
            paired_converged = (
                convergence_by_initialization[NATIVE_START]
                & convergence_by_initialization[GAMMA_STAR_START]
                & np.isfinite(component_difference)
            )
            median, minimum, maximum = _masked_summary(
                component_difference,
                paired_converged,
            )
            difference_axis.plot(
                dimension_values,
                median,
                marker="o",
                color=method_colors[method],
                label=r"native result $-$ $\Gamma^*$ result",
            )
            difference_axis.fill_between(
                dimension_values,
                minimum,
                maximum,
                color=method_colors[method],
                alpha=0.18,
            )
            difference_axis.axhline(
                0,
                color="black",
                linewidth=1,
                linestyle="--",
            )

            raw_axis.set_title(f"{method}: {component_label}")
            difference_axis.set_title(f"Native - Gamma*: {component_label}")
            for axis in (raw_axis, difference_axis):
                axis.set_xticks(dimension_values)
                axis.grid(alpha=0.3)
            difference_axis.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(-3, 3),
            )

            if component_index == 0:
                raw_axis.set_ylabel(f"{method}\nPID value (bits)")
                difference_axis.set_ylabel(
                    f"{method}\nNative - Gamma* (bits)"
                )
                raw_axis.legend(fontsize=8)
                difference_axis.legend(fontsize=8)

    for axis in axes[-1]:
        axis.set_xlabel("Balanced dimension")
    figure.suptitle(
        "PID components: raw results and native-minus-Gamma* differences",
        fontsize=14,
    )
    figure.tight_layout()
    return figure


def save_figure(
    figure: Figure,
    path: str | Path,
    dpi: int = 300,
) -> Path:
    """Save one Matplotlib figure to disk.

    Args:
        figure: Matplotlib figure to save.
        path: Destination image path; missing parent directories are created.
        dpi: Positive output resolution in dots per inch.

    Returns:
        The destination as a ``Path``.
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def show_figures() -> None:
    """Display all currently open Matplotlib figures and return ``None``."""
    plt.show()
