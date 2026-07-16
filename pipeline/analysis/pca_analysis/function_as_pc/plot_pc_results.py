"""Plot PID and mutual information as target PCs are accumulated."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PID_SERIES = (
    ("red", "Redundancy"),
    ("unq1", "Unique source 1"),
    ("unq2", "Unique source 2"),
    ("syn", "Synergy"),
)
MI_SERIES = (
    ("bi_mi_1", "MI source 1-target"),
    ("bi_mi_2", "MI source 2-target"),
    ("tri_mi", "Trivariate MI"),
)


def plot_pid_mi_as_function_of_pcs(
    pair_results: dict[int, dict[str, Any]],
    model_1_name: str,
    model_2_name: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Plot absolute and trivariate-MI-normalized results for one model pair.

    Inputs:
        pair_results: dict mapping cumulative target-PC counts to dictionaries
            containing ``pid`` and ``mi`` values.
        model_1_name: str containing the first source model name.
        model_2_name: str containing the second source model name.
        output_dir: str or Path naming the directory for the two figures.

    Outputs:
        tuple[Path, Path] containing the absolute-value figure path followed by
        the trivariate-MI-normalized figure path.
    """

    pc_counts = sorted(pair_results)
    tri_mi = np.asarray(
        [float(pair_results[pc_count]["mi"]["tri_mi"]) for pc_count in pc_counts]
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_model_1 = model_1_name.replace("/", "_").replace("\\", "_")
    safe_model_2 = model_2_name.replace("/", "_").replace("\\", "_")
    figure_paths = []

    for normalized in (False, True):
        denominator = tri_mi if normalized else np.ones_like(tri_mi)
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

        for key, label in PID_SERIES:
            values = np.asarray(
                [float(pair_results[pc_count]["pid"][key]) for pc_count in pc_counts]
            )
            axes[0].plot(pc_counts, values / denominator, marker="o", label=label)

        for key, label in MI_SERIES:
            values = np.asarray(
                [float(pair_results[pc_count]["mi"][key]) for pc_count in pc_counts]
            )
            axes[1].plot(pc_counts, values / denominator, marker="o", label=label)

        axes[0].set_title("PID components")
        axes[1].set_title("Mutual information")
        for axis in axes:
            axis.set_xlabel("Number of target PCs")
            axis.set_xticks(pc_counts)
            axis.grid(alpha=0.3)
            axis.legend()

        y_label = "Fraction of trivariate MI" if normalized else "Information (bits)"
        axes[0].set_ylabel(y_label)
        axes[1].set_ylabel(y_label)
        scale_name = "Normalized by trivariate MI" if normalized else "Absolute values"
        figure.suptitle(
            f"PID and MI as a function of target PCs\n"
            f"{model_1_name} vs {model_2_name} — {scale_name}"
        )
        figure.tight_layout()

        suffix = "normalized_by_tri_mi" if normalized else "absolute"
        figure_path = output_path / (
            f"{safe_model_1}__{safe_model_2}_pid_mi_{suffix}.png"
        )
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        figure_paths.append(figure_path)

    return figure_paths[0], figure_paths[1]
