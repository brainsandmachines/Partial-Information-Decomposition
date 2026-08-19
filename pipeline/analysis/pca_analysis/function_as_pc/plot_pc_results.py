"""Plot PID and mutual information as target PCs are accumulated."""

from __future__ import annotations

import pickle
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


def plot_pc_results_from_pickle(
    pkl_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    name: str | None = None,
) -> tuple[Path, Path]:
    """Load one model pair's PC results and save its plots.

    Inputs:
        pkl_path: str or Path pointing to a trusted pickle containing the
            cumulative target-PC result mapping.
        model_name: str containing both source model names separated by
            ``"__"``, for example ``"RN50_clip__ghostnet_100_classification"``.
        output_dir: str or Path naming the directory for the two figures.
        name: optional str to include in the figure title and output filenames.

    Outputs:
        tuple[Path, Path] containing the absolute-value figure path followed by
        the trivariate-MI-normalized figure path.

    Raises:
        ValueError: If ``model_name`` does not contain two non-empty model names.
        TypeError: If the pickle does not contain a dictionary.
    """

    model_names = model_name.split("__", maxsplit=1)
    if len(model_names) != 2 or not all(model_names):
        raise ValueError(
            "model_name must contain two model names separated by '__'"
        )

    pickle_path = Path(pkl_path)
    with pickle_path.open("rb") as results_file:
        pair_results = pickle.load(results_file)

    if not isinstance(pair_results, dict):
        raise TypeError("The results pickle must contain a dictionary")

    return plot_pid_mi_as_function_of_pcs(
        pair_results=pair_results,
        model_1_name=model_names[0],
        model_2_name=model_names[1],
        output_dir=output_dir,
        name=name,
    )


def plot_pid_mi_as_function_of_pcs(
    pair_results: dict[int, dict[str, Any]],
    model_1_name: str,
    model_2_name: str,
    output_dir: str | Path,
    name: str | None = None,
) -> tuple[Path, Path]:
    """Plot absolute and trivariate-MI-normalized results for one model pair.

    Inputs:
        pair_results: dict mapping cumulative target-PC counts to dictionaries
            containing ``pid`` and ``mi`` values.
        model_1_name: str containing the first source model name.
        model_2_name: str containing the second source model name.
        output_dir: str or Path naming the directory for the two figures.
        name: optional str to include in the figure title and output filenames.

    Outputs:
        tuple[Path, Path] containing the absolute-value figure path followed by
        the trivariate-MI-normalized figure path.
    """

    pc_counts = sorted(pair_results)
    tri_mi = np.asarray(
        [float(pair_results[pc_count]["mi"]["tri_mi"]) for pc_count in pc_counts]
    )
    maximum_pc_count = max(pc_counts)
    x_ticks = np.arange(20, maximum_pc_count + 1, 20, dtype=int)
    if not x_ticks.size:
        x_ticks = np.asarray([maximum_pc_count])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_model_1 = model_1_name.replace("/", "_").replace("\\", "_")
    safe_model_2 = model_2_name.replace("/", "_").replace("\\", "_")
    safe_name = name.strip().replace("/", "_").replace("\\", "_") if name else ""
    figure_paths = []

    for normalized in (False, True):
        denominator = tri_mi if normalized else np.ones_like(tri_mi)
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

        for key, label in PID_SERIES:
            values = np.asarray(
                [float(pair_results[pc_count]["pid"][key]) for pc_count in pc_counts]
            )
            displayed_values = values / denominator
            legend_label = label
            if key in {"unq1", "unq2"}:
                finite_positions = np.flatnonzero(np.isfinite(displayed_values))
                if finite_positions.size:
                    minimum_position = finite_positions[
                        np.argmin(displayed_values[finite_positions])
                    ]
                    maximum_position = finite_positions[
                        np.argmax(displayed_values[finite_positions])
                    ]
                    minimum_value = displayed_values[minimum_position]
                    maximum_value = displayed_values[maximum_position]
                    value_unit = " bits" if not normalized else ""
                    legend_label = (
                        f"{label} (min {minimum_value:.4g}{value_unit} at "
                        f"PC {pc_counts[minimum_position]}; max "
                        f"{maximum_value:.4g}{value_unit} at "
                        f"PC {pc_counts[maximum_position]})"
                    )
                else:
                    legend_label = f"{label} (minimum and maximum unavailable)"
            axes[0].plot(
                pc_counts,
                displayed_values,
                marker="o",
                label=legend_label,
            )

        for key, label in MI_SERIES:
            values = np.asarray(
                [float(pair_results[pc_count]["mi"][key]) for pc_count in pc_counts]
            )
            axes[1].plot(pc_counts, values / denominator, marker="o", label=label)

        axes[0].set_title("PID components")
        axes[1].set_title("Mutual information")
        for axis in axes:
            axis.set_xlabel("Number of target PCs")
            axis.set_xticks(x_ticks)
            axis.grid(alpha=0.3)
            axis.legend()

        y_label = "Fraction of trivariate MI" if normalized else "Information (bits)"
        axes[0].set_ylabel(y_label)
        axes[1].set_ylabel(y_label)
        scale_name = "Normalized by trivariate MI" if normalized else "Absolute values"
        name_prefix = f"{name.strip()} — " if name and name.strip() else ""
        figure.suptitle(
            f"{name_prefix}PID and MI as a function of target PCs\n"
            f"{model_1_name} vs {model_2_name} — {scale_name}"
        )
        figure.tight_layout()

        suffix = "normalized_by_tri_mi" if normalized else "absolute"
        filename_prefix = f"{safe_name}_" if safe_name else ""
        figure_path = output_path / (
            f"{filename_prefix}{safe_model_1}__{safe_model_2}_pid_mi_{suffix}.png"
        )
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        figure_paths.append(figure_path)

    return figure_paths[0], figure_paths[1]



if __name__ == "__main__":
    model1_name = "RN50_clip"
    model2_name = "ResNet50-SimCLR_selfsupervised"
    model_pair_name = f"{model1_name}__{model2_name}"
    pickle_path = Path("/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/pipeline/analysis/pca_analysis/function_as_pc/results/NORIDGE_RN50_clip_ResNet50-SimCLR_selfsupervised/RN50_clip__ResNet50-SimCLR_selfsupervised_pc_results.pkl")
    output_dir = Path("/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/pipeline/analysis/pca_analysis/function_as_pc/plots/RN50_clip_ResNet50-SimCLR_selfsupervised/no_ridge_RN50_clip_ResNet50-SimCLR_selfsupervised.png")
    plot_pc_results_from_pickle(
        pkl_path=pickle_path,
        model_name=model_pair_name,
        output_dir=output_dir,
        name = "No_Ridge_RN50_clip vs ResNet50-SimCLR_selfsupervised"
    )
