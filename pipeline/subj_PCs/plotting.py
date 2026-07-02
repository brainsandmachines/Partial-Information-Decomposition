"""Plot held-out explained variance from subject-level PCA analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_heldout_variance_explained(
    variance_csv_path: str | Path,
    output_path: str | Path | None = None,
    *,
    show_cumulative: bool = True,
    show_training: bool = False,
    plot_training_minus_heldout: bool = False,
    separate_panels: bool = False,
    number_of_pcs: int | None = None,
    dpi: int = 300,
) -> Path:
    """Plot variance explained on held-out data by each retained PC.

    Inputs:
        variance_csv_path: str or Path, CSV produced by
            ``subj_pc_analysis.main``.
        output_path: str, Path, or None, destination PNG path. When None, save
            beside the CSV using the same filename stem.
        show_cumulative: bool, whether to overlay cumulative explained variance.
        show_training: bool, whether to overlay the per-PC training explained
            variance after converting its stored ratios to percentages.
        plot_training_minus_heldout: bool, whether to plot the difference between
            training and held-out explained variance.
        separate_panels: bool, whether to plot held-out and training explained
            variance as two bar graphs next to each other. This mode plots
            both datasets regardless of ``show_training`` and cannot be
            combined with ``plot_training_minus_heldout``.
        number_of_pcs: int or None, number of PCs to include in the plot.
        dpi: int, resolution of the saved PNG.

    Output:
        figure_path: Path, location of the saved variance-explained figure.
    """

    if separate_panels and plot_training_minus_heldout:
        raise ValueError(
            "separate_panels cannot be combined with "
            "plot_training_minus_heldout."
        )

    csv_path = Path(variance_csv_path)
    variance_table = pd.read_csv(csv_path)
    required_columns = {
        "pc_index",
        "heldout_explained_variance_ratio",
    }

    required_columns.add("training_explained_variance_ratio")

    missing_columns = required_columns.difference(variance_table.columns)
    if missing_columns:
        raise ValueError(
            "Held-out variance CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    if variance_table.empty:
        raise ValueError("Held-out variance CSV contains no PC rows.")

    pc_indices = pd.to_numeric(
        variance_table["pc_index"],
        errors="raise",
    ).to_numpy()

    if number_of_pcs is not None:
        pc_indices = pc_indices[:number_of_pcs]

    explained_ratios = pd.to_numeric(
        variance_table["heldout_explained_variance_ratio"],
        errors="raise",
    ).to_numpy(dtype=float)
    if number_of_pcs is not None:
        explained_ratios = explained_ratios[:number_of_pcs]

    training_ratios = None

    training_ratios = pd.to_numeric(
        variance_table["training_explained_variance_ratio"],
        errors="raise",
    ).to_numpy(dtype=float)
    if number_of_pcs is not None:
        training_ratios = training_ratios[:number_of_pcs]

    if not np.all(np.isfinite(pc_indices)) or not np.all(
        np.isfinite(explained_ratios)
    ):
        raise ValueError("PC indices and explained-variance ratios must be finite.")
    if np.any(explained_ratios < 0.0):
        raise ValueError("Explained-variance ratios cannot be negative.")
    if training_ratios is not None:
        if not np.all(np.isfinite(training_ratios)):
            raise ValueError(
                "Training explained-variance ratios must be finite."
            )
        if np.any(training_ratios < 0.0):
            raise ValueError(
                "Training explained-variance ratios cannot be negative."
            )
    if len(np.unique(pc_indices)) != len(pc_indices):
        raise ValueError("PC indices must be unique.")

    order = np.argsort(pc_indices)
    pc_indices = pc_indices[order]
    explained_percent = 100.0 * explained_ratios[order]
    training_percent = (
        100.0 * training_ratios[order]
        if training_ratios is not None
        else None
    )

    panel_width = min(18.0, max(8.0, 0.18 * len(pc_indices) + 6.0))
    if separate_panels:
        figure, (heldout_axis, training_axis) = plt.subplots(
            ncols=2,
            figsize=(min(30.0, 2.0 * panel_width), 6.0),
            sharey=True,
            constrained_layout=True,
        )
        heldout_axis.bar(
            pc_indices,
            explained_percent,
            width=0.8,
            color="#000000",
            alpha=0.85,
            label="Per-PC held-out variance",
        )
        training_axis.bar(
            pc_indices,
            training_percent,
            width=0.8,
            color="#12AA35",
            alpha=0.85,
            label="Per-PC training variance",
        )

        if show_cumulative:
            heldout_axis.plot(
                pc_indices,
                np.cumsum(explained_percent),
                color="#DD8452",
                linewidth=2.0,
                label="Cumulative held-out variance",
            )

        heldout_axis.set_title("Held-out variance explained")
        training_axis.set_title("Training variance explained")
        for panel_axis in (heldout_axis, training_axis):
            panel_axis.set_xlabel("PC index")
            panel_axis.set_ylabel("Variance explained (%)")
            panel_axis.grid(axis="y", alpha=0.3)
            panel_axis.legend()
    else:
        figure, axis = plt.subplots(
            figsize=(panel_width, 6.0),
            constrained_layout=True,
        )
        bar_alpha_heldout = 0.0 if training_percent is not None else 0.85
        bar_alpha_training = 0.85 if training_percent is not None else 0.85
        trn_minus_heldout_alpha = (
            0.85 if plot_training_minus_heldout else 0.0
        )

        trn_minus_heldout = (
            training_percent - explained_percent
            if training_percent is not None
            else None
        )

        if plot_training_minus_heldout:
            axis.bar(
                pc_indices,
                trn_minus_heldout,
                width=0.8,
                color="#0F0093",
                alpha=trn_minus_heldout_alpha,
                label="Per-PC training minus held-out variance",
            )

        if not plot_training_minus_heldout:
            axis.bar(
                pc_indices,
                explained_percent,
                width=0.8,
                color="#000000",
                alpha=bar_alpha_heldout,
                label="Per-PC held-out variance",
            )

        if training_percent is not None and show_training:
            axis.bar(
                pc_indices,
                training_percent,
                width=0.8,
                color="#12AA35",
                alpha=bar_alpha_training,
                label="Per-PC training variance",
            )

        if show_cumulative:
            axis.plot(
                pc_indices,
                np.cumsum(explained_percent),
                color="#DD8452",
                linewidth=2.0,
                label="Cumulative held-out variance",
            )

        axis.set_xlabel("PC index")
        axis.set_ylabel(
            "Variance explained (%)"
            if show_training or plot_training_minus_heldout
            else "Held-out variance explained (%)"
        )
        axis.set_title(
            "Training Explained Variance Minus Held-out Variance Explained by Principal Component"
            if plot_training_minus_heldout
            else "Training and held-out variance explained by principal component"
            if show_training
            else "Held-out variance explained by principal component"
        )
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

    figure_path = (
        Path(output_path)
        if output_path is not None
        else csv_path.with_suffix(".png")
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return figure_path




if __name__ == "__main__":
    plot_heldout_variance_explained(
        variance_csv_path="/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/subj_PCs/subj01_results/subj01_heldout_pca_variance_explained.csv",
        output_path="/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/subj_PCs/subj01_heldout_pca_variance_explained.png",
        show_cumulative=False,
        show_training=False,
        plot_training_minus_heldout=True,
        dpi=600,
        number_of_pcs=300,
    )
