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
    dpi: int = 300,
) -> Path:
    """Plot variance explained on held-out data by each retained PC.

    Inputs:
        variance_csv_path: str or Path, CSV produced by
            ``subj_pc_analysis.main``.
        output_path: str, Path, or None, destination PNG path. When None, save
            beside the CSV using the same filename stem.
        show_cumulative: bool, whether to overlay cumulative explained variance.
        dpi: int, resolution of the saved PNG.

    Output:
        figure_path: Path, location of the saved variance-explained figure.
    """

    csv_path = Path(variance_csv_path)
    variance_table = pd.read_csv(csv_path)
    required_columns = {
        "pc_index",
        "heldout_explained_variance_ratio",
    }
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
    explained_ratios = pd.to_numeric(
        variance_table["heldout_explained_variance_ratio"],
        errors="raise",
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(pc_indices)) or not np.all(
        np.isfinite(explained_ratios)
    ):
        raise ValueError("PC indices and explained-variance ratios must be finite.")
    if np.any(explained_ratios < 0.0):
        raise ValueError("Explained-variance ratios cannot be negative.")
    if len(np.unique(pc_indices)) != len(pc_indices):
        raise ValueError("PC indices must be unique.")

    order = np.argsort(pc_indices)
    pc_indices = pc_indices[order]
    explained_percent = 100.0 * explained_ratios[order]

    figure_width = min(18.0, max(8.0, 0.18 * len(pc_indices) + 6.0))
    figure, axis = plt.subplots(
        figsize=(figure_width, 6.0),
        constrained_layout=True,
    )
    axis.bar(
        pc_indices,
        explained_percent,
        color="#4C72B0",
        alpha=0.85,
        label="Per-PC held-out variance",
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
    axis.set_ylabel("Held-out variance explained (%)")
    axis.set_title("Held-out variance explained by principal component")
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
