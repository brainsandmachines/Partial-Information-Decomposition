"""Create matrix heatmaps from pairwise PID pipeline CSV results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pairwise_pid_matrices(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    model_order: list[str] | None = None,
    value_format: str = ".3f",
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
) -> dict[str, Path]:
    """Create and save unique-information, redundancy, and synergy matrices.

    Inputs:
        csv_path: str or Path, CSV created by run_pairwise_pid_pipeline.
        output_dir: str or Path, directory for the matrix CSVs and PNG figures.
        model_order: list[str] or None, optional model order and subset. When
            None, use the sorted union of model_1 and model_2 names.
        value_format: str, format specification used for cell annotations.
        cmap: str, matplotlib colormap name.
        figsize: tuple[float, float] or None, optional figure size in inches.
        dpi: int, resolution used when saving each PNG figure.

    Output:
        figure_paths: dict[str, Path], paths keyed by "unique_information",
            "redundancy", and "synergy".

    Unique-information cells follow UI(row_model relative to column_model).
    A direct ordered row uses unq1; when only the reverse row exists, its unq2
    is used. Redundancy and synergy average both directions when available.
    Every diagonal is zero and unavailable off-diagonal pairs remain NaN.
    """

    required_columns = {
        "model_1",
        "model_2",
        "red",
        "unq1",
        "unq2",
        "syn",
    }
    results = pd.read_csv(csv_path)
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(
            "Pairwise PID CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    if results[["model_1", "model_2"]].isna().any().any():
        raise ValueError("Columns 'model_1' and 'model_2' cannot contain missing values.")

    results = results.copy()
    results["model_1"] = results["model_1"].astype(str)
    results["model_2"] = results["model_2"].astype(str)
    duplicate_mask = results.duplicated(["model_1", "model_2"], keep=False)
    if duplicate_mask.any():
        duplicate_pairs = (
            results.loc[duplicate_mask, ["model_1", "model_2"]]
            .drop_duplicates()
            .apply(tuple, axis=1)
            .tolist()
        )
        raise ValueError(
            "Pairwise PID CSV contains duplicate ordered model pairs: "
            f"{duplicate_pairs}"
        )

    for column in ("red", "unq1", "unq2", "syn"):
        try:
            results[column] = pd.to_numeric(results[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Column {column!r} must contain numeric values.") from error

    if model_order is None:
        models = sorted(set(results["model_1"]).union(results["model_2"]))
    else:
        models = [str(model) for model in model_order]
        if len(models) != len(set(models)):
            raise ValueError("model_order cannot contain duplicate model names.")
    if not models:
        raise ValueError("No models are available to plot.")

    try:
        format(0.0, value_format)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid value_format: {value_format!r}") from error

    pair_rows = {
        (row.model_1, row.model_2): row
        for row in results.itertuples(index=False)
    }
    matrix_size = len(models)
    unique_matrix = np.full((matrix_size, matrix_size), np.nan, dtype=float)
    redundancy_matrix = np.full((matrix_size, matrix_size), np.nan, dtype=float)
    synergy_matrix = np.full((matrix_size, matrix_size), np.nan, dtype=float)
    np.fill_diagonal(unique_matrix, 0.0)
    np.fill_diagonal(redundancy_matrix, 0.0)
    np.fill_diagonal(synergy_matrix, 0.0)

    for row_index, row_model in enumerate(models):
        for column_index, column_model in enumerate(models):
            if row_index == column_index:
                continue

            direct = pair_rows.get((row_model, column_model))
            reverse = pair_rows.get((column_model, row_model))
            if direct is not None:
                unique_matrix[row_index, column_index] = float(direct.unq1)
            elif reverse is not None:
                unique_matrix[row_index, column_index] = float(reverse.unq2)

            redundancy_values = []
            synergy_values = []
            if direct is not None:
                redundancy_values.append(float(direct.red))
                synergy_values.append(float(direct.syn))
            if reverse is not None:
                redundancy_values.append(float(reverse.red))
                synergy_values.append(float(reverse.syn))
            if redundancy_values:
                redundancy_matrix[row_index, column_index] = float(
                    np.mean(redundancy_values)
                )
                synergy_matrix[row_index, column_index] = float(
                    np.mean(synergy_values)
                )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matrix_specs = [
        (
            "unique_information",
            unique_matrix,
            "Unique information across model pairs",
            r"Unique information: UI(row \ column)",
        ),
        (
            "redundancy",
            redundancy_matrix,
            "Redundancy across model pairs",
            "Redundancy",
        ),
        (
            "synergy",
            synergy_matrix,
            "Synergy across model pairs",
            "Synergy",
        ),
    ]
    figure_paths: dict[str, Path] = {}
    figure_size = figsize or (
        max(6.0, min(18.0, 0.75 * matrix_size + 3.0)),
        max(6.0, min(18.0, 0.75 * matrix_size + 3.0)),
    )

    for name, matrix, title, colorbar_label in matrix_specs:
        matrix_frame = pd.DataFrame(matrix, index=models, columns=models)
        matrix_frame.index.name = "row_model"
        matrix_frame.columns.name = "column_model"
        matrix_frame.to_csv(output_path / f"{name}_matrix.csv")

        figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
        plot_cmap = plt.get_cmap(cmap).copy()
        plot_cmap.set_bad(color="#d9d9d9")
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            cmap=plot_cmap,
            aspect="equal",
        )
        axis.set_xticks(np.arange(matrix_size))
        axis.set_yticks(np.arange(matrix_size))
        axis.set_xticklabels(
            models,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticklabels(models)
        axis.set_xlabel("Column model")
        axis.set_ylabel("Row model")
        axis.set_title(title)
        axis.set_xticks(np.arange(-0.5, matrix_size, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, matrix_size, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.8)
        axis.tick_params(which="minor", bottom=False, left=False)

        for row_index in range(matrix_size):
            for column_index in range(matrix_size):
                value = matrix[row_index, column_index]
                if np.isnan(value):
                    annotation = "NA"
                    text_color = "#555555"
                else:
                    annotation = format(value, value_format)
                    red, green, blue, _ = image.cmap(image.norm(value))
                    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                    text_color = "black" if luminance > 0.55 else "white"
                axis.text(
                    column_index,
                    row_index,
                    annotation,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)
        figure_path = output_path / f"{name}_matrix.png"
        figure.savefig(figure_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        figure_paths[name] = figure_path

    return figure_paths


if __name__ == "__main__":
    plot_pairwise_pid_matrices(
        csv_path="path/to/pairwise_pid_results.csv",
        output_dir="path/to/figures",
    )
