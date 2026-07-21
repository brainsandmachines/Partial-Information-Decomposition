"""Plot per-PC ridge correlations for selected models on one figure."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Sequence

import matplotlib.pyplot as plt


PC_CORRELATION_COLUMN = re.compile(r"^pc_(\d+)_correlation$")


@dataclass(frozen=True)
class ModelCorrelationSeries:
    """Store one model's layer and its correlations at individual PC indexes."""

    model_name: str
    layer_index: int
    pc_indexes: tuple[int, ...]
    correlations: tuple[float, ...]


def load_pc_correlation_series(
    csv_paths: Sequence[str | Path],
    model_names: Sequence[str],
) -> list[ModelCorrelationSeries]:
    """Load selected models from CSV files produced by pcIndex_predictions.py.

    Inputs:
        csv_paths: sequence of str or Path objects naming one or more per-PC
            correlation CSV files produced by ``pcIndex_predictions.py``.
        model_names: sequence of str model names to load. Their order is kept
            in the returned list and, consequently, in the plot legend.

    Outputs:
        list[ModelCorrelationSeries] with one item for each requested model.

    Raises:
        ValueError: If inputs are empty or duplicated, a CSV has an invalid
            schema/value, or a requested model is missing or occurs more than
            once across the supplied CSV files.
        FileNotFoundError: If a supplied CSV path does not exist.
    """

    paths = [Path(csv_path) for csv_path in csv_paths]
    requested_models = [model_name.strip() for model_name in model_names]
    if not paths:
        raise ValueError("At least one CSV path must be supplied.")
    if not requested_models or any(not model_name for model_name in requested_models):
        raise ValueError("At least one non-empty model name must be supplied.")

    duplicate_requests = sorted(
        {
            model_name
            for model_name in requested_models
            if requested_models.count(model_name) > 1
        }
    )
    if duplicate_requests:
        raise ValueError(
            "The model list contains duplicates: " + ", ".join(duplicate_requests)
        )

    requested_set = set(requested_models)
    loaded_series: dict[str, ModelCorrelationSeries] = {}
    model_sources: dict[str, Path] = {}

    for csv_path in paths:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Correlation CSV does not exist: {csv_path}")

        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"Correlation CSV is empty: {csv_path}")

            missing_columns = {"model_name", "layer_index"}.difference(fieldnames)
            if missing_columns:
                raise ValueError(
                    f"{csv_path} is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )

            pc_columns = []
            for column_name in fieldnames:
                column_match = PC_CORRELATION_COLUMN.fullmatch(column_name)
                if column_match:
                    pc_columns.append((int(column_match.group(1)), column_name))
            pc_columns.sort(key=lambda column: column[0])
            if not pc_columns:
                raise ValueError(
                    f"{csv_path} has no columns named pc_<index>_correlation."
                )

            pc_indexes = tuple(pc_index for pc_index, _ in pc_columns)
            if len(pc_indexes) != len(set(pc_indexes)):
                raise ValueError(f"{csv_path} contains duplicate PC indexes.")

            for row_number, row in enumerate(reader, start=2):
                model_name = (row.get("model_name") or "").strip()
                if model_name not in requested_set:
                    continue
                if model_name in loaded_series:
                    first_source = model_sources[model_name]
                    raise ValueError(
                        f"Model {model_name!r} occurs more than once across the CSV "
                        f"inputs (first in {first_source}, again in {csv_path} at "
                        f"row {row_number})."
                    )

                raw_layer_index = (row.get("layer_index") or "").strip()
                try:
                    numeric_layer_index = float(raw_layer_index)
                    if (
                        not math.isfinite(numeric_layer_index)
                        or not numeric_layer_index.is_integer()
                    ):
                        raise ValueError
                    layer_index = int(numeric_layer_index)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid layer_index for model {model_name!r} in "
                        f"{csv_path} at row {row_number}: {raw_layer_index!r}"
                    ) from error

                try:
                    correlations = tuple(
                        float(row[column_name]) for _, column_name in pc_columns
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid correlation value for model {model_name!r} in "
                        f"{csv_path} at row {row_number}."
                    ) from error

                loaded_series[model_name] = ModelCorrelationSeries(
                    model_name=model_name,
                    layer_index=layer_index,
                    pc_indexes=pc_indexes,
                    correlations=correlations,
                )
                model_sources[model_name] = csv_path

    missing_models = [
        model_name for model_name in requested_models if model_name not in loaded_series
    ]
    if missing_models:
        raise ValueError(
            "Requested models were not found in the supplied CSV files: "
            + ", ".join(missing_models)
        )

    return [loaded_series[model_name] for model_name in requested_models]


def plot_pc_correlations(
    csv_paths: Sequence[str | Path],
    model_names: Sequence[str],
    output_path: str | Path,
    *,
    title: str = "Ridge prediction correlation by PC index",
    dpi: int = 300,
) -> Path:
    """Plot selected models' per-PC correlations together and save the figure.

    Inputs:
        csv_paths: sequence of str or Path objects naming CSV files produced by
            ``pcIndex_predictions.py``.
        model_names: sequence of str model names to draw on the shared axes.
        output_path: str or Path for the saved figure. The file extension
            selects the Matplotlib output format.
        title: str displayed as the figure title.
        dpi: int resolution used when saving the figure.

    Outputs:
        Path to the saved plot. Each curve is annotated with its layer index,
        and the legend follows the supplied model order.

    Raises:
        ValueError: If ``dpi`` is not positive or a selected series has no
            finite correlation values.
    """

    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    selected_series = load_pc_correlation_series(csv_paths, model_names)
    figure_width = min(18.0, 10.0 + 0.2 * len(selected_series))
    figure, axis = plt.subplots(figsize=(figure_width, 6.5))

    try:
        for series_index, series in enumerate(selected_series):
            (line,) = axis.plot(
                series.pc_indexes,
                series.correlations,
                linewidth=1.8,
                label=series.model_name,
            )

            finite_positions = [
                position
                for position, correlation in enumerate(series.correlations)
                if math.isfinite(correlation)
            ]
            if not finite_positions:
                raise ValueError(
                    f"Model {series.model_name!r} has no finite correlations to plot."
                )

            annotation_fraction = (series_index + 1) / (len(selected_series) + 1)
            finite_list_index = round(
                annotation_fraction * (len(finite_positions) - 1)
            )
            annotation_position = finite_positions[finite_list_index]
            axis.annotate(
                str(series.layer_index),
                xy=(
                    series.pc_indexes[annotation_position],
                    series.correlations[annotation_position],
                ),
                xytext=(0, 7),
                textcoords="offset points",
                color=line.get_color(),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )

        lower_limit, upper_limit = axis.get_ylim()
        y_span = upper_limit - lower_limit
        axis.set_ylim(lower_limit, upper_limit + 0.08 * y_span)
        axis.set_xlabel("PC index (zero-based)")
        axis.set_ylabel("Pearson correlation")
        axis.set_title(f"{title}\nNumbers above curves indicate layer indexes")
        axis.grid(alpha=0.3)
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=max(6, 9 - len(selected_series) // 12),
        )
        figure.tight_layout()

        figure_path = Path(output_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)

    return figure_path


def main() -> Path:
    """Run a three-model example using the alpha-search correlation CSV.

    Inputs:
        None.

    Outputs:
        Path to the saved three-model example figure.
    """

    ridge_analysis_dir = Path(__file__).resolve().parent
    csv_paths = [
        ridge_analysis_dir
        / "alpha(1,50,100)_GPU_200_pcs_correlations_by_model.csv"
    ]
    model_names = [
        "nf_resnet50_classification",
        "hardcorenas_f_classification",
        "eca_nfnet_l0_classification",
    ]
    output_path = ridge_analysis_dir / "alpha_GPU_200_pcs_three_models.png"

    figure_path = plot_pc_correlations(
        csv_paths=csv_paths,
        model_names=model_names,
        output_path=output_path,
        title="Alpha (1, 50, 100): ridge correlation by PC index",
    )
    print(f"Saved correlation plot to {figure_path}")
    return figure_path


if __name__ == "__main__":
    main()
