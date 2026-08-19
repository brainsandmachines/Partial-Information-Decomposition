"""Render an evil-twin summary CSV in the shared simulation table style."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Simulations.evil_twin.evil_twin_pid_batch_utils import (
    SUMMARY_METADATA_FIELDS,
    SUMMARY_TWINS,
    SUMMARY_VALUE_FIELDS,
    save_summary_table_images,
    summary_fieldnames,
)


# Edit only this dictionary to choose the summary CSV and plot metadata.
CONFIG = {
    "summary_csv": "simulation_results/evil_twin_config/evil_twin_uncorrected_summary.csv",
    "output_dir": None,
    "prefix": None,
    "n_samples": None,
    "dimension": None,
    "bias_correction": None,
}


def load_summary_rows(summary_csv: Path | str) -> list[dict]:
    """Load and validate rows from an evil-twin mean-summary CSV.

    Inputs:
        summary_csv: Path or str pointing to a CSV created by
            ``save_summary_csv``.

    Outputs:
        list[dict], parsed summary rows with integer experiment dimensions,
            boolean bias metadata, PID/MI floats, and integer result counts.
    """
    summary_path = Path(summary_csv)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Evil-twin summary CSV not found: {summary_path}")

    required_fields = summary_fieldnames()
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        csv_fields = reader.fieldnames or []
        missing_fields = [field for field in required_fields if field not in csv_fields]
        if missing_fields:
            raise ValueError(
                f"Summary CSV is missing required columns: {missing_fields}"
            )

        rows = []
        for row_number, csv_row in enumerate(reader, start=2):
            method = (csv_row.get("method") or "").strip()
            if not method:
                raise ValueError(
                    f"Summary CSV row {row_number} has an empty method value."
                )

            try:
                bias_value = (csv_row.get("bias_correction") or "").strip().lower()
                if bias_value not in {"true", "false"}:
                    raise ValueError(
                        "bias_correction must be written as True or False"
                    )
                parsed_row = {
                    "method": method,
                    "n_samples": int(csv_row["n_samples"]),
                    "dimension": int(csv_row["dimension"]),
                    "bias_correction": bias_value == "true",
                }
                for twin in SUMMARY_TWINS:
                    for field in SUMMARY_VALUE_FIELDS:
                        column = f"{twin}_{field}_mean"
                        value = (csv_row.get(column) or "").strip()
                        parsed_row[column] = "" if value == "" else float(value)
                    for count_name in ("n_ok", "n_error"):
                        column = f"{twin}_{count_name}"
                        value = (csv_row.get(column) or "").strip()
                        parsed_row[column] = 0 if value == "" else int(value)
            except ValueError as error:
                raise ValueError(
                    f"Summary CSV row {row_number} contains invalid metadata, "
                    "PID, MI, or count values."
                ) from error
            rows.append(parsed_row)

    if not rows:
        raise ValueError(f"Summary CSV contains no method rows: {summary_path}")
    return rows


def plot_summary_csv(
    summary_csv: Path | str,
    output_dir: Path | str | None = None,
    prefix: str | None = None,
    n_samples: int | None = None,
    dimension: int | None = None,
    bias_correction: bool | None = None,
) -> list[Path]:
    """Create Sonic and Shadow plots from an existing evil-twin summary CSV.

    Inputs:
        summary_csv: Path or str pointing to an evil-twin summary CSV.
        output_dir: optional Path or str for output images; defaults to the CSV
            directory.
        prefix: optional filename prefix; defaults to the CSV stem without a
            trailing ``_summary``.
        n_samples: optional positive int overriding the value stored in the CSV.
        dimension: optional positive int overriding the value stored in the CSV.
        bias_correction: optional bool overriding the value stored in the CSV.

    Outputs:
        list[Path], written Sonic and Shadow PNG paths. A twin with no
            successful methods is omitted.
    """
    summary_path = Path(summary_csv)
    rows = load_summary_rows(summary_path)
    summary_metadata = {}
    for field in SUMMARY_METADATA_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(
                f"Summary CSV contains inconsistent {field} values: {sorted(values)}"
            )
        summary_metadata[field] = next(iter(values))

    plot_n_samples = summary_metadata["n_samples"] if n_samples is None else n_samples
    plot_dimension = summary_metadata["dimension"] if dimension is None else dimension
    plot_bias_correction = (
        summary_metadata["bias_correction"]
        if bias_correction is None
        else bias_correction
    )
    if plot_n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if plot_dimension <= 0:
        raise ValueError("dimension must be positive.")

    plot_output_dir = Path(output_dir) if output_dir is not None else summary_path.parent
    plot_prefix = prefix
    if plot_prefix is None:
        plot_prefix = summary_path.stem
        if plot_prefix.endswith("_summary"):
            plot_prefix = plot_prefix.removesuffix("_summary")
    if not plot_prefix:
        raise ValueError("prefix must be non-empty when provided.")

    plot_config = {
        "n_samples": plot_n_samples,
        "dx1": plot_dimension,
        "dx2": plot_dimension,
        "dt": plot_dimension,
        "bias_correction": plot_bias_correction,
    }
    return save_summary_table_images(
        rows,
        plot_output_dir,
        plot_prefix,
        plot_config,
    )


def main(config: dict | None = None) -> list[Path]:
    """Render one configured evil-twin summary CSV and print output paths.

    Inputs:
        config: optional dict containing summary_csv, output_dir, prefix,
            n_samples, dimension, and bias_correction. When None, the editable
            module-level ``CONFIG`` dictionary is used.

    Outputs:
        list[Path], Sonic and Shadow PNG paths written by ``plot_summary_csv``.
    """
    plot_settings = CONFIG if config is None else config
    required_keys = {
        "summary_csv",
        "output_dir",
        "prefix",
        "n_samples",
        "dimension",
        "bias_correction",
    }
    missing_keys = required_keys.difference(plot_settings)
    if missing_keys:
        raise ValueError(f"Missing plot CONFIG keys: {sorted(missing_keys)}")

    image_paths = plot_summary_csv(
        summary_csv=plot_settings["summary_csv"],
        output_dir=plot_settings["output_dir"],
        prefix=plot_settings["prefix"],
        n_samples=plot_settings["n_samples"],
        dimension=plot_settings["dimension"],
        bias_correction=plot_settings["bias_correction"],
    )
    for image_path in image_paths:
        print(f"Saved evil-twin summary plot to: {image_path}")
    return image_paths


if __name__ == "__main__":
    main()
