"""Plot Sonic and Shadow PID atoms from an evil-twin results CSV."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from statistics import fmean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG = {
    "results_csv": "simulation_results/evil_twin_config/evil_twin_uncorrected.csv",
    "output_path": None,
}

PID_ATOMS = (
    ("red", "Redundancy"),
    ("unq1", "Unique X1"),
    ("unq2", "Unique X2"),
    ("syn", "Synergy"),
)
TWINS = ("sonic", "shadow")


def plot_pid_atoms(
    results_csv: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Plot mean PID atoms for Sonic and Shadow from detailed result rows.

    Inputs:
        results_csv: Path or str, CSV produced by the evil-twin runner.
        output_path: optional Path or str for the PNG. By default, the image is
            saved beside the CSV with ``_pid_atoms`` appended to its stem.

    Outputs:
        Path, the saved grouped-bar-chart image.
    """
    results_path = Path(results_csv)
    if not results_path.is_file():
        raise FileNotFoundError(f"Evil-twin results CSV not found: {results_path}")

    required_fields = {
        "seed",
        "twin",
        "n",
        "p",
        "bias_correction",
        "status",
        *(field for field, _ in PID_ATOMS),
    }
    with results_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                f"Results CSV is missing required columns: {sorted(missing_fields)}"
            )
        rows = list(reader)

    successful_rows = [row for row in rows if row["status"] == "ok"]
    if not successful_rows:
        raise ValueError(f"Results CSV contains no successful rows: {results_path}")

    metadata = {}
    for field in ("n", "p", "bias_correction"):
        values = {row[field] for row in successful_rows}
        if len(values) != 1:
            raise ValueError(f"Results CSV contains inconsistent {field}: {sorted(values)}")
        metadata[field] = next(iter(values))
    try:
        if int(metadata["n"]) <= 0 or int(metadata["p"]) <= 0:
            raise ValueError
    except ValueError as error:
        raise ValueError("Results CSV requires positive integer n and p values.") from error
    bias_value = metadata["bias_correction"].strip().lower()
    if bias_value not in {"true", "false"}:
        raise ValueError("Results CSV bias_correction must be True or False.")

    means = {}
    for twin in TWINS:
        twin_rows = [row for row in successful_rows if row["twin"] == twin]
        if not twin_rows:
            raise ValueError(f"Results CSV contains no successful {twin.title()} rows.")
        try:
            means[twin] = [
                fmean(float(row[field]) for row in twin_rows)
                for field, _ in PID_ATOMS
            ]
        except ValueError as error:
            raise ValueError(f"Results CSV contains a non-numeric PID atom for {twin}.") from error

    atom_positions = list(range(len(PID_ATOMS)))
    bar_width = 0.38
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for twin, offset, color in (
        ("sonic", -bar_width / 2, "#2563eb"),
        ("shadow", bar_width / 2, "#f97316"),
    ):
        positions = [position + offset for position in atom_positions]
        axis.bar(positions, means[twin], bar_width, label=twin.title(), color=color)

    correction = (
        "bias corrected"
        if bias_value == "true"
        else "uncorrected"
    )
    axis.set(
        title=f"Sonic and Shadow PID atoms (n={metadata['n']}, p={metadata['p']}, {correction})",
        ylabel="Information (bits)",
        xticks=atom_positions,
        xticklabels=[label for _, label in PID_ATOMS],
    )
    axis.axhline(0.0, color="#374151", linewidth=0.8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()

    image_path = (
        Path(output_path)
        if output_path is not None
        else results_path.with_name(f"{results_path.stem}_pid_atoms.png")
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return image_path


def main(config: dict | None = None) -> Path:
    """Plot the configured evil-twin results CSV.

    Inputs:
        config: optional dict with ``results_csv`` and ``output_path``. When
            omitted, the editable module-level ``CONFIG`` is used.

    Outputs:
        Path, the saved grouped-bar-chart image.
    """
    settings = CONFIG if config is None else config
    image_path = plot_pid_atoms(settings["results_csv"], settings["output_path"])
    print(f"Saved evil-twin PID atom plot to: {image_path}")
    return image_path


if __name__ == "__main__":
    main()
