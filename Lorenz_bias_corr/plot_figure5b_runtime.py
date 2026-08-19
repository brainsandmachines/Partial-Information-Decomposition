#!/usr/bin/env python3
"""Plot Figure 5B method runtimes as a function of simulation dimension."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "figure5b-eigenvalues_pid_full.csv"
OUTPUT_STEM = SCRIPT_DIR / "figure5b - eigenvalue_pid_runtime_full"
LINEAR_OUTPUT_STEM = SCRIPT_DIR / "figure5b - eigenvalue_pid_runtime_linear_full"

METHODS = (
    ("Resampling", "resampling_seconds", "#edb120", "o"),
    ("Shuffle", "shuffle_seconds", "#d95319", "s"),
    ("Merged", "merged_seconds", "#7e2f8e", "D"),
    ("Venkatesh", "venkatesh_seconds", "#77ac30", "^"),
    ("Eigen-PID + Lorenz", "eigenvalue_pid_seconds", "#0072bd", "P"),
)


if __name__ == "__main__":
    if not INPUT_PATH.is_file():
        raise SystemExit(f"Timing CSV was not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise SystemExit(f"Timing CSV contains no data rows: {INPUT_PATH}")

    dimensions = [int(row["dimension"]) for row in rows]
    first_row = rows[0]
    plot_specs = (
        (OUTPUT_STEM, "Runtime [minutes, log scale]", "log"),
        (LINEAR_OUTPUT_STEM, "Runtime [minutes]", "linear"),
    )
    for output_stem, y_label, y_scale in plot_specs:
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        for method_name, column_name, color, marker in METHODS:
            runtimes = [float(row[column_name]) / 60.0 for row in rows]
            axis.plot(
                dimensions,
                runtimes,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.7,
                label=method_name,
            )

        axis.set_title(
            "Figure 5B runtime by dimension\n"
            f"N={first_row['ntrials']}, repetitions={first_row['n_repetitions']}, "
            f"bias iterations={first_row['bias_iterations']}",
            fontweight="bold",
        )
        axis.set_xlabel("Dimension")
        axis.set_ylabel(y_label)
        axis.set_xticks(dimensions)
        axis.set_yscale(y_scale)
        axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
        axis.legend(frameon=False, ncol=2)
        figure.tight_layout()

        svg_path = output_stem.with_suffix(".svg")
        png_path = output_stem.with_suffix(".png")
        figure.savefig(svg_path, format="svg", bbox_inches="tight")
        figure.savefig(png_path, format="png", dpi=220, bbox_inches="tight")
        plt.close(figure)

        print(f"Runtime SVG: {svg_path}")
        print(f"Runtime PNG: {png_path}")
