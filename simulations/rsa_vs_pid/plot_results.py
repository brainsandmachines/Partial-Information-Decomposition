"""Plot RSA scores, BROJA PID atoms, and their paired differences from saved CSVs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "trials_csv": "simulation_results/rsa_vs_pid/rsa_vs_pid_trials.csv",
    "summary_csv": "simulation_results/rsa_vs_pid/rsa_vs_pid_summary.csv",
    "output_dir": "simulation_results/rsa_vs_pid",
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
REFERENCE_FILL = "#f0efec"
COLOR_X = "#2a78d6"  # categorical slot 1: source X
COLOR_Y = "#eb6834"  # categorical slot 2: source Y
N_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")  # ordinal blue ramp, light -> dark

PID_ROWS = (
    ("red", "Redundancy", INK_SECONDARY),
    ("unq_x", "Unique X", COLOR_X),
    ("unq_y", "Unique Y", COLOR_Y),
    ("syn", "Synergy", INK_SECONDARY),
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "font.size": 9,
    }
)


def _load_results(trials_csv: str | Path, summary_csv: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load trial and summary CSVs written by ``run_rsa_vs_pid.py``.

    Inputs:
        trials_csv: str or Path, one row per trial.
        summary_csv: str or Path, long-format summary per (dimension, n_samples, metric).

    Outputs:
        tuple (trials, summary) of DataFrames; ``summary`` is indexed by
        (dimension, n_samples, metric).
    """
    trials = pd.read_csv(trials_csv)
    summary = pd.read_csv(summary_csv).set_index(["dimension", "n_samples", "metric"]).sort_index()
    return trials, summary


def _condition_grid(trials: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Return the sorted dimensions and sample sizes present in the trials.

    Inputs:
        trials: DataFrame with integer ``dimension`` and ``n_samples`` columns.

    Outputs:
        tuple (dimensions, sample_sizes) of sorted integer lists.
    """
    return sorted(trials["dimension"].unique().tolist()), sorted(trials["n_samples"].unique().tolist())


def _summary_series(summary: pd.DataFrame, dimension: int, sample_sizes: list[int], metric: str) -> dict[str, np.ndarray]:
    """Read the median and trial quartiles of one metric across sample sizes.

    Inputs:
        summary: DataFrame indexed by (dimension, n_samples, metric).
        dimension: int, the dimension to read.
        sample_sizes: list[int], sample sizes in plotting order.
        metric: str, metric name in the summary.

    Outputs:
        dict with float arrays ``median``, ``q25``, ``q75``, and ``sd``, each
        shaped (len(sample_sizes),).
    """
    rows = summary.loc[[(dimension, n, metric) for n in sample_sizes]]
    return {
        "median": rows["median"].to_numpy(),
        "q25": rows["q25"].to_numpy(),
        "q75": rows["q75"].to_numpy(),
        "sd": rows["sd"].to_numpy(),
    }  # len(sample_sizes) summary rows -> four (len(sample_sizes),) arrays


def _jitter(rng: np.random.Generator, center: float, count: int, width: float = 0.07) -> np.ndarray:
    """Return horizontal jitter positions around one categorical x position.

    Inputs:
        rng: np.random.Generator for reproducible jitter.
        center: float, categorical x position.
        count: int, number of points.
        width: float, half-width of the uniform jitter.

    Outputs:
        np.ndarray shaped (count,) of x positions.
    """
    return center + rng.uniform(-width, width, size=count)


def _finish_n_axis(ax: plt.Axes, sample_sizes: list[int]) -> None:
    """Label a categorical sample-size x-axis.

    Inputs:
        ax: matplotlib Axes to update.
        sample_sizes: list[int], sample sizes at positions 0, 1, ...

    Outputs:
        None. Sets ticks, tick labels, limits, and the x label.
    """
    ax.set_xticks(range(len(sample_sizes)))
    ax.set_xticklabels([str(n) for n in sample_sizes])
    ax.set_xlim(-0.5, len(sample_sizes) - 0.5)
    ax.set_xlabel("Sample size n")


def plot_rsa_comparison(trials_csv: str | Path, summary_csv: str | Path, output_path: str | Path) -> Path:
    """Plot RSA(T, X), RSA(T, Y), and their difference across sample sizes.

    Inputs:
        trials_csv: str or Path, trial-level CSV.
        summary_csv: str or Path, summary CSV with trial quartiles.
        output_path: str or Path, PNG output path.

    Outputs:
        Path to the saved 2 x n_dimensions figure. Top row: per-trial RSA
        scores with medians and 25th-75th percentiles across trials. Bottom
        row: per-trial delta RSA with its median and 25th-75th percentiles,
        against a +/-1 SD band of RSA(T, X).
    """
    trials, summary = _load_results(trials_csv, summary_csv)
    dimensions, sample_sizes = _condition_grid(trials)
    rng = np.random.default_rng(0)
    positions = np.arange(len(sample_sizes))  # n_sample_sizes -> (n_sample_sizes,) categorical x positions
    fig, axes = plt.subplots(2, len(dimensions), figsize=(3.6 * len(dimensions), 6.2), squeeze=False)

    for column, dimension in enumerate(dimensions):
        ax_scores, ax_delta = axes[0, column], axes[1, column]
        condition = trials[trials["dimension"] == dimension]
        for metric, label, color, offset in (
            ("rsa_x", "RSA(T, X)", COLOR_X, -0.12),
            ("rsa_y", "RSA(T, Y)", COLOR_Y, 0.12),
        ):
            for index, n_samples in enumerate(sample_sizes):
                values = condition.loc[condition["n_samples"] == n_samples, metric].to_numpy()
                ax_scores.scatter(
                    _jitter(rng, index + offset, len(values), 0.05), values,
                    s=9, color=color, alpha=0.25, linewidths=0,
                )
            series = _summary_series(summary, dimension, sample_sizes, metric)
            ax_scores.errorbar(
                positions + offset, series["median"],
                yerr=[series["median"] - series["q25"], series["q75"] - series["median"]],
                color=color, marker="o", markersize=5, linewidth=1.5, capsize=2.5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=label,
            )
        ax_scores.set_title(f"d = {dimension}")
        ax_scores.set_xticks(positions)
        ax_scores.set_xticklabels([])
        ax_scores.set_xlim(-0.5, len(sample_sizes) - 0.5)

        reference = _summary_series(summary, dimension, sample_sizes, "rsa_x")
        for index, reference_sd in enumerate(reference["sd"]):
            ax_delta.fill_between(
                [index - 0.3, index + 0.3], -reference_sd, reference_sd, color=REFERENCE_FILL, zorder=0,
                label="±1 SD of RSA(T, X) across trials" if index == 0 else None,
            )
        ax_delta.axhline(0, color=BASELINE, linewidth=1)
        for index, n_samples in enumerate(sample_sizes):
            values = condition.loc[condition["n_samples"] == n_samples, "delta_rsa"].to_numpy()
            ax_delta.scatter(_jitter(rng, index, len(values)), values, s=9, color=COLOR_X, alpha=0.3, linewidths=0)
        delta = _summary_series(summary, dimension, sample_sizes, "delta_rsa")
        ax_delta.errorbar(
            positions, delta["median"], yerr=[delta["median"] - delta["q25"], delta["q75"] - delta["median"]],
            color=INK_PRIMARY, marker="o", markersize=5, linewidth=1.5, capsize=2.5,
            markeredgecolor=SURFACE, markeredgewidth=1, label="Median, 25th–75th percentile",
        )
        _finish_n_axis(ax_delta, sample_sizes)

    axes[0, 0].set_ylabel("RSA (Spearman ρ between RDMs)")
    axes[1, 0].set_ylabel("ΔRSA = RSA(T, X) − RSA(T, Y)")
    score_handles, score_labels = axes[0, 0].get_legend_handles_labels()
    delta_handles, delta_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        score_handles + delta_handles, score_labels + delta_labels,
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle("Pearson-RDM RSA does not separate X from Y", color=INK_PRIMARY, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    saved_path = Path(output_path)
    fig.savefig(saved_path, dpi=200)
    plt.close(fig)
    return saved_path


def plot_pid_decomposition(trials_csv: str | Path, summary_csv: str | Path, output_path: str | Path) -> Path:
    """Plot the four BROJA PID atoms across trials and sample sizes.

    Inputs:
        trials_csv: str or Path, trial-level CSV.
        summary_csv: str or Path, summary CSV with medians of the raw atoms.
        output_path: str or Path, PNG output path.

    Outputs:
        Path to the saved 4 x n_dimensions figure. Box plots show the
        bias-corrected atoms across trials (median and 25th-75th percentiles);
        the dashed line with open markers shows the median raw (uncorrected)
        atom. Values are in bits.
    """
    trials, summary = _load_results(trials_csv, summary_csv)
    dimensions, sample_sizes = _condition_grid(trials)
    positions = np.arange(len(sample_sizes))  # n_sample_sizes -> (n_sample_sizes,) categorical x positions
    fig, axes = plt.subplots(
        len(PID_ROWS), len(dimensions), figsize=(3.6 * len(dimensions), 2.3 * len(PID_ROWS)), squeeze=False,
    )

    for row, (metric, label, color) in enumerate(PID_ROWS):
        for column, dimension in enumerate(dimensions):
            ax = axes[row, column]
            condition = trials[trials["dimension"] == dimension]
            values = [
                condition.loc[condition["n_samples"] == n_samples, metric].to_numpy()
                for n_samples in sample_sizes
            ]  # n_sample_sizes arrays shaped (n_trials,)
            ax.axhline(0, color=BASELINE, linewidth=1)
            ax.boxplot(
                values, positions=positions, widths=0.45, patch_artist=True, showfliers=True,
                boxprops={"facecolor": SURFACE, "edgecolor": color, "linewidth": 1.2},
                whiskerprops={"color": color, "linewidth": 1.0},
                capprops={"color": color, "linewidth": 1.0},
                medianprops={"color": color, "linewidth": 1.8},
                flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": color,
                            "markeredgecolor": color, "alpha": 0.5},
            )
            raw_median = _summary_series(summary, dimension, sample_sizes, f"{metric}_raw")["median"]
            ax.plot(
                positions, raw_median, color=INK_PRIMARY, linestyle="--", linewidth=1.0,
                marker="o", markersize=4, markerfacecolor=SURFACE, markeredgecolor=INK_PRIMARY,
            )
            if row == 0:
                ax.set_title(f"d = {dimension}")
            if column == 0:
                ax.set_ylabel(f"{label} (bits)")
            if row == len(PID_ROWS) - 1:
                _finish_n_axis(ax, sample_sizes)
            else:
                ax.set_xticks(positions)
                ax.set_xticklabels([])
                ax.set_xlim(-0.5, len(sample_sizes) - 0.5)

    legend_handles = [
        plt.Line2D([], [], color=INK_SECONDARY, linewidth=1.2, label="Bias-corrected, across trials (box)"),
        plt.Line2D([], [], color=INK_PRIMARY, linestyle="--", linewidth=1.0, marker="o", markersize=4,
                   markerfacecolor=SURFACE, label="Raw (uncorrected) median"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Gaussian BROJA PID of T from sources X and Y", color=INK_PRIMARY, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    saved_path = Path(output_path)
    fig.savefig(saved_path, dpi=200)
    plt.close(fig)
    return saved_path


def plot_rsa_vs_unique(trials_csv: str | Path, summary_csv: str | Path, output_path: str | Path) -> Path:
    """Plot per-trial (delta RSA, delta unique information) pairs.

    Inputs:
        trials_csv: str or Path, trial-level CSV.
        summary_csv: str or Path, summary CSV with trial quartiles.
        output_path: str or Path, PNG output path.

    Outputs:
        Path to the saved 1 x n_dimensions figure. Each point is one trial:
        x = RSA(T, X) - RSA(T, Y) (Spearman units), y = bias-corrected
        U_X - U_Y (bits). Crosses show the per-n median with 25th-75th
        percentiles on both axes. The two quantities keep separate axes and units.
    """
    trials, summary = _load_results(trials_csv, summary_csv)
    dimensions, sample_sizes = _condition_grid(trials)
    if len(sample_sizes) > len(N_RAMP):
        raise ValueError(f"At most {len(N_RAMP)} sample sizes can be colored; received {len(sample_sizes)}.")
    ramp = N_RAMP[len(N_RAMP) - len(sample_sizes):]
    fig, axes = plt.subplots(1, len(dimensions), figsize=(3.8 * len(dimensions), 3.9), squeeze=False)

    for column, dimension in enumerate(dimensions):
        ax = axes[0, column]
        condition = trials[trials["dimension"] == dimension]
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.axvline(0, color=BASELINE, linewidth=1)
        for index, n_samples in enumerate(sample_sizes):
            subset = condition[condition["n_samples"] == n_samples]
            ax.scatter(
                subset["delta_rsa"], subset["delta_ui"], s=16, color=ramp[index], alpha=0.75,
                edgecolors=SURFACE, linewidths=0.6, label=f"n = {n_samples}",
            )
        for index, n_samples in enumerate(sample_sizes):
            delta_rsa = _summary_series(summary, dimension, [n_samples], "delta_rsa")
            delta_ui = _summary_series(summary, dimension, [n_samples], "delta_ui")
            ax.errorbar(
                delta_rsa["median"], delta_ui["median"],
                xerr=[delta_rsa["median"] - delta_rsa["q25"], delta_rsa["q75"] - delta_rsa["median"]],
                yerr=[delta_ui["median"] - delta_ui["q25"], delta_ui["q75"] - delta_ui["median"]],
                color=INK_PRIMARY, linewidth=1.0, capsize=2, marker="o", markersize=5,
                markerfacecolor=ramp[index], markeredgecolor=INK_PRIMARY, markeredgewidth=0.8,
            )
        x_limit = 1.1 * float(np.abs(condition["delta_rsa"]).max())
        ax.set_xlim(-x_limit, x_limit)
        y_low, y_high = ax.get_ylim()
        ax.set_ylim(min(y_low, 0.0), y_high)
        ax.set_title(f"d = {dimension}")
        ax.set_xlabel("ΔRSA = RSA(T, X) − RSA(T, Y)")
    axes[0, 0].set_ylabel("ΔUI = U_X − U_Y, bias-corrected (bits)")
    trial_handles, trial_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        trial_handles, [f"Trials, {label}" for label in trial_labels],
        loc="lower center", ncol=len(sample_sizes), bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Equal RSA, unequal unique information (crosses: median, 25th–75th percentile)",
        color=INK_PRIMARY, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    saved_path = Path(output_path)
    fig.savefig(saved_path, dpi=200)
    plt.close(fig)
    return saved_path


def plot_all(trials_csv: str | Path, summary_csv: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Create the three experiment figures.

    Inputs:
        trials_csv: str or Path, trial-level CSV.
        summary_csv: str or Path, summary CSV.
        output_dir: str or Path, directory for the PNG files.

    Outputs:
        dict mapping figure keys to saved Paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "fig1_rsa_comparison": plot_rsa_comparison(trials_csv, summary_csv, output_dir / "fig1_rsa_comparison.png"),
        "fig2_pid_decomposition": plot_pid_decomposition(
            trials_csv, summary_csv, output_dir / "fig2_pid_decomposition.png"
        ),
        "fig3_rsa_vs_unique": plot_rsa_vs_unique(trials_csv, summary_csv, output_dir / "fig3_rsa_vs_unique.png"),
    }


def main(config: dict | None = None) -> dict[str, Path]:
    """Plot the configured results; relative paths resolve from the repository root.

    Inputs:
        config: dict or None with ``trials_csv``, ``summary_csv``, and
            ``output_dir``; defaults to ``CONFIG``.

    Outputs:
        dict mapping figure keys to saved Paths.
    """
    config = dict(CONFIG if config is None else config)
    project_root = Path(__file__).resolve().parents[2]
    paths = {
        key: Path(value) if Path(value).is_absolute() else project_root / value
        for key, value in config.items()
    }
    return plot_all(paths["trials_csv"], paths["summary_csv"], paths["output_dir"])


if __name__ == "__main__":
    for figure_path in main().values():
        print(f"Saved figure: {figure_path}")
