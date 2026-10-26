"""Save the cross-decomposition comparison tables and two requested figures."""

from hashlib import blake2b
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
import torch
import xarray as xr
import yaml

# Configure Bonner before importing the authors' plotting implementation.
from pipeline.testing.cross_decomposition import cross_decomposition_inputs as _inputs

from lib.spectra import plot_spectrum

REPO_ROOT = Path(__file__).resolve().parents[3]
OBSERVED_COLOR = "#2a78d6"
OPTIMUM_COLOR = "#eb6834"
REFERENCE_COLOR = "#52514e"
NULL_COLOR = "#8c8b87"
FIGURE_RC = {
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
}


def summarize_criteria(
    bin_table: pd.DataFrame,
    rank_table: pd.DataFrame,
    true_rank: int | None,
    k_reconstruction: int | None,
) -> pd.DataFrame:
    """Create the data-source-appropriate rank result summary.

    Inputs:
        bin_table: pd.DataFrame with bin intervals and significance flags.
        rank_table: pd.DataFrame with component ranks and significance flags.
        true_rank: int known synthetic signal rank, or None for OTC data.
        k_reconstruction: int minimizing held-out reconstruction error, or None
            when the separate reconstruction criterion was not run.

    Output:
        summary: pd.DataFrame with ``quantity``, human-readable ``result``,
            and numeric ``value`` columns. Ground truth is reported only for
            synthetic data.
    """

    significant_bins = bin_table.loc[bin_table["significant"]]
    significant_ranks = rank_table.loc[
        rank_table["significant"], "component"
    ].astype(int)
    if significant_bins.empty:
        highest_interval, highest_bin_rank = "none", np.nan
    else:
        highest = significant_bins.iloc[-1]
        highest_interval = f"{int(highest.first_rank)}-{int(highest.last_rank)}"
        highest_bin_rank = int(highest.last_rank)
    intervals = ", ".join(
        f"{int(row.first_rank)}-{int(row.last_rank)}"
        for row in significant_bins.itertuples()
    ) or "none"
    ranks = ", ".join(map(str, significant_ranks)) or "none"
    highest_rank = int(significant_ranks.max()) if len(significant_ranks) else np.nan
    rows = []
    if true_rank is not None:
        rows.append(("Ground-truth signal rank", str(true_rank), true_rank))
    rows.extend(
        [
            ("Significant cross-decomposition bins", intervals, len(significant_bins)),
            ("Highest significant rank interval", highest_interval, highest_bin_rank),
            ("Significant individual ranks", ranks, len(significant_ranks)),
            (
                "Highest significant individual rank",
                str(highest_rank) if np.isfinite(highest_rank) else "none",
                highest_rank,
            ),
        ]
    )
    if k_reconstruction is not None:
        rows.append(
            ("Reconstruction-optimal k", str(k_reconstruction), k_reconstruction)
        )
    return pd.DataFrame(rows, columns=["quantity", "result", "value"])


def plot_permutation_significance(
    results: dict[str, Any],
    output_stem: Path,
) -> None:
    """Save the requested binned and unbinned permutation figure.

    Inputs:
        results: dict[str, Any] returned by the experiment runner.
        output_stem: Path without suffix for the PDF and PNG outputs.

    Output:
        None. Writes one PDF and one PNG.
    """

    true_rank = results["true_rank"]
    null_quantile = results["config"]["permutation"]["null_quantile"]
    panels = (
        (
            results["binned_spectra"],
            results["bin_table"]["bin_center"].to_numpy(),
            results["bin_table"]["null_threshold"].to_numpy(),
            "(a) Logarithmically binned ranks",
        ),
        (
            results["unbinned_spectra"].rename({"component": "rank"}),
            results["rank_table"]["component"].to_numpy(),
            results["rank_table"]["null_threshold"].to_numpy(),
            "(b) Individual ranks",
        ),
    )
    with matplotlib.rc_context(FIGURE_RC):
        figure = Figure(figsize=(8.2, 3.7), layout="constrained")
        axes = figure.subplots(1, 2, sharey=True)
        for axis, (spectrum, ranks, threshold, title) in zip(axes, panels):
            axis.axhline(0, color=REFERENCE_COLOR, lw=0.6)
            if true_rank is not None:
                axis.axvline(true_rank, color=REFERENCE_COLOR, ls="--", lw=1)
            axis.plot(ranks, threshold, color=NULL_COLOR, lw=1.4)
            plot_spectrum(
                axis,
                spectrum=spectrum,
                hide_insignificant=True,
                null_quantile=null_quantile,
                kwargs_significant={
                    "ls": "None",
                    "marker": "s",
                    "ms": 6,
                    "color": OBSERVED_COLOR,
                    "mew": 0,
                },
                kwargs_insignificant={
                    "ls": "None",
                    "marker": "s",
                    "ms": 6,
                    "color": OBSERVED_COLOR,
                    "mfc": "white",
                    "mew": 1,
                },
            )
            axis.set_xscale("log")
            axis.set_yscale("symlog", linthresh=1e-4, linscale=0.4)
            axis.set_xlim(0.8, 1.4 * results["rank_table"]["component"].max())
            axis.set_xlabel("cross-decomposition rank")
            axis.set_title(title, loc="left")
        axes[0].set_ylabel("held-out covariance / sqrt(d_x d_y)")
        legend_handles = [
            Line2D([], [], ls="None", marker="s", color=OBSERVED_COLOR,
                   label="significant"),
            Line2D([], [], ls="None", marker="s", color=OBSERVED_COLOR,
                   mfc="white", label="not significant"),
            Line2D([], [], color=NULL_COLOR,
                   label=f"null {100 * null_quantile:g}th percentile"),
        ]
        if true_rank is not None:
            legend_handles.append(
                Line2D([], [], color=REFERENCE_COLOR, ls="--",
                       label=f"true rank = {true_rank}")
            )
        figure.legend(
            handles=legend_handles,
            loc="outside lower center",
            ncols=len(legend_handles),
            frameon=False,
        )
        figure.savefig(output_stem.with_suffix(".pdf"))
        figure.savefig(output_stem.with_suffix(".png"), dpi=200)


def plot_reconstruction(results: dict[str, Any], output_stem: Path) -> None:
    """Save held-out reconstruction error as a function of retained rank.

    Inputs:
        results: dict[str, Any] returned by the experiment runner.
        output_stem: Path without suffix for the PDF and PNG outputs.

    Output:
        None. Writes one PDF and one PNG.
    """

    reconstruction = results["reconstruction"]
    candidate_k = reconstruction["candidate_k"]
    errors = reconstruction["mean_errors"]
    standard_errors = reconstruction["standard_errors"]
    k_star = reconstruction["k_reconstruction"]
    true_rank = results["true_rank"]
    with matplotlib.rc_context(FIGURE_RC):
        figure = Figure(figsize=(6.2, 3.5), layout="constrained")
        axis = figure.subplots()
        axis.fill_between(
            candidate_k,
            errors - standard_errors,
            errors + standard_errors,
            color=OBSERVED_COLOR,
            alpha=0.18,
        )
        axis.plot(candidate_k, errors, color=OBSERVED_COLOR, marker="o", ms=3.5)
        if true_rank is not None:
            axis.axvline(true_rank, color=REFERENCE_COLOR, ls="--", lw=1,
                        label=f"true rank = {true_rank}")
        axis.plot(k_star, errors[k_star], ls="None", marker="D", ms=8,
                  color=OPTIMUM_COLOR, label=f"argmin k = {k_star}")
        axis.set_xlabel("retained cross-decomposition components k")
        axis.set_ylabel("normalized held-out MSE, E(k)")
        axis.set_title(
            "Authors' transform/inverse-transform scored against an independent repeat"
        )
        axis.legend(frameon=False)
        figure.savefig(output_stem.with_suffix(".pdf"))
        figure.savefig(output_stem.with_suffix(".png"), dpi=200)


def save_outputs(results: dict[str, Any], smoke: bool) -> Path:
    """Save tables, resolved settings, and available figures.

    Inputs:
        results: dict[str, Any] returned by the experiment runner.
        smoke: bool selecting the smoke subdirectory.

    Output:
        output_dir: Path to a newly created content-addressed result directory.
            Raises FileExistsError rather than silently overwriting a run.
    """

    settings = {
        "config": results["config"],
        "n_permutations_used": results["n_permutations"],
        "data_metadata": results["data_metadata"],
    }
    digest = blake2b(
        yaml.safe_dump(settings, sort_keys=True).encode(),
        digest_size=6,
    ).hexdigest()
    root = REPO_ROOT / results["config"]["output_dir"]
    if smoke:
        root = root / "smoke"
    metadata = results["data_metadata"]
    if metadata["source"] == "synthetic":
        generator = metadata["generator_config"]
        run_name = (
            f"synthetic_images={metadata['n_images']}_rank={generator['true_rank']}"
            f"_seed={generator['seed']}_run={digest}"
        )
    else:
        run_name = (
            f"otc_subject={metadata['subject']}_image_set={metadata['image_set']}"
            f"_images={metadata['n_images']}_voxels={metadata['n_voxels']}"
            f"_run={digest}"
        )
    output_dir = root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    results["summary"].to_csv(output_dir / "summary.csv", index=False)
    results["bin_table"].to_csv(output_dir / "binned_significance.csv", index=False)
    results["rank_table"].to_csv(output_dir / "rank_significance.csv", index=False)
    reconstruction = results["reconstruction"]
    if reconstruction is not None:
        reconstruction_table = pd.DataFrame(
            {
                "k": reconstruction["candidate_k"],
                "mean_normalized_error": reconstruction["mean_errors"],
                "standard_error": reconstruction["standard_errors"],
            }
        )
        for fold, errors in enumerate(reconstruction["fold_errors"]):
            reconstruction_table[f"fold_{fold}_normalized_error"] = errors
        reconstruction_table.to_csv(
            output_dir / "reconstruction_curve.csv",
            index=False,
        )

    run_record = settings | {
        "ground_truth": results["ground_truth"],
        # (images,) -> list[images]
        "image_ids": results["image_ids"].tolist(),
        # (images,) -> list[images]
        "fold_of_each_image": results["folds"].tolist(),
        "versions": {
            "numpy": str(np.__version__),
            "scipy": str(scipy.__version__),
            "torch": str(torch.__version__),
            "xarray": str(xr.__version__),
        },
    }
    with (output_dir / "run_config.yaml").open("w") as handle:
        yaml.safe_dump(run_record, handle, sort_keys=False)

    plot_permutation_significance(
        results,
        output_dir / "figure1_permutation_significance",
    )
    if reconstruction is not None:
        plot_reconstruction(results, output_dir / "figure2_reconstruction")
    return output_dir
