"""Cross-decomposition spectrum with the authors' implementation (Gauthaman et al., 2025).

Repetitions 0 and 1 of each image form two datasets; the authors' code fits
cross-covariance SVD directions on training folds, measures each direction's
held-out covariance, and compares it with a permutation null. This module only
adapts inputs and outputs: every decomposition, cross-validation, permutation,
and binning step is the authors' (``lib`` from external/scale-free-visual-cortex).
It needs the ``cross_decomposition`` conda environment (bonner + lib).
"""

import inspect
import os
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

# Bonner reads these settings during import. Use harmless cache locations and
# never reuse a fit belonging to another fold, seed, subject, or simulation.
for variable, folder in (
    ("BONNER_CACHING_HOME", "bonner-caching"),
    ("BONNER_DATASETS_HOME", "bonner-datasets"),
    ("BONNER_BRAINIO_HOME", "bonner-brainio"),
    ("BONNER_MODELS_HOME", "bonner-models"),
):
    os.environ.setdefault(variable, str(Path.home() / ".cache" / folder))
os.environ["BONNER_CACHING_MODE"] = "ignore"


def skip_font_installation() -> None:
    """Disable Bonner's unrelated import-time font installation.

    Inputs:
        None.

    Output:
        None. No fonts or font-configuration files are written.
    """


# Importing the authors' package otherwise downloads fonts and rewrites fonts.conf.
fonts_stub = types.ModuleType("bonner.plotting._fonts")
fonts_stub.install_newcomputermodern = skip_font_installation
sys.modules.setdefault("bonner.plotting._fonts", fonts_stub)
from bonner.caching import cache
from lib.datasets import split_by_repetition
from lib.spectra import compute_spectra_with_n_fold_cross_validation
from lib.utilities import assign_bins, bin_data, extract_geometrically_spaced_bins

from pipeline.rank_selection.data import RankData
from pipeline.rank_selection.result import RankResult


def assert_caching_disabled() -> None:
    """Require the authors' cache wrapper to recompute every result.

    Inputs:
        None.

    Output:
        None. Raises RuntimeError if Bonner was imported before caching was
            configured as ``ignore``.
    """

    mode = inspect.signature(cache).parameters["mode"].default
    if mode != "ignore":
        raise RuntimeError(
            f"bonner caching mode is {mode!r}; import this module before bonner."
        )


def build_repetition_dataarrays(
    responses: np.ndarray,
    image_ids: np.ndarray,
    dataset_name: str,
    n_repetitions: int = 2,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Adapt repeated responses to the authors' two-repetition interface.

    Inputs:
        responses: finite np.ndarray shaped (images, voxels, repeats).
        image_ids: unique np.ndarray shaped (images,) identifying stimuli.
        dataset_name: str stable name used by the authors' cache labels.
        n_repetitions: int number of leading repetitions exposed to the
            authors' ``split_by_repetition``; it must be exactly two.

    Output:
        x, y: the authors' repetition-0 and repetition-1 ``xr.DataArray``
            objects, each shaped (presentation, neuroid). No averaging,
            scaling, centering, fitting, or decomposition is performed here.
    """

    response_array = np.asarray(responses)
    stimulus_ids = np.asarray(image_ids)
    if response_array.ndim != 3 or not np.isfinite(response_array).all():
        raise ValueError("responses must be finite with shape (images, voxels, repeats).")
    if stimulus_ids.shape != (response_array.shape[0],):
        raise ValueError("image_ids must have one entry per response image.")
    if np.unique(stimulus_ids).size != stimulus_ids.size:
        raise ValueError("image_ids must be unique so images cannot cross folds.")
    if n_repetitions != 2:
        raise ValueError("The authors' within-subject calculation requires two repetitions.")
    if response_array.shape[2] < n_repetitions:
        raise ValueError(
            f"responses contain {response_array.shape[2]} repeats, "
            f"but {n_repetitions} are required."
        )

    n_images, n_voxels, _ = response_array.shape
    # (images, voxels, repeats) -> (images, repetitions, voxels)
    selected = response_array[:, :, :n_repetitions].transpose(0, 2, 1)
    # (images, repetitions, voxels) -> (presentations, neuroids)
    values = selected.reshape(n_images * n_repetitions, n_voxels)
    repeated = xr.DataArray(
        values,
        dims=("presentation", "neuroid"),
        coords={
            "presentation": np.arange(n_images * n_repetitions),
            "neuroid": np.arange(n_voxels),
            # (images,) -> (presentations,)
            "stimulus": ("presentation", np.repeat(stimulus_ids, n_repetitions)),
            # (repetitions,) -> (presentations,)
            "repetition": (
                "presentation",
                np.tile(np.arange(n_repetitions), n_images),
            ),
        },
        name=dataset_name,
    ).set_xindex(["stimulus", "repetition"])
    repetitions = split_by_repetition(repeated, n_repetitions=n_repetitions)
    return repetitions[0], repetitions[1]



def significance_table(
    spectra: xr.Dataset,
    dimension: str,
    null_quantile: float,
) -> pd.DataFrame:
    """Apply the authors' plotting significance rule to a rank spectrum.

    Inputs:
        spectra: xr.Dataset containing observed ``covariance`` and permuted
            ``covariance (permuted)`` across folds.
        dimension: str spectrum dimension, either ``component`` or ``rank``.
        null_quantile: float upper null quantile, 0.999 for p < 0.001.

    Output:
        table: pd.DataFrame with rank coordinates, observed fold means, fold
            SDs, null thresholds, empirical p-values, and significance flags.
    """

    # (fold, dimension) -> (dimension,)
    observed = spectra["covariance"].mean("fold")
    # (fold, permutation, dimension) -> (permutation, dimension)
    null = spectra["covariance (permuted)"].mean("fold")
    # (permutation, dimension) -> (dimension,)
    threshold = null.quantile(null_quantile, dim="permutation")
    n_exceeding = (null >= observed).sum("permutation")
    return pd.DataFrame(
        {
            dimension: observed[dimension].to_numpy(),
            "observed_covariance": observed.to_numpy(),
            "fold_sd": spectra["covariance"].std("fold", ddof=1).to_numpy(),
            "null_threshold": threshold.to_numpy(),
            "significant": (observed > threshold).to_numpy(),
            "empirical_p": (
                (1 + n_exceeding) / (1 + null.sizes["permutation"])
            ).to_numpy(),
        }
    )


def calculate_authors_spectra(
    x: xr.DataArray,
    y: xr.DataArray,
    cross_decomposition_config: dict[str, Any],
    permutation_config: dict[str, Any],
    n_permutations: int,
) -> dict[str, Any]:
    """Run the authors' complete spectrum, permutation, and binning algorithm.

    Inputs:
        x: xr.DataArray shaped (presentation, neuroid), repetition 0.
        y: xr.DataArray shaped (presentation, neuroid), repetition 1.
        cross_decomposition_config: dict containing ``n_folds``, ``seed``, and
            ``randomized`` settings passed to the authors' implementation.
        permutation_config: dict containing the authors' binning, batching,
            and null-quantile settings.
        n_permutations: int number of permutations to request.

    Output:
        results: dict[str, Any] containing the authors' unbinned and binned
            spectra plus rank-significance tables. No decomposition,
            cross-validation, permutation, or binning algorithm is local.
    """

    n_folds = cross_decomposition_config["n_folds"]
    seed = cross_decomposition_config["seed"]
    torch.manual_seed(seed)
    unbinned_spectra = compute_spectra_with_n_fold_cross_validation(
        x_train=x,
        y_train=y,
        x_test=x,
        y_test=y,
        n_folds=n_folds,
        seed=seed,
        n_permutations=n_permutations,
        randomized=cross_decomposition_config["randomized"],
        batch_size=permutation_config["batch_size"],
    )
    bin_edges, bin_centers = extract_geometrically_spaced_bins(
        start=1,
        stop=permutation_config["bin_stop"],
        density=permutation_config["bin_density"],
    )
    # (fold, [permutation,] component) -> (fold, [permutation,] rank)
    binned_spectra = bin_data(
        unbinned_spectra,
        bin_edges={"component": bin_edges},
        bin_centers={"component": bin_centers},
        dim="rank",
    )
    rank_table = significance_table(
        unbinned_spectra,
        "component",
        permutation_config["null_quantile"],
    )
    bin_table = significance_table(
        binned_spectra,
        "rank",
        permutation_config["null_quantile"],
    ).rename(columns={"rank": "bin_center"})
    # (components,) -> (components,)
    component_ranks = 1 + np.arange(unbinned_spectra.sizes["component"])
    assigned_centers = assign_bins(
        component_ranks,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
    )
    intervals = (
        pd.DataFrame({"component": component_ranks, "bin_center": assigned_centers})
        .groupby("bin_center", sort=True)["component"]
        .agg(first_rank="min", last_rank="max", n_ranks="size")
        .reset_index()
    )
    bin_table = intervals.merge(bin_table, on="bin_center", validate="one_to_one")
    return {
        "unbinned_spectra": unbinned_spectra,
        "binned_spectra": binned_spectra,
        "rank_table": rank_table,
        "bin_table": bin_table,
    }



def leading_significant_count(significant: np.ndarray) -> int:
    """Count the leading consecutive significant components.

    Input: significant, np.ndarray of bool (n_components,), in rank order.
    Output: int, the number of components before the first non-significant one.
    """
    flags = np.asarray(significant, dtype=bool)
    nonsignificant = np.flatnonzero(~flags)
    return int(flags.shape[0] if nonsignificant.size == 0 else nonsignificant[0])


def run(
    data: RankData,
    n_permutations: int = 5000,
    folds: int = 8,
    seed: int = 0,
    randomized: bool = False,
    batch_size: int = 10,
    null_quantile: float = 0.999,
    bin_density: int = 3,
    bin_stop: int = 10000,
) -> RankResult:
    """Run the authors' cross-decomposition spectrum on a dataset.

    Inputs:
        data: RankData with at least two repeats; repeats 0 and 1 are used.
        n_permutations: int, permutations for the null (5,000 in Figure 3).
        folds: int, the authors' cross-validation folds.
        seed: int, the authors' seed for folds and permutations.
        randomized: bool, randomized SVD. The exact SVD (False) is the
            authors' Figure 3 setting but fails on CPU in their ``_svd_flip``;
            use True on CPU.
        batch_size, null_quantile, bin_density, bin_stop: the authors'
            permutation and binning settings (p < 0.001, 3 bins per decade).
    Output: RankResult with components 1..K, the held-out covariance as score,
        and the number of leading consecutive components above the null
        threshold as selected_rank. ``info`` also reports the total number of
        significant components and of significant bins.
    """
    assert_caching_disabled()
    # (images, voxels, repeats) -> two (presentations, neuroids) DataArrays
    x, y = build_repetition_dataarrays(data.responses, data.image_ids, data.name)
    spectra = calculate_authors_spectra(
        x,
        y,
        {"n_folds": folds, "seed": seed, "randomized": randomized},
        {
            "batch_size": batch_size,
            "null_quantile": null_quantile,
            "bin_density": bin_density,
            "bin_stop": bin_stop,
        },
        n_permutations,
    )
    table = spectra["rank_table"]
    significant = table["significant"].to_numpy(dtype=bool)  # (n_components,)
    return RankResult(
        method="cross_decomposition",
        dataset=data.name,
        seed=seed,
        selected_rank=leading_significant_count(significant),
        selection_rule=f"leading consecutive components above the {null_quantile} permutation quantile",
        score_name="held-out cross-covariance (mean over folds)",
        rank=table["component"].to_numpy(),
        score=table["observed_covariance"].to_numpy(),
        true_rank=data.true_rank,
        extra_curves={
            "fold_sd": table["fold_sd"].to_numpy(),
            "null_threshold": table["null_threshold"].to_numpy(),
            "empirical_p": table["empirical_p"].to_numpy(),
            "significant": significant,
        },
        info={
            "n_significant_total": int(significant.sum()),
            "n_significant_bins": int(spectra["bin_table"]["significant"].sum()),
            "n_permutations": n_permutations,
            "randomized_svd": randomized,
            "used_repetitions": "0,1",
        },
    )
