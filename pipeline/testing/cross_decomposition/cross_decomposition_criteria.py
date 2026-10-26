"""Calculate rank criteria with the authors' cross-decomposition code."""

from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

# Configure Bonner before importing the authors' spectrum implementation.
from pipeline.testing.cross_decomposition import cross_decomposition_inputs as _inputs

from lib.spectra import CrossDecomposition, compute_spectra_with_n_fold_cross_validation
from lib.spectra._computation import filter_datasets
from lib.utilities import assign_bins, bin_data, extract_geometrically_spaced_bins


def verify_ground_truth(synthetic_data: dict[str, Any], true_rank: int) -> dict[str, Any]:
    """Verify the exact synthetic signal rank before rank criteria are evaluated.

    Inputs:
        synthetic_data: dict[str, Any] returned by the shared generator.
        true_rank: int rank requested from that generator.

    Output:
        checks: dict[str, Any] containing the verified rank and maximum error
            between the empirical noiseless-signal covariance and its target.
            Raises AssertionError when construction is inconsistent.
    """

    covariance = synthetic_data["true_signal_covariance"]
    # (voxels, voxels) -> (voxels,)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    tolerance = covariance.shape[0] * np.finfo(float).eps * eigenvalues[0]
    rank = int(np.sum(eigenvalues > tolerance))
    # (images, voxels) -> (voxels, voxels)
    sample_covariance = np.cov(synthetic_data["signal"], rowvar=False)
    covariance_error = float(np.max(np.abs(sample_covariance - covariance)))
    assert rank == true_rank, f"signal covariance rank is {rank}, expected {true_rank}."
    np.testing.assert_allclose(
        eigenvalues[:true_rank],
        synthetic_data["true_signal_eigenvalues"],
        rtol=1e-10,
    )
    assert covariance_error < 1e-10
    return {
        "true_signal_covariance_rank": rank,
        "max_abs_sample_covariance_error": covariance_error,
    }


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


def calculate_reconstruction_criterion(
    x: xr.DataArray,
    y: xr.DataArray,
    folds: np.ndarray,
    randomized: bool,
    seed: int,
) -> dict[str, Any]:
    """Calculate the reconstruction-based retained-rank criterion.

    This criterion uses only the authors' fitted transforms, but is separate
    from ``calculate_authors_spectra`` because the authors do not provide a
    reconstruction-error rank-selection criterion.

    Inputs:
        x: xr.DataArray shaped (presentation, neuroid), repetition 0.
        y: xr.DataArray shaped (presentation, neuroid), repetition 1.
        folds: np.ndarray shaped (presentation,) from the authors' ``_get_folds``.
        randomized: bool passed unchanged to the authors' ``CrossDecomposition``.
        seed: int Torch seed for reproducible randomized decompositions.

    Output:
        reconstruction: dict[str, Any] containing k=0..K, fold-normalized
            symmetric reconstruction errors, means, standard errors, and the
            error-minimizing rank. All transforms use the authors' classes.
    """

    n_folds = int(np.unique(folds).size)
    max_k = min(
        x.sizes["neuroid"],
        y.sizes["neuroid"],
        min(int(np.sum(folds != fold)) - 1 for fold in range(n_folds)),
    )
    # Scalar dimensions -> (folds, max_k + 1)
    fold_errors = np.empty((n_folds, max_k + 1))
    torch.manual_seed(seed)
    for fold in range(n_folds):
        x_train, y_train, x_test, y_test = filter_datasets(
            x_train=x,
            y_train=y,
            x_test=x,
            y_test=y,
            filter_train=folds != fold,
            filter_test=folds == fold,
        )
        decomposition = CrossDecomposition(randomized=randomized)
        decomposition.fit(x_train, y_train)
        direction_errors = []
        for observed, target, direction in (
            (x_test, y_test, "left"),
            (y_test, x_test, "right"),
        ):
            # (test_images, voxels) -> (test_images, components)
            scores = decomposition.transform(observed, direction=direction)
            # (test_images, components) -> (test_images, voxels), k=0
            baseline = decomposition.inverse_transform(
                scores,
                direction=direction,
                components=0,
            ).to_numpy()
            denominator = float(np.mean((target.to_numpy() - baseline) ** 2))
            if denominator == 0:
                raise ValueError("held-out responses contain no reconstructable variance.")
            errors = []
            for k in range(max_k + 1):
                # (test_images, components) -> (test_images, voxels)
                reconstruction = decomposition.inverse_transform(
                    scores,
                    direction=direction,
                    components=k,
                ).to_numpy()
                errors.append(
                    float(np.mean((target.to_numpy() - reconstruction) ** 2) / denominator)
                )
            # list[max_k + 1] -> (max_k + 1,)
            direction_errors.append(np.asarray(errors))
        # (2, max_k + 1) -> (max_k + 1,)
        fold_errors[fold] = np.mean(direction_errors, axis=0)

    # (folds, max_k + 1) -> (max_k + 1,)
    mean_errors = fold_errors.mean(axis=0)
    standard_errors = fold_errors.std(axis=0, ddof=1) / np.sqrt(n_folds)
    # Scalar max_k -> (max_k + 1,)
    candidate_k = np.arange(max_k + 1)
    return {
        "candidate_k": candidate_k,
        "fold_errors": fold_errors,
        "mean_errors": mean_errors,
        "standard_errors": standard_errors,
        "k_reconstruction": int(np.argmin(mean_errors)),
    }
