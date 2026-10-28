"""Naive PCA baselines: a training-variance threshold and plain held-out projection.

The count is the smallest number of PCs explaining ``variance_threshold`` of the
variance of all images, exactly as ``subject_pca.pca_by_variance`` chooses it.
For comparison, a K-fold curve projects held-out images onto PCs fitted
without them and reports the normalized reconstruction error. That error can
only fall as PCs are added, so it cannot pick a count by itself (Bro et al.,
2008); it is reported, not used for the selection.
"""

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from pipeline.rank_selection.data import RankData
from pipeline.rank_selection.result import RankResult


def variance_threshold_curve(
    averaged: np.ndarray,
    variance_threshold: float = 0.99,
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Choose PCs by a variance threshold and compute the naive held-out curve.

    Inputs:
        averaged: np.ndarray (n_images, n_voxels), repeat-averaged responses.
        variance_threshold: float in (0, 1]; 1.0 keeps every component.
        n_folds: int, image folds for the held-out curve.
        seed: int, fold seed.

    Output:
        results: dict with ``selected_n_components`` (sklearn's count for the
            threshold), ``candidate_n_components`` 0..K with
            K = min(n_images, n_voxels), the all-image
            ``cumulative_explained_variance_ratio`` (K + 1,), and the held-out
            ``heldout_normalized_mse`` (K + 1,) with its fold values
            ``fold_heldout_normalized_mse`` (n_folds, K + 1). Counts a training
            fold cannot fit are NaN.
    """
    data = np.asarray(averaged, dtype=float)
    if data.ndim != 2:
        raise ValueError("averaged must have shape (n_images, n_voxels).")
    if not np.all(np.isfinite(data)):
        raise ValueError("averaged must be finite.")
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1].")
    n_images, n_voxels = data.shape
    if not 2 <= n_folds <= n_images:
        raise ValueError("n_folds must be between 2 and n_images.")
    largest = min(n_images, n_voxels)

    # 1. The selection, as in subject_pca.pca_by_variance.
    components = None if variance_threshold == 1.0 else variance_threshold
    selected = int(PCA(n_components=components, svd_solver="full").fit(data).n_components_)
    full = PCA(svd_solver="full").fit(data)
    # (largest,) -> (largest + 1,), starting at 0 PCs
    cumulative = np.r_[0.0, np.cumsum(full.explained_variance_ratio_)]

    # 2. Naive held-out reconstruction error over image folds.
    rng = np.random.default_rng(seed)
    # (n_images,) -> n_folds arrays with shape (n_validation_images,)
    validation_folds = np.array_split(rng.permutation(n_images), n_folds)
    fold_errors = np.full((n_folds, largest + 1), np.nan)
    for fold_index, validation_indices in enumerate(validation_folds):
        training_mask = np.ones(n_images, dtype=bool)
        training_mask[validation_indices] = False
        training_mean = data[training_mask].mean(axis=0)  # (n_training, n_voxels) -> (n_voxels,)
        # (n_training, n_voxels) -> Vt: (min(n_training, n_voxels), n_voxels)
        _, _, Vt = np.linalg.svd(data[training_mask] - training_mean, full_matrices=False)
        heldout = data[validation_indices] - training_mean  # (n_validation, n_voxels)
        # (n_validation, n_voxels) @ (n_voxels, n_basis) -> (n_validation, n_basis)
        scores = heldout @ Vt.T
        baseline = np.sum(heldout ** 2)
        # Orthonormal basis: SSE(k) = SSE(0) - sum of the first k squared scores.
        sse = baseline - np.r_[0.0, np.cumsum(np.sum(scores ** 2, axis=0))]  # (n_basis + 1,)
        fold_errors[fold_index, : sse.shape[0]] = sse / baseline

    return {
        "selected_n_components": selected,
        "candidate_n_components": np.arange(largest + 1),
        "cumulative_explained_variance_ratio": cumulative,
        "heldout_normalized_mse": fold_errors.mean(axis=0),  # (n_folds, K + 1) -> (K + 1,)
        "fold_heldout_normalized_mse": fold_errors,
    }


def run(
    data: RankData,
    variance_threshold: float = 0.99,
    folds: int = 5,
    seed: int = 0,
) -> RankResult:
    """Run the variance-threshold baseline on a dataset.

    Inputs: data, RankData (responses averaged over repeats); variance_threshold,
        folds, seed as in ``variance_threshold_curve``.
    Output: RankResult whose score is the cumulative explained-variance ratio
        and whose extra curve is the naive held-out normalized MSE.
    """
    curve = variance_threshold_curve(
        data.averaged(), variance_threshold=variance_threshold, n_folds=folds, seed=seed,
    )
    return RankResult(
        method=f"variance_threshold[threshold={variance_threshold}]",
        dataset=data.name,
        seed=seed,
        selected_rank=curve["selected_n_components"],
        selection_rule=f"smallest rank explaining {variance_threshold} of all-image variance",
        score_name="cumulative explained variance ratio (all images)",
        rank=curve["candidate_n_components"],
        score=curve["cumulative_explained_variance_ratio"],
        true_rank=data.true_rank,
        extra_curves={"heldout_normalized_mse": curve["heldout_normalized_mse"]},
        info={"variance_threshold": variance_threshold},
    )
