"""Estimate GSN covariances and eigendecompose them for PCA diagnostics."""

from itertools import combinations
from typing import Any

import numpy as np

try:
    from gsn.perform_gsn import perform_gsn
except ImportError as error:
    perform_gsn = None
    GSN_IMPORT_ERROR = error
else:
    GSN_IMPORT_ERROR = None


def estimate_gsn_covariances(responses: np.ndarray) -> dict[str, Any]:
    """Estimate optimized signal and noise covariances with official GSN.

    Inputs:
        responses: np.ndarray with shape (n_images, n_voxels, n_repeats),
            containing repeated neural responses without additional scaling.

    Output:
        results: dict[str, Any], the unmodified ``perform_gsn`` result. It
            includes optimized signal covariance ``cSb``, optimized noise
            covariance ``cNb``, and the diagnostics returned by GSN.
    """

    response_array = np.asarray(responses, dtype=float)
    if response_array.ndim != 3:
        raise ValueError(
            "responses must have shape (n_images, n_voxels, n_repeats)."
        )
    if response_array.shape[2] < 2:
        raise ValueError("GSN requires at least two repeats per image.")
    if GSN_IMPORT_ERROR is not None:
        raise ImportError(
            "estimate_gsn_covariances requires the official cvnlab GSN package."
        ) from GSN_IMPORT_ERROR

    # (n_images, n_voxels, n_repeats) -> (n_voxels, n_images, n_repeats)
    gsn_input = np.transpose(response_array, (1, 0, 2))
    return perform_gsn(
        gsn_input,
        {"returns": ["cSb", "cNb"], "wantverbose": 0},
    )


def pca_signal_noise_covariances(
    signal_covariance: np.ndarray,
    noise_covariance: np.ndarray,
) -> dict[str, np.ndarray]:
    """Eigendecompose optimized GSN signal and noise covariance matrices.

    Inputs:
        signal_covariance: np.ndarray with shape (n_voxels, n_voxels), the
            optimized GSN signal covariance ``cSb``.
        noise_covariance: np.ndarray with shape (n_voxels, n_voxels), the
            optimized GSN noise covariance ``cNb``.

    Output:
        results: dict[str, np.ndarray] containing descending eigenvalues,
            column-wise eigenvectors, explained-variance ratios, and cumulative
            explained-variance ratios for both signal and noise.
    """

    covariances = {
        "signal": np.asarray(signal_covariance, dtype=float),
        "noise": np.asarray(noise_covariance, dtype=float),
    }
    if any(matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
           for matrix in covariances.values()):
        raise ValueError("Signal and noise covariances must be square matrices.")
    if covariances["signal"].shape != covariances["noise"].shape:
        raise ValueError("Signal and noise covariances must have matching shapes.")

    results = {}
    for name, covariance in covariances.items():
        if not np.all(np.isfinite(covariance)):
            raise ValueError(f"{name.capitalize()} covariance must be finite.")
        if not np.allclose(covariance, covariance.T, rtol=1e-7, atol=1e-10):
            raise ValueError(f"{name.capitalize()} covariance must be symmetric.")

        # (n_voxels, n_voxels) -> (n_voxels,), (n_voxels, n_voxels)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
        if eigenvalues[0] < -tolerance:
            raise ValueError(
                f"{name.capitalize()} covariance must be positive semidefinite."
            )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        eigenvectors = eigenvectors[:, order]
        total_variance = float(eigenvalues.sum())
        if total_variance == 0:
            raise ValueError(
                f"{name.capitalize()} covariance must contain positive variance."
            )

        # (n_voxels,) -> (n_components,)
        explained_variance_ratio = eigenvalues / total_variance
        # (n_components,) -> (n_components,)
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        results[f"{name}_eigenvalues"] = eigenvalues
        results[f"{name}_eigenvectors"] = eigenvectors
        results[f"{name}_explained_variance_ratio"] = explained_variance_ratio
        results[f"{name}_cumulative_variance_ratio"] = cumulative_variance_ratio

    return results


def select_gsn_pca_components(
    responses: np.ndarray,
    max_components: int | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Select GSN signal PCs by independent-repeat reconstruction.

    Repeat-split errors are weighted in proportion to the number of input
    repeats. With three repeats, the three two-repeat-input directions receive
    total weight 2/3 and the three one-repeat-input directions receive total
    weight 1/3. The target side necessarily has the complementary repeat count.

    Inputs:
        responses: np.ndarray with shape (n_images, n_voxels, n_repeats).
        max_components: int or None, largest candidate count; None tests every
            voxel dimension.
        n_folds: int, number of held-out image folds used for validation.
        seed: int, random seed used to assign images to folds.

    Output:
        results: dict[str, Any] containing candidate component counts, fold
            normalized reconstruction errors, their means and standard errors,
            the error-minimizing count, the one-standard-error threshold, and
            the smallest count within that threshold. Each validation score
            compares a projected repeat average with an independent repeat
            average; no DNN representations are used. It also reports that
            input-repeat weighting was used and the maximum-to-minimum
            per-split input-repeat weight ratio.
    """

    response_array = np.asarray(responses, dtype=float)
    if response_array.ndim != 3:
        raise ValueError(
            "responses must have shape (n_images, n_voxels, n_repeats)."
        )
    if not np.all(np.isfinite(response_array)):
        raise ValueError("responses must be finite.")
    n_images, n_voxels, n_repeats = response_array.shape
    if n_images < 3:
        raise ValueError("Component selection requires at least three images.")
    if n_repeats < 2:
        raise ValueError("Component selection requires at least two repeats.")
    if not isinstance(n_folds, (int, np.integer)):
        raise TypeError("n_folds must be an integer.")
    if not 2 <= n_folds <= n_images:
        raise ValueError("n_folds must be between 2 and n_images.")
    if max_components is None:
        max_components = n_voxels
    if not isinstance(max_components, (int, np.integer)):
        raise TypeError("max_components must be an integer or None.")
    if not 1 <= max_components <= n_voxels:
        raise ValueError("max_components must be between 1 and n_voxels.")

    rng = np.random.default_rng(seed)
    # (n_images,) -> n_folds arrays with shape (n_validation_images,)
    validation_folds = np.array_split(rng.permutation(n_images), n_folds)
    repeat_indices = np.arange(n_repeats)
    repeat_splits = []
    for group_a in combinations(repeat_indices, n_repeats // 2):
        group_b = tuple(index for index in repeat_indices if index not in group_a)
        if len(group_a) == len(group_b) and tuple(group_a) > group_b:
            continue
        repeat_splits.extend(((group_a, group_b), (group_b, group_a)))
    # n_repeat_splits tuples -> (n_repeat_splits,)
    input_repeat_counts = np.asarray(
        [len(input_repeats) for input_repeats, _ in repeat_splits],
        dtype=float,
    )
    # (n_repeat_splits,) -> (n_repeat_splits,)
    repeat_split_weights = input_repeat_counts / input_repeat_counts.sum()

    fold_errors = np.empty((n_folds, max_components))
    for fold_index, validation_indices in enumerate(validation_folds):
        training_mask = np.ones(n_images, dtype=bool)
        training_mask[validation_indices] = False
        training_responses = response_array[training_mask]
        validation_responses = response_array[validation_indices]
        gsn_results = estimate_gsn_covariances(training_responses)
        pca_results = pca_signal_noise_covariances(
            gsn_results["cSb"],
            gsn_results["cNb"],
        )
        # (n_voxels, n_voxels) -> (n_voxels, max_components)
        basis = pca_results["signal_eigenvectors"][:, :max_components]
        # (n_training_images, n_voxels, n_repeats) -> (n_voxels,)
        training_mean = training_responses.mean(axis=(0, 2))
        split_errors = np.empty((len(repeat_splits), max_components))

        for split_index, (input_repeats, target_repeats) in enumerate(repeat_splits):
            # (n_validation_images, n_voxels, n_input_repeats) ->
            # (n_validation_images, n_voxels)
            observed = validation_responses[:, :, input_repeats].mean(axis=2)
            # (n_validation_images, n_voxels, n_target_repeats) ->
            # (n_validation_images, n_voxels)
            target = validation_responses[:, :, target_repeats].mean(axis=2)
            centered = observed - training_mean
            # (n_validation_images, n_voxels) @
            # (n_voxels, max_components) ->
            # (n_validation_images, max_components)
            scores = centered @ basis
            # (n_voxels,) -> (n_validation_images, n_voxels)
            reconstruction = np.broadcast_to(training_mean, observed.shape).copy()
            baseline_error = np.mean((target - training_mean) ** 2)
            if baseline_error == 0:
                raise ValueError(
                    "Validation responses contain no reconstructable variance."
                )

            for component_index in range(max_components):
                # (n_validation_images, 1) * (1, n_voxels) ->
                # (n_validation_images, n_voxels)
                reconstruction += (
                    scores[:, component_index, None]
                    * basis[None, :, component_index]
                )
                split_errors[split_index, component_index] = (
                    np.mean((target - reconstruction) ** 2) / baseline_error
                )

        # (n_repeat_splits, max_components) -> (max_components,)
        fold_errors[fold_index] = np.average(
            split_errors,
            axis=0,
            weights=repeat_split_weights,
        )

    # (n_folds, max_components) -> (max_components,)
    mean_errors = fold_errors.mean(axis=0)
    # (n_folds, max_components) -> (max_components,)
    standard_errors = fold_errors.std(axis=0, ddof=1) / np.sqrt(n_folds)
    best_index = int(np.argmin(mean_errors))
    threshold = float(mean_errors[best_index] + standard_errors[best_index])
    selected_index = int(np.flatnonzero(mean_errors <= threshold)[0])
    # Scalars and arrays with shape (max_components,) or
    # (n_folds, max_components) -> dict[str, Any]
    return {
        "selected_n_components": selected_index + 1,
        "best_n_components": best_index + 1,
        "candidate_n_components": np.arange(1, max_components + 1),
        "fold_normalized_reconstruction_errors": fold_errors,
        "mean_normalized_reconstruction_errors": mean_errors,
        "reconstruction_error_standard_errors": standard_errors,
        "one_standard_error_threshold": threshold,
        "repeat_splits_weighted_by_input_count": True,
        "repeat_split_input_weight_ratio": float(
            input_repeat_counts.max() / input_repeat_counts.min()
        ),
    }
