"""Independent-repeat reconstruction: choose PCs that reconstruct a new repeat.

Held-out images are projected from one group of repeats and compared with the
average of the complementary repeats, whose noise is independent of the input.
The basis is either GSN's signal covariance or ordinary PCA of the training images.
"""

from itertools import combinations
from typing import Any

import numpy as np

from pipeline.rank_selection.data import RankData
from pipeline.rank_selection.methods import gsn
from pipeline.rank_selection.result import RankResult

BASES = ("gsn", "pca")


def repeat_reconstruction_curve(
    responses: np.ndarray,
    basis: str = "gsn",
    max_components: int | None = None,
    n_folds: int = 5,
    seed: int = 0,
    repeat_split_weighting: str = "input_repeat_count",
) -> dict[str, Any]:
    """Select PCs by independent-repeat reconstruction of held-out images.

    In each image fold the basis is fitted on the training images only:
    ``basis="gsn"`` uses the eigenvectors of GSN's signal covariance (all
    training repeats); ``basis="pca"`` uses the right singular vectors of the
    training images' repeat-averaged, training-mean-centered responses.

    Repeat-split errors can be weighted in proportion to the number of input
    repeats. With three repeats, this gives the three two-repeat-input
    directions total weight 2/3 and the three one-repeat-input directions total
    weight 1/3. Equal weighting remains available for comparison. The target
    side necessarily has the complementary repeat count.

    Inputs:
        responses: np.ndarray with shape (n_images, n_voxels, n_repeats).
        basis: str in BASES.
        max_components: int or None, largest candidate count; None tests every
            voxel dimension (bounded by the training images for ``"pca"``).
        n_folds: int, number of held-out image folds used for validation.
        seed: int, random seed used to assign images to folds.
        repeat_split_weighting: str, ``"input_repeat_count"`` weights each
            split by its number of input repeats, while ``"equal"`` gives
            every directed repeat split equal weight.

    Output:
        results: dict[str, Any] containing candidate component counts, fold
            normalized reconstruction errors, their means and standard errors,
            the error-minimizing count, the one-standard-error threshold, and
            the smallest count within that threshold. Each validation score
            compares a projected repeat average with an independent repeat
            average; no DNN representations are used. It also reports whether
            input-repeat weighting was used and the maximum-to-minimum
            per-split weight ratio.
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
    if repeat_split_weighting not in ("equal", "input_repeat_count"):
        raise ValueError(
            "repeat_split_weighting must be 'equal' or 'input_repeat_count'."
        )
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}.")
    largest = n_voxels
    if basis == "pca":
        largest = min(n_voxels, n_images - int(np.ceil(n_images / n_folds)))
    if max_components is None:
        max_components = largest
    if not isinstance(max_components, (int, np.integer)):
        raise TypeError("max_components must be an integer or None.")
    if not 1 <= max_components <= largest:
        raise ValueError(f"max_components must be between 1 and {largest}.")

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
    if repeat_split_weighting == "input_repeat_count":
        unnormalized_weights = input_repeat_counts
    else:
        # (n_repeat_splits,) -> (n_repeat_splits,)
        unnormalized_weights = np.ones_like(input_repeat_counts)
    # (n_repeat_splits,) -> (n_repeat_splits,)
    repeat_split_weights = unnormalized_weights / unnormalized_weights.sum()

    fold_errors = np.empty((n_folds, max_components))
    for fold_index, validation_indices in enumerate(validation_folds):
        training_mask = np.ones(n_images, dtype=bool)
        training_mask[validation_indices] = False
        training_responses = response_array[training_mask]
        validation_responses = response_array[validation_indices]
        # (n_training_images, n_voxels, n_repeats) -> (n_voxels,)
        training_mean = training_responses.mean(axis=(0, 2))
        if basis == "gsn":
            gsn_results = gsn.estimate_gsn_covariances(training_responses)
            pca_results = gsn.pca_signal_noise_covariances(
                gsn_results["cSb"],
                gsn_results["cNb"],
            )
            # (n_voxels, n_voxels) -> (n_voxels, max_components)
            basis_vectors = pca_results["signal_eigenvectors"][:, :max_components]
        else:
            # (n_training_images, n_voxels, n_repeats) -> (n_training_images, n_voxels)
            training_average = training_responses.mean(axis=2) - training_mean
            # (n_training_images, n_voxels) -> Vt: (min(n_training_images, n_voxels), n_voxels)
            _, _, Vt = np.linalg.svd(training_average, full_matrices=False)
            basis_vectors = Vt[:max_components].T  # -> (n_voxels, max_components)
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
            scores = centered @ basis_vectors
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
                    * basis_vectors[None, :, component_index]
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
        "repeat_splits_weighted_by_input_count": (
            repeat_split_weighting == "input_repeat_count"
        ),
        "repeat_split_input_weight_ratio": float(
            repeat_split_weights.max() / repeat_split_weights.min()
        ),
    }



def run(
    data: RankData,
    basis: str = "gsn",
    max_components: int | None = None,
    folds: int = 5,
    seed: int = 0,
    repeat_split_weighting: str = "input_repeat_count",
) -> RankResult:
    """Run independent-repeat reconstruction on a dataset.

    Inputs: data, RankData with at least two repeats; basis, max_components,
        folds (number of image folds), seed, repeat_split_weighting as in
        ``repeat_reconstruction_curve``.
    Output: RankResult with ranks 1..max_components, the mean normalized error
        as score, the one-SE count as selected_rank (the original GSN rule),
        and the minimum as best_rank.
    """
    curve = repeat_reconstruction_curve(
        data.responses, basis=basis, max_components=max_components,
        n_folds=folds, seed=seed, repeat_split_weighting=repeat_split_weighting,
    )
    return RankResult(
        method=f"repeat_reconstruction[basis={basis}]",
        dataset=data.name,
        seed=seed,
        selected_rank=curve["selected_n_components"],
        selection_rule="smallest rank within one SE of the minimum",
        score_name="normalized independent-repeat reconstruction MSE",
        rank=curve["candidate_n_components"],
        score=curve["mean_normalized_reconstruction_errors"],
        standard_error=curve["reconstruction_error_standard_errors"],
        fold_scores=curve["fold_normalized_reconstruction_errors"],
        best_rank=curve["best_n_components"],
        one_se_rank=curve["selected_n_components"],
        true_rank=data.true_rank,
        info={
            "repeat_split_weighting": repeat_split_weighting,
            "repeat_split_input_weight_ratio": curve["repeat_split_input_weight_ratio"],
        },
    )
