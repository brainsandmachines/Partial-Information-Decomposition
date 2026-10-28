"""Cross-subject PLS: how many PLS ranks predict a second subject's responses.

One subject's repeat-averaged responses (source) predict another subject's
responses (target) to the same images. PLS k* can exceed the shared rank when
the source has private signal (suppressor components).
"""

from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from pipeline.rank_selection.data import RankData
from pipeline.rank_selection.result import RankResult


def pls_rank_curve(
    source: np.ndarray,
    target: np.ndarray,
    max_rank: int,
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Cross-validate the PLS rank for predicting target responses from source responses.

    Images are split into folds. In each fold, source and target are centered
    with training means (no scaling), one PLS regression with ``max_rank``
    components is fit, and held-out predictions for every rank k come from its
    first k components. The held-out MSE is normalized by the MSE of predicting
    the training mean, so rank 0 has error 1.

    Inputs:
        source: np.ndarray with shape (n_images, n_source_voxels).
        target: np.ndarray with shape (n_images, n_target_voxels), same image order.
        max_rank: int, largest PLS rank tested; at most n_training_images - 1.
        n_folds: int, number of held-out image folds.
        seed: int, random seed used to assign images to folds.

    Output:
        results: dict[str, Any] with ``rank`` of shape (max_rank + 1,) starting
            at 0, ``fold_normalized_mse`` of shape (n_folds, max_rank + 1),
            ``mean_normalized_mse`` and ``standard_error`` across folds,
            ``best_rank`` (k*, lowest mean), and ``one_se_rank`` (smallest rank
            whose mean is within one standard error of the lowest).
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        raise ValueError("source and target must be 2D with the same number of images.")
    if not (np.all(np.isfinite(source)) and np.all(np.isfinite(target))):
        raise ValueError("source and target must be finite.")
    n_images = source.shape[0]
    if not 2 <= n_folds <= n_images:
        raise ValueError("n_folds must be between 2 and n_images.")
    min_training = n_images - int(np.ceil(n_images / n_folds))
    if not 1 <= max_rank <= min_training - 1:
        raise ValueError(f"max_rank must be between 1 and {min_training - 1}.")

    rng = np.random.default_rng(seed)
    # (n_images,) -> n_folds arrays with shape (n_validation_images,)
    validation_folds = np.array_split(rng.permutation(n_images), n_folds)
    fold_normalized_mse = np.empty((n_folds, max_rank + 1))

    for fold_index, validation_indices in enumerate(validation_folds):
        print(f"Cross-subject PLS: fold {fold_index + 1} of {n_folds}...")
        training_mask = np.ones(n_images, dtype=bool)
        training_mask[validation_indices] = False

        # 1. Center with training means only.
        source_mean = source[training_mask].mean(axis=0)  # (n_train, v_s) -> (v_s,)
        target_mean = target[training_mask].mean(axis=0)  # (n_train, v_t) -> (v_t,)
        source_train = source[training_mask] - source_mean
        target_train = target[training_mask] - target_mean
        source_test = source[validation_indices] - source_mean
        target_test = target[validation_indices] - target_mean

        # 2. One PLS fit with max_rank components; the k-rank model uses its
        #    first k x-rotations and y-loadings.
        pls = PLSRegression(n_components=max_rank, scale=False)
        pls.fit(source_train, target_train)
        # (n_test, v_s) @ (v_s, max_rank) -> (n_test, max_rank)
        test_scores = source_test @ pls.x_rotations_

        # 3. Held-out MSE for every rank, normalized by the rank-0 error.
        baseline_mse = np.mean(target_test ** 2)
        if baseline_mse == 0:
            raise ValueError("Held-out target responses have no variance.")
        prediction = np.zeros_like(target_test)  # (n_test, v_t)
        fold_normalized_mse[fold_index, 0] = 1.0
        for rank_index in range(max_rank):
            # (n_test, 1) * (1, v_t) -> (n_test, v_t)
            prediction += (
                test_scores[:, rank_index, None]
                * pls.y_loadings_[None, :, rank_index]
            )
            fold_normalized_mse[fold_index, rank_index + 1] = (
                np.mean((target_test - prediction) ** 2) / baseline_mse
            )

    # (n_folds, max_rank + 1) -> (max_rank + 1,)
    mean_normalized_mse = fold_normalized_mse.mean(axis=0)
    standard_error = fold_normalized_mse.std(axis=0, ddof=1) / np.sqrt(n_folds)
    best_rank = int(np.argmin(mean_normalized_mse))
    threshold = mean_normalized_mse[best_rank] + standard_error[best_rank]
    one_se_rank = int(np.flatnonzero(mean_normalized_mse <= threshold)[0])
    return {
        "rank": np.arange(max_rank + 1),
        "fold_normalized_mse": fold_normalized_mse,
        "mean_normalized_mse": mean_normalized_mse,
        "standard_error": standard_error,
        "best_rank": best_rank,
        "one_se_rank": one_se_rank,
    }


def run(
    data: RankData,
    max_rank: int = 300,
    folds: int = 5,
    seed: int = 0,
) -> RankResult:
    """Run cross-subject PLS on a paired dataset.

    Inputs: data, RankData with a second subject (``target_responses``);
        max_rank, folds (image folds), seed as in ``pls_rank_curve``.
    Output: RankResult with ranks 0..max_rank, the mean normalized held-out MSE
        as score, k* (the minimum) as selected_rank, and the 1-SE rank.
    """
    curve = pls_rank_curve(
        data.averaged(), data.averaged_target(), max_rank=max_rank, n_folds=folds, seed=seed,
    )
    return RankResult(
        method="pls",
        dataset=data.name,
        seed=seed,
        selected_rank=curve["best_rank"],
        selection_rule="minimum mean normalized MSE (k*)",
        score_name="normalized held-out MSE of the target subject",
        rank=curve["rank"],
        score=curve["mean_normalized_mse"],
        standard_error=curve["standard_error"],
        fold_scores=curve["fold_normalized_mse"],
        best_rank=curve["best_rank"],
        one_se_rank=curve["one_se_rank"],
        true_rank=data.true_rank,
        info={"source_subject": data.subject, "target_subject": data.target_subject},
    )
