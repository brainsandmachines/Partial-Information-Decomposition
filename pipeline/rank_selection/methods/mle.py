"""Minka's MLE of the PCA dimension, via sklearn's PCA(n_components="mle").

Assumes isotropic noise: unequal or voxel-correlated noise inflates the count.
"""

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from pipeline.rank_selection.data import RankData
from pipeline.rank_selection.result import RankResult


def pca_mle(responses: np.ndarray) -> dict[str, Any]:
    """Select the PC count with Minka's MLE, as ``PCA(n_components='mle')``.

    The repeat-averaged responses of all images are fit with
    ``PCA(n_components='mle', svd_solver='full')``, which scores every rank
    with Minka's (2000) Laplace-approximated evidence under probabilistic PCA
    with isotropic noise and keeps the best one. No images are held out.
    sklearn scores the ranks in pure Python, so ~8000 voxels take hours.

    Inputs:
        responses: np.ndarray with shape (n_images, n_voxels, n_repeats);
            requires n_images >= n_voxels, as sklearn does for 'mle'.

    Output:
        results: dict[str, Any] with ``selected_n_components`` (sklearn's
            ``n_components_``), ``candidate_n_components`` of shape
            (selected_n_components,) for PCs 1..selected, and
            ``explained_variance`` (variance of each retained PC) of that shape.
    """

    response_array = np.asarray(responses, dtype=float)
    if response_array.ndim != 3:
        raise ValueError(
            "responses must have shape (n_images, n_voxels, n_repeats)."
        )
    if not np.all(np.isfinite(response_array)):
        raise ValueError("responses must be finite.")
    n_images, n_voxels, _ = response_array.shape
    if n_images < n_voxels:
        raise ValueError(
            "Minka's MLE requires n_images >= n_voxels, as sklearn's 'mle' does."
        )

    # 1. Average each image over its repeats.
    # (n_images, n_voxels, n_repeats) -> (n_images, n_voxels)
    averaged = response_array.mean(axis=2)

    # 2. sklearn's Minka MLE (it centers the data itself).
    print(f"Minka MLE: fitting PCA(n_components='mle') on {averaged.shape}...")
    pca = PCA(n_components="mle", svd_solver="full").fit(averaged)
    selected_n_components = int(pca.n_components_)

    return {
        "selected_n_components": selected_n_components,
        "candidate_n_components": np.arange(1, selected_n_components + 1),
        "explained_variance": pca.explained_variance_,  # (selected_n_components,)
    }


def run(data: RankData) -> RankResult:
    """Run Minka's MLE on a dataset (deterministic; no folds or seed).

    Input: data, RankData with n_images >= n_voxels.
    Output: RankResult with ranks 1..selected and each retained PC's variance
        as score; selected_rank is sklearn's ``n_components_``.
    """
    result = pca_mle(data.responses)
    return RankResult(
        method="mle",
        dataset=data.name,
        seed=None,
        selected_rank=result["selected_n_components"],
        selection_rule="maximum Minka evidence (sklearn PCA n_components='mle')",
        score_name="explained variance of each retained PC",
        rank=result["candidate_n_components"],
        score=result["explained_variance"],
        true_rank=data.true_rank,
    )
