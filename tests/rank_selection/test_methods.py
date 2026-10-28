"""Tests for the PRESS kernel, MLE, PLS, variance threshold, cross-decomposition, and registry."""

import numpy as np
import pytest
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from pipeline.rank_selection.data import make_toy
from pipeline.rank_selection.methods import mle, pls, variance
from pipeline.rank_selection.press import direct_press, eigenvector_press
from pipeline.rank_selection.registry import METHODS, unmet_requirement

CROSS_DECOMPOSITION_ERROR = METHODS["cross_decomposition"].import_error
try:
    from pipeline.rank_selection.methods.cross_decomposition import leading_significant_count
except Exception:
    leading_significant_count = None  # needs the cross_decomposition environment


def _orthonormal_basis(n_voxels: int, n_components: int, seed: int) -> np.ndarray:
    """
    Draw a basis with orthonormal columns.

    Input:
        n_voxels: int, rows.
        n_components: int, columns.
        seed: int, random seed.

    Output:
        basis: np.ndarray of shape (n_voxels, n_components).
    """

    # (n_voxels, n_components) -> (n_voxels, n_components)
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(n_voxels, n_components)))[0]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_press_kernel_matches_explicit_voxel_deletion(backend: str) -> None:
    """
    Compare the r_j / (1 - h_j) kernel with refitting after deleting each voxel.

    Input:
        backend: str, ``"numpy"`` or ``"torch"`` inputs to the kernel.

    Output:
        None. Assertions compare PRESS for every component count.
    """

    rng = np.random.default_rng(1)
    heldout = rng.normal(size=(5, 12))  # (n_rows, n_voxels)
    basis = _orthonormal_basis(12, 6, seed=2)
    inputs = (heldout, basis) if backend == "numpy" else (torch.tensor(heldout), torch.tensor(basis))

    press, stable = eigenvector_press(*inputs)

    reference = [
        sum(direct_press(row, basis[:, :k], eps=1e-12) for row in heldout) for k in range(1, 7)
    ]
    np.testing.assert_allclose(press, reference, rtol=1e-10)
    assert stable.all()


def test_press_kernel_flags_a_voxel_the_basis_fully_explains() -> None:
    """
    Flag counts where a voxel's leverage reaches 1 (it cannot be predicted from others).

    Input:
        None.

    Output:
        None. The count whose basis contains a voxel's unit vector is unstable.
    """

    basis = np.eye(4)[:, :2]  # voxel 0 and voxel 1 unit vectors
    _, stable = eigenvector_press(np.ones((1, 4)), basis)
    assert stable.tolist() == [False, False]


def test_mle_matches_sklearn_and_recovers_isotropic_rank() -> None:
    """
    Check the MLE count against sklearn's PCA(n_components='mle') directly.

    Input:
        None.

    Output:
        None. Assertions compare with sklearn and the known rank.
    """

    toy = make_toy(n_images=400, n_voxels=30, true_rank=5, noise_std=0.5, seed=0)
    result = mle.run(toy)
    expected = PCA(n_components="mle", svd_solver="full").fit(toy.averaged()).n_components_
    assert result.selected_rank == expected == 5


def test_pls_truncated_ranks_equal_separate_fits() -> None:
    """
    Check that rank k from one PLS fit equals a separate k-component fit.

    Input:
        None.

    Output:
        None. pls_rank_curve relies on this for every rank.
    """

    toy = make_toy(n_images=150, n_voxels=20, true_rank=3, n_subjects=2, n_target_voxels=15, seed=4)
    source, target = toy.averaged(), toy.averaged_target()
    train, test = slice(0, 120), slice(120, None)
    source_train = source[train] - source[train].mean(0)
    target_train = target[train] - target[train].mean(0)
    source_test = source[test] - source[train].mean(0)
    full = PLSRegression(n_components=8, scale=False).fit(source_train, target_train)
    for rank in range(1, 9):
        # (n_test, v_s) @ (v_s, rank) @ (rank, v_t) -> (n_test, v_t)
        truncated = source_test @ full.x_rotations_[:, :rank] @ full.y_loadings_[:, :rank].T
        separate = PLSRegression(n_components=rank, scale=False).fit(
            source_train, target_train
        ).predict(source_test) - target_train.mean(0)
        np.testing.assert_allclose(truncated, separate, rtol=1e-8, atol=1e-10)


def test_pls_recovers_shared_rank_without_private_signal() -> None:
    """
    Select the shared rank when subjects share signal and have no private signal.

    Input:
        None.

    Output:
        None. Assertion on the selected rank.
    """

    toy = make_toy(n_images=600, n_voxels=40, true_rank=4, noise_std=1.0, n_subjects=2, seed=1)
    assert pls.run(toy, max_rank=12, folds=5, seed=0).selected_rank == 4


def test_variance_threshold_matches_sklearn_and_heldout_curve_is_monotone() -> None:
    """
    Check the threshold count against sklearn and the naive held-out curve shape.

    Input:
        None.

    Output:
        None. The held-out reconstruction error never rises with more PCs.
    """

    toy = make_toy(n_images=200, n_voxels=25, true_rank=3, seed=2)
    result = variance.run(toy, variance_threshold=0.8, folds=4, seed=0)
    assert result.selected_rank == PCA(n_components=0.8, svd_solver="full").fit(toy.averaged()).n_components_
    heldout = result.extra_curves["heldout_normalized_mse"]
    finite = heldout[np.isfinite(heldout)]
    assert finite[0] == pytest.approx(1.0)
    assert np.all(np.diff(finite) <= 1e-12)


def test_leading_significant_count_rule() -> None:
    """
    Count only the leading run of significant components.

    Input:
        None.

    Output:
        None. Uses the authors' table from the old-code reference toy, where
        components 1-3 and 7-8 are significant.
    """

    if leading_significant_count is None:
        pytest.skip("the authors' lib/bonner are not installed in this environment")
    flags = np.array([True, True, True, False, False, False, True, True])
    assert leading_significant_count(flags) == 3
    assert leading_significant_count(np.ones(4, bool)) == 4
    assert leading_significant_count(np.zeros(4, bool)) == 0


def test_requirements_explain_why_a_method_cannot_run() -> None:
    """
    Report unmet data requirements instead of failing inside a method.

    Input:
        None.

    Output:
        None. Assertions on the returned reasons.
    """

    single_repeat = make_toy(n_images=50, n_voxels=60, true_rank=3, n_repeats=1, seed=0)
    assert "repeats" in unmet_requirement("repeat_reconstruction", {}, single_repeat)
    assert "repeats" in unmet_requirement("eigenvector", {"basis": "gsn"}, single_repeat)
    assert unmet_requirement("eigenvector", {"basis": "pca"}, single_repeat) is None
    assert "second subject" in unmet_requirement("pls", {}, single_repeat)
    assert "n_images >= n_voxels" in unmet_requirement("mle", {}, single_repeat)
