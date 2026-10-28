"""Eigenvector CV: fast path, refit path, torch backend (ported), K-fold and legacy checkpoints (new)."""

import numpy as np
import pytest
import torch

from pipeline.rank_selection.methods import gsn
from pipeline.rank_selection.methods.eigenvector import (
    eigenvector_kfold_curve,
    eigenvector_pca_cv,
    fit_pca_loadings_svd,
)
from pipeline.rank_selection.methods.eigenvector_checkpoint import (
    LEGACY_FIT_FUNCTION_NAMES,
    build_checkpoint_metadata,
    load_eigenvector_pca_checkpoint,
    write_eigenvector_pca_checkpoint,
)


def _low_rank_matrix() -> np.ndarray:
    """
    Build a deterministic rank-3 matrix plus noise for PRESS tests.

    Input:
        None.

    Output:
        X: np.ndarray of shape (9, 7), deterministic float test data.
    """

    rng = np.random.default_rng(7)
    scores = rng.normal(size=(9, 3))
    loadings = rng.normal(size=(3, 7))
    return scores @ loadings + 0.3 * rng.normal(size=(9, 7))  # (9, 3) @ (3, 7) -> (9, 7)


def _reference_press(
    X: np.ndarray,
    max_components: int,
    scale: bool,
) -> np.ndarray:
    """
    Compute eigenvector-CV PRESS by brute force, without the leverage formula.

    For each held-out sample i and component count f, refit SVD loadings on the
    other samples and predict each feature j of sample i by least squares on
    its remaining features.

    Input:
        X: np.ndarray of shape (n_samples, n_features), test data.
        max_components: int, largest component count evaluated.
        scale: bool, whether to standardize with training standard deviations.

    Output:
        press: np.ndarray of shape (max_components + 1,), summed squared errors.
    """

    n_samples, n_features = X.shape
    press = np.zeros(max_components + 1)
    for i in range(n_samples):
        X_train = np.delete(X, i, axis=0)  # (n_samples, n_features) -> (n_samples - 1, n_features)
        x_test = X[i] - X_train.mean(axis=0)  # (n_samples, n_features) -> (n_features,)
        X_train = X_train - X_train.mean(axis=0)
        if scale:
            train_std = X_train.std(axis=0, ddof=1)
            X_train = X_train / train_std
            x_test = x_test / train_std
        press[0] += np.sum(x_test ** 2)
        Vt = np.linalg.svd(X_train, full_matrices=False)[2]  # (n_samples - 1, n_features) -> (min(n_samples - 1, n_features), n_features)
        for f in range(1, max_components + 1):
            P = Vt[:f].T  # (min(n_samples - 1, n_features), n_features) -> (n_features, f)
            for j in range(n_features):
                P_minus_j = np.delete(P, j, axis=0)  # (n_features, f) -> (n_features - 1, f)
                t_hat = np.linalg.lstsq(P_minus_j, np.delete(x_test, j), rcond=None)[0]  # (n_features - 1,) -> (f,)
                press[f] += (x_test[j] - t_hat @ P[j]) ** 2
    return press


@pytest.mark.parametrize("scale", [False, True])
def test_fast_path_matches_brute_force_reference(scale: bool) -> None:
    """
    Check that the single-SVD fast path reproduces brute-force PRESS.

    Input:
        scale: bool, whether training standardization is applied.

    Output:
        None. Assertions compare PRESS against the brute-force reference.
    """

    X = _low_rank_matrix()
    result = eigenvector_pca_cv(
        X, max_components=5, center=True, scale=scale, method_pca="SVD"
    )

    np.testing.assert_allclose(result.press, _reference_press(X, 5, scale), rtol=1e-10)


def test_fast_path_matches_refit_path() -> None:
    """
    Check that the single-SVD path matches refitting PCA for each count.

    Input:
        None.

    Output:
        None. Assertions compare PRESS and the selected component count.
    """

    X = _low_rank_matrix()
    fast = eigenvector_pca_cv(X, max_components=5, method_pca="SVD")
    refit = eigenvector_pca_cv(
        X, max_components=5, pca_fit_fn=fit_pca_loadings_svd, method_pca=None
    )

    np.testing.assert_allclose(fast.press, refit.press, rtol=1e-10)
    assert fast.selected_n_components == refit.selected_n_components



@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available."
            ),
        ),
    ],
)
@pytest.mark.parametrize("scale", [False, True])
def test_torch_device_matches_numpy(device: str, scale: bool) -> None:
    """
    Check that the torch backend reproduces the NumPy PRESS.

    Input:
        device: str, torch device used for the computation.
        scale: bool, whether training standardization is applied.

    Output:
        None. Assertions compare PRESS and the selected component count.
    """

    X = _low_rank_matrix()
    numpy_result = eigenvector_pca_cv(X, max_components=5, scale=scale, method_pca="SVD")
    torch_result = eigenvector_pca_cv(
        X, max_components=5, scale=scale, method_pca="SVD", device=device
    )

    np.testing.assert_allclose(torch_result.press, numpy_result.press, rtol=1e-10)
    assert torch_result.selected_n_components == numpy_result.selected_n_components


def test_device_requires_svd_mode() -> None:
    """
    Check that the torch backend rejects the per-count refit mode.

    Input:
        None.

    Output:
        None. Assertion verifies the ValueError.
    """

    with pytest.raises(ValueError, match="method_pca='SVD'"):
        eigenvector_pca_cv(_low_rank_matrix(), max_components=3, device="cpu")


def _kfold_reference_press(
    responses: np.ndarray,
    basis_fn,
    max_components: int,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """
    Compute K-fold eigenvector PRESS by deleting each voxel and refitting scores.

    Input:
        responses: np.ndarray of shape (n_images, n_voxels, n_repeats).
        basis_fn: callable mapping training responses to an (n_voxels, n_voxels)
            basis with orthonormal columns in rank order.
        max_components: int, largest component count evaluated.
        n_folds: int, number of image folds.
        seed: int, fold seed.

    Output:
        press: np.ndarray of shape (max_components + 1,).
    """

    n_images, n_voxels, _ = responses.shape
    folds = np.array_split(np.random.default_rng(seed).permutation(n_images), n_folds)
    press = np.zeros(max_components + 1)
    for validation in folds:
        training = np.ones(n_images, bool)
        training[validation] = False
        mean = responses[training].mean(axis=(0, 2))  # (n_train, v, r) -> (v,)
        basis = basis_fn(responses[training])
        heldout = responses[validation].mean(axis=2) - mean  # (n_val, v, r) -> (n_val, v)
        press[0] += np.sum(heldout ** 2)
        for f in range(1, max_components + 1):
            P = basis[:, :f]  # (v, v) -> (v, f)
            for x in heldout:
                for j in range(n_voxels):
                    # (v - 1, f), (v - 1,) -> (f,)
                    t = np.linalg.lstsq(np.delete(P, j, 0), np.delete(x, j), rcond=None)[0]
                    press[f] += (x[j] - t @ P[j]) ** 2
    return press


def _repeated_toy() -> np.ndarray:
    """
    Build a small repeated-response array with a rank-3 signal.

    Input:
        None.

    Output:
        responses: np.ndarray of shape (40, 9, 3).
    """

    rng = np.random.default_rng(0)
    # (40, 3) @ (3, 9) -> (40, 9) -> (40, 9, 3)
    signal = (rng.normal(size=(40, 3)) @ rng.normal(size=(3, 9)))[:, :, None]
    return signal + 0.5 * rng.normal(size=(40, 9, 3))


def test_kfold_pca_basis_matches_brute_force() -> None:
    """
    Check K-fold PRESS with the PCA basis against explicit voxel deletion.

    Input:
        None.

    Output:
        None. Assertions compare PRESS and the selected count.
    """

    responses = _repeated_toy()

    def pca_basis(training: np.ndarray) -> np.ndarray:
        """Right singular vectors of the centered training repeat averages."""
        average = training.mean(axis=2)  # (n, v, r) -> (n, v)
        return np.linalg.svd(average - average.mean(0), full_matrices=False)[2].T

    result = eigenvector_kfold_curve(responses, basis="pca", max_components=6, n_folds=4, seed=3)
    reference = _kfold_reference_press(responses, pca_basis, 6, 4, 3)
    np.testing.assert_allclose(result["press"], reference, rtol=1e-10)
    assert result["selected_n_components"] == int(np.argmin(reference))


def test_kfold_gsn_basis_matches_brute_force(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check K-fold PRESS with the GSN basis, GSN replaced by a known covariance.

    Input:
        monkeypatch: pytest.MonkeyPatch, replaces GSN with the training
            repeat-average covariance.

    Output:
        None. Assertions compare PRESS with explicit voxel deletion.
    """

    responses = _repeated_toy()

    def fake_gsn(training: np.ndarray) -> dict:
        """Covariance of the training repeat averages as the signal covariance."""
        covariance = np.cov(training.mean(axis=2), rowvar=False)  # (n, v) -> (v, v)
        return {"cSb": covariance, "cNb": np.eye(covariance.shape[0])}

    def gsn_basis(training: np.ndarray) -> np.ndarray:
        """Eigenvectors of the fake signal covariance, largest first."""
        _, vectors = np.linalg.eigh(fake_gsn(training)["cSb"])
        return vectors[:, ::-1]

    monkeypatch.setattr(gsn, "estimate_gsn_covariances", fake_gsn)
    result = eigenvector_kfold_curve(responses, basis="gsn", max_components=6, n_folds=4, seed=3)
    reference = _kfold_reference_press(responses, gsn_basis, 6, 4, 3)
    np.testing.assert_allclose(result["press"], reference, rtol=1e-10)


def test_checkpoint_written_before_the_move_still_resumes(tmp_path) -> None:
    """
    Resume a checkpoint that records the fit function's old module path.

    Input:
        tmp_path: Path, pytest temporary directory.

    Output:
        None. Assertions verify that the legacy row loads and the result matches.
    """

    X = _low_rank_matrix()
    baseline = eigenvector_pca_cv(X, max_components=4, method_pca="SVD")
    path = tmp_path / "checkpoint.csv"
    metadata = build_checkpoint_metadata(X, 4, fit_pca_loadings_svd, True, False, True, "SVD", 1e-12)
    legacy_name = next(
        old for old, new in LEGACY_FIT_FUNCTION_NAMES.items() if new == metadata["pca_fit_fn"]
    )
    legacy_metadata = {**metadata, "pca_fit_fn": legacy_name}
    # Sample 0's PRESS from a fresh checkpoint, rewritten under the legacy name.
    fresh = tmp_path / "fresh.csv"
    eigenvector_pca_cv(X, max_components=4, method_pca="SVD", checkpoint_csv_path=fresh)
    rows = load_eigenvector_pca_checkpoint(fresh, metadata, 4)
    write_eigenvector_pca_checkpoint(path, legacy_metadata, {0: rows[0]}, 4)

    resumed = eigenvector_pca_cv(X, max_components=4, method_pca="SVD", checkpoint_csv_path=path)

    np.testing.assert_allclose(resumed.press, baseline.press, rtol=1e-12)
    assert len(load_eigenvector_pca_checkpoint(path, metadata, 4)) == X.shape[0]
