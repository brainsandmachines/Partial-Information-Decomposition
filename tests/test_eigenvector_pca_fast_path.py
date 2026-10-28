import numpy as np
import pytest
import torch

import simulations.pca.rank.eigenvector_pca as eigenvector_pca
from simulations.pca.rank.eigenvector_pca import (
    eigenvector_pca_cv,
    fit_pca_loadings_svd,
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


def test_component_chunks_do_not_change_press(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check that splitting components into chunks leaves PRESS unchanged.

    Input:
        monkeypatch: pytest.MonkeyPatch, used to shrink the chunk size.

    Output:
        None. Assertions compare PRESS with one chunk and with chunks of two.
    """

    X = _low_rank_matrix()
    one_chunk = eigenvector_pca_cv(X, max_components=5, method_pca="SVD")
    monkeypatch.setattr(eigenvector_pca, "COMPONENT_CHUNK_SIZE", 2)
    small_chunks = eigenvector_pca_cv(X, max_components=5, method_pca="SVD")

    np.testing.assert_allclose(small_chunks.press, one_chunk.press, rtol=1e-12)


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
