"""Tests for the standalone GSN covariance PCA utilities."""

from pathlib import Path
import sys

import numpy as np
import pytest

try:
    import xarray as xr
except ImportError as error:
    xr = None
    XR_IMPORT_ERROR = error
else:
    XR_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pipeline.testing.gsn import gsn_pca
from pipeline.testing.gsn.gsn_pca import (
    GSN_IMPORT_ERROR,
    estimate_gsn_covariances,
    pca_signal_noise_covariances,
    select_gsn_pca_components,
)


REAL_DATA_ROOT = Path(
    "/groups/golan_neurogroup/bml_group/datasets/nsddata/otc_betas/"
    "otc_betas_img_vox_rep"
)
REAL_DATA_STORES = (
    sorted(REAL_DATA_ROOT.glob("*_betas_img_vox_rep.zarr"))
    if REAL_DATA_ROOT.exists()
    else []
)


def test_estimate_gsn_covariances_transposes_and_returns_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass GSN its required axis order and return its result object unchanged."""

    captured = {}
    expected = {"cSb": np.eye(2), "cNb": np.eye(2)}

    def fake_perform_gsn(data: np.ndarray, options: dict) -> dict:
        """Capture the wrapper call and return a fixed GSN-like dictionary."""

        captured["data"] = data
        captured["options"] = options
        return expected

    monkeypatch.setattr(gsn_pca, "GSN_IMPORT_ERROR", None)
    monkeypatch.setattr(gsn_pca, "perform_gsn", fake_perform_gsn)
    # (12,) -> (3, 2, 2)
    responses = np.arange(12.0).reshape(3, 2, 2)

    result = estimate_gsn_covariances(responses)

    assert result is expected
    assert captured["data"].shape == (2, 3, 2)
    # (3, 2, 2) -> (2, 3, 2)
    np.testing.assert_array_equal(captured["data"], responses.transpose(1, 0, 2))
    assert captured["options"] == {
        "returns": ["cSb", "cNb"],
        "wantverbose": 0,
    }


def test_pca_signal_noise_covariances_known_spectra() -> None:
    """Recover known descending PCA spectra from diagonal covariances."""

    signal_covariance = np.diag([1.0, 4.0, 2.0])
    noise_covariance = np.diag([3.0, 1.0, 2.0])

    results = pca_signal_noise_covariances(
        signal_covariance,
        noise_covariance,
    )

    np.testing.assert_allclose(results["signal_eigenvalues"], [4.0, 2.0, 1.0])
    np.testing.assert_allclose(results["noise_eigenvalues"], [3.0, 2.0, 1.0])
    np.testing.assert_allclose(
        np.abs(results["signal_eigenvectors"]),
        np.eye(3)[:, [1, 2, 0]],
    )
    np.testing.assert_allclose(
        results["signal_explained_variance_ratio"],
        np.array([4.0, 2.0, 1.0]) / 7.0,
    )
    np.testing.assert_allclose(
        results["signal_cumulative_variance_ratio"],
        np.cumsum(np.array([4.0, 2.0, 1.0]) / 7.0),
    )

    with pytest.raises(ValueError, match="square"):
        pca_signal_noise_covariances(np.ones((2, 3)), np.eye(2))
    with pytest.raises(ValueError, match="matching"):
        pca_signal_noise_covariances(np.eye(2), np.eye(3))
    with pytest.raises(ValueError, match="symmetric"):
        pca_signal_noise_covariances(np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(2))


@pytest.mark.skipif(GSN_IMPORT_ERROR is not None, reason="official GSN is unavailable")
def test_gsn_recovers_stable_low_rank_signal_subspace() -> None:
    """Recover a known low-rank signal subspace in two stimulus halves."""

    rng = np.random.default_rng(0)
    raw_basis = rng.normal(size=(6, 2))
    # (6, 2) -> (6, 2)
    true_basis, _ = np.linalg.qr(raw_basis)
    scores = rng.normal(size=(400, 2)) * np.array([3.0, 1.0])
    # (400, 2) @ (2, 6) -> (400, 6)
    signal = scores @ true_basis.T
    noise = 0.2 * rng.normal(size=(400, 6, 4))
    # (400, 6) -> (400, 6, 4)
    responses = signal[:, :, None] + noise

    full_gsn = estimate_gsn_covariances(responses)
    assert full_gsn["cSb"].shape == (6, 6)
    assert full_gsn["cNb"].shape == (6, 6)
    np.testing.assert_allclose(full_gsn["cSb"], full_gsn["cSb"].T, atol=1e-10)
    np.testing.assert_allclose(full_gsn["cNb"], full_gsn["cNb"].T, atol=1e-10)
    assert np.linalg.eigvalsh(full_gsn["cSb"]).min() >= -1e-8
    assert np.linalg.eigvalsh(full_gsn["cNb"]).min() >= -1e-8

    full_pca = pca_signal_noise_covariances(full_gsn["cSb"], full_gsn["cNb"])
    # (6, 6) -> (6, 2)
    recovered_basis = full_pca["signal_eigenvectors"][:, :2]
    recovery = np.linalg.norm(true_basis.T @ recovered_basis, ord="fro") ** 2 / 2
    assert recovery > 0.9

    half_pcas = []
    for half in np.array_split(responses, 2, axis=0):
        half_gsn = estimate_gsn_covariances(half)
        half_pcas.append(
            pca_signal_noise_covariances(half_gsn["cSb"], half_gsn["cNb"])
        )
    # Two (6, 6) bases -> two (6, 2) leading subspaces.
    left_basis = half_pcas[0]["signal_eigenvectors"][:, :2]
    right_basis = half_pcas[1]["signal_eigenvectors"][:, :2]
    stability = np.linalg.norm(left_basis.T @ right_basis, ord="fro") ** 2 / 2
    assert stability > 0.9

    selection = select_gsn_pca_components(
        responses,
        max_components=5,
        n_folds=4,
        seed=0,
    )
    assert selection["selected_n_components"] == 2
    assert selection["fold_normalized_reconstruction_errors"].shape == (4, 5)


@pytest.mark.skipif(
    GSN_IMPORT_ERROR is not None
    or XR_IMPORT_ERROR is not None
    or not REAL_DATA_STORES,
    reason="official GSN, xarray, or cluster OTC trial data is unavailable",
)
def test_real_otc_trial_data_smoke() -> None:
    """Run GSN and covariance PCA on a small deterministic OTC data subset."""

    dataset = xr.open_zarr(REAL_DATA_STORES[0])
    assert "betas_by_image_voxel_repeat" in dataset
    betas = dataset["betas_by_image_voxel_repeat"].transpose(
        "image", "voxel", "repeat"
    )
    # (n_images, n_voxels, n_repeats) -> (<=200, <=128, n_repeats)
    responses = betas.isel(image=slice(200), voxel=slice(128)).load().values

    gsn_results = estimate_gsn_covariances(responses)
    pca_results = pca_signal_noise_covariances(
        gsn_results["cSb"],
        gsn_results["cNb"],
    )

    n_voxels = responses.shape[1]
    assert gsn_results["cSb"].shape == (n_voxels, n_voxels)
    assert gsn_results["cNb"].shape == (n_voxels, n_voxels)
    assert np.all(np.isfinite(pca_results["signal_eigenvalues"]))
    assert np.all(np.diff(pca_results["signal_eigenvalues"]) <= 0)
    assert np.all(np.diff(pca_results["noise_eigenvalues"]) <= 0)
