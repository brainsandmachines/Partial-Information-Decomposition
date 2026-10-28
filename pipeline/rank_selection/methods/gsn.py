"""Estimate GSN signal and noise covariances and eigendecompose them."""

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

