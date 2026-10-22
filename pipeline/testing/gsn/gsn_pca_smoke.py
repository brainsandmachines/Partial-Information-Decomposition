"""Debuggable smoke example for GSN PCA with known signal rank."""

from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pipeline.testing.gsn.gsn_pca import (
    estimate_gsn_covariances,
    pca_signal_noise_covariances,
    select_gsn_pca_components,
)


def run_gsn_pca_smoke(
    n_images: int = 500,
    n_voxels: int = 50,
    true_rank: int = 5,
    n_repeats: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Run GSN PCA on repeated responses with known low-rank signal.

    Inputs:
        n_images: int, number of independently generated stimulus conditions.
        n_voxels: int, observed neural dimensions containing the latent signal.
        true_rank: int, exact rank of the stimulus-related signal covariance.
        n_repeats: int, independent noisy measurements of each stimulus.
        seed: int, NumPy random seed for reproducibility.

    Output:
        results: dict[str, Any] containing responses with shape
            (n_images, n_voxels, n_repeats), the noiseless signal, true signal
            and noise covariances, their bases and eigenvalues, the official
            GSN results, PCA and component-selection results, and four
            validation metrics.
    """

    if not 1 <= true_rank < n_voxels:
        raise ValueError("true_rank must be between 1 and n_voxels - 1.")
    if n_images <= true_rank:
        raise ValueError("n_images must be greater than true_rank.")
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2 for GSN.")

    rng = np.random.default_rng(seed)
    signal_eigenvalues = np.geomspace(9.0, 1.0, true_rank)
    noise_eigenvalues = np.linspace(2.0, 0.2, n_voxels)

    # (n_voxels, true_rank) -> (n_voxels, true_rank)
    signal_basis, _ = np.linalg.qr(rng.normal(size=(n_voxels, true_rank)))
    latent = rng.normal(size=(n_images, true_rank))
    latent -= latent.mean(axis=0, keepdims=True)
    # (n_images, true_rank) -> (n_images, true_rank)
    latent_basis, _ = np.linalg.qr(latent)
    scores = latent_basis * np.sqrt((n_images - 1) * signal_eigenvalues)
    # (n_images, true_rank) @ (true_rank, n_voxels) -> (n_images, n_voxels)
    signal = scores @ signal_basis.T
    # (n_voxels, true_rank) @ (true_rank, n_voxels) -> (n_voxels, n_voxels)
    true_signal_covariance = (
        signal_basis * signal_eigenvalues
    ) @ signal_basis.T

    # (n_voxels, n_voxels) -> (n_voxels, n_voxels)
    noise_basis, _ = np.linalg.qr(rng.normal(size=(n_voxels, n_voxels)))
    # (n_voxels, n_voxels) @ (n_voxels, n_voxels) -> (n_voxels, n_voxels)
    true_noise_covariance = (noise_basis * noise_eigenvalues) @ noise_basis.T
    noise_cholesky = np.linalg.cholesky(true_noise_covariance)
    standard_noise = rng.normal(size=(n_images, n_repeats, n_voxels))
    # (n_images, n_repeats, n_voxels) -> (n_images, n_voxels, n_repeats)
    noise = (standard_noise @ noise_cholesky.T).transpose(0, 2, 1)
    # (n_images, n_voxels) -> (n_images, n_voxels, n_repeats)
    responses = signal[:, :, None] + noise

    gsn_results = estimate_gsn_covariances(responses)
    pca_results = pca_signal_noise_covariances(
        gsn_results["cSb"],
        gsn_results["cNb"],
    )

    # (n_voxels, n_voxels) -> (n_voxels, true_rank)
    estimated_signal_basis = pca_results["signal_eigenvectors"][:, :true_rank]
    subspace_overlap = (
        np.linalg.norm(signal_basis.T @ estimated_signal_basis, ord="fro") ** 2
        / true_rank
    )
    selection_results = select_gsn_pca_components(
        responses,
        n_folds=5,
        seed=seed,
    )
    signal_eigenvalues_estimated = pca_results["signal_eigenvalues"]
    metrics = {
        "selected_n_components": selection_results["selected_n_components"],
        "minimum_error_n_components": selection_results["best_n_components"],
        "eigenvalue_gap_after_true_rank": float(
            signal_eigenvalues_estimated[true_rank - 1]
            / signal_eigenvalues_estimated[true_rank]
        ),
        "subspace_overlap": float(subspace_overlap),
    }

    print(f"Response shape: {responses.shape}")
    print(f"True signal rank: {true_rank} of {n_voxels} observed dimensions")
    print("True signal eigenvalues:", signal_eigenvalues)
    selected_n_components = selection_results["selected_n_components"]
    # (n_voxels,) -> (selected_n_components,)
    retained_signal_eigenvalues = signal_eigenvalues_estimated[
        :selected_n_components
    ]
    print(
        f"Estimated retained signal eigenvalues ({selected_n_components}):",
        retained_signal_eigenvalues,
    )
    print("Validation metrics:", metrics)

    assert subspace_overlap > 0.9, "GSN did not recover the true signal subspace."
    assert selection_results["best_n_components"] == true_rank, (
        "Reconstruction error was not minimized at the true signal rank."
    )
    assert selection_results["selected_n_components"] == true_rank, (
        "Independent-repeat reconstruction did not recover the true signal rank."
    )

    return {
        "responses": responses,
        "signal": signal,
        "noise": noise,
        "true_signal_basis": signal_basis,
        "true_noise_basis": noise_basis,
        "true_signal_eigenvalues": signal_eigenvalues,
        "true_noise_eigenvalues": noise_eigenvalues,
        "true_signal_covariance": true_signal_covariance,
        "true_noise_covariance": true_noise_covariance,
        "gsn_results": gsn_results,
        "pca_results": pca_results,
        "selection_results": selection_results,
        "metrics": metrics,
    }


if __name__ == "__main__":
    run_gsn_pca_smoke()
