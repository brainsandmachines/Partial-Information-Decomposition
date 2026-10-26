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
from pipeline.testing.synthetic_repeated_responses import (
    generate_low_rank_repeated_responses,
)


def run_gsn_pca_smoke(
    n_images: int = 200,
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

    synthetic_data = generate_low_rank_repeated_responses(
        n_images=n_images,
        n_voxels=n_voxels,
        true_rank=true_rank,
        n_repeats=n_repeats,
        seed=seed,
        noise_scale=0.3,
    )
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2 for GSN.")

    responses = synthetic_data["responses"]
    signal_basis = synthetic_data["true_signal_basis"]
    signal_eigenvalues = synthetic_data["true_signal_eigenvalues"]

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
        **synthetic_data,
        "gsn_results": gsn_results,
        "pca_results": pca_results,
        "selection_results": selection_results,
        "metrics": metrics,
    }


if __name__ == "__main__":
    run_gsn_pca_smoke()
