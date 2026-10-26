"""Repeated synthetic neural responses with a known low-rank stimulus signal."""

from typing import Any

import numpy as np


def generate_low_rank_repeated_responses(
    n_images: int = 200,
    n_voxels: int = 50,
    true_rank: int = 5,
    n_repeats: int = 4,
    seed: int = 0,
    noise_scale: float = 1.0,
) -> dict[str, Any]:
    """Generate repeated responses whose stimulus covariance has known rank.

    The stimulus-related signal is identical across repeats. Each repeat adds
    independent Gaussian noise with a full-rank covariance. The random draws
    are made in a fixed order, so equal inputs give identical arrays.

    Inputs:
        n_images: int, number of independently generated stimulus conditions.
        n_voxels: int, observed neural dimensions containing the latent signal.
        true_rank: int, exact rank of the stimulus-related signal covariance.
        n_repeats: int, independent noisy measurements of each stimulus.
        seed: int, NumPy random seed for reproducibility.
        noise_scale: float, positive multiplier of every noise-covariance
            eigenvalue. The default 1.0 reproduces the original simulation.

    Output:
        data: dict[str, Any] containing ``responses`` with shape
            (n_images, n_voxels, n_repeats), the noiseless ``signal`` with
            shape (n_images, n_voxels), the ``noise`` with shape
            (n_images, n_voxels, n_repeats), the true signal basis
            (n_voxels, true_rank) and noise basis (n_voxels, n_voxels), their
            eigenvalues, and the true signal and noise covariances, each with
            shape (n_voxels, n_voxels).
    """

    if not 1 <= true_rank < n_voxels:
        raise ValueError("true_rank must be between 1 and n_voxels - 1.")
    if n_images <= true_rank:
        raise ValueError("n_images must be greater than true_rank.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if not noise_scale > 0:
        raise ValueError("noise_scale must be positive.")

    rng = np.random.default_rng(seed)
    signal_eigenvalues = np.geomspace(9.0, 1.0, true_rank)
    noise_eigenvalues = noise_scale * np.linspace(2.0, 0.2, n_voxels)

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
    }
