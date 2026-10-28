import torch
import numpy as np
import matplotlib.pyplot as plt

"""This file will contain controlled examples of disagreement between the 3 methods. 
Regular PCA, GSN-PCA, and Cross-Decomposition."""






from typing import Any

import numpy as np


def generate_low_rank_repeated_responses(
    n_images: int = 200,
    n_voxels: int = 50,
    true_rank: int = 5,
    n_repeats: int = 4,
    seed: int = 0,
    noise_scale: float = 0.02,
) -> dict[str, Any]:
    """Generate repeated neural responses with known signal rank.

    Each stimulus has a low-rank Gaussian signal embedded in a
    higher-dimensional voxel space.

    The stimulus-related signal is identical across repetitions.
    Each repetition adds independent Gaussian noise with a
    full-rank, anisotropic covariance.

    Unlike an orthogonalized-score construction, the latent
    stimulus variables are sampled independently.

    Parameters
    ----------
    n_images : int
        Number of independently generated stimulus conditions.
    n_voxels : int
        Number of observed neural dimensions.
    true_rank : int
        Exact rank of the population signal covariance.
    n_repeats : int
        Number of independent noisy measurements per stimulus.
    seed : int
        Random seed for reproducibility.
    noise_scale : float
        Positive multiplier applied to the noise eigenvalues.

    Returns
    -------
    dict[str, Any]
        responses : ndarray, shape (n_images, n_voxels, n_repeats)
            Observed neural responses.

        signal : ndarray, shape (n_images, n_voxels)
            Noiseless stimulus-related responses.

        noise : ndarray, shape (n_images, n_voxels, n_repeats)
            Independent measurement noise.

        true_signal_basis : ndarray, shape (n_voxels, true_rank)
            Orthonormal signal directions.

        true_noise_basis : ndarray, shape (n_voxels, n_voxels)
            Orthonormal noise directions.

        true_signal_eigenvalues : ndarray, shape (true_rank,)
            Nonzero population signal eigenvalues.

        true_noise_eigenvalues : ndarray, shape (n_voxels,)
            Population noise eigenvalues.

        true_signal_covariance : ndarray, shape (n_voxels, n_voxels)
            Population signal covariance.

        true_noise_covariance : ndarray, shape (n_voxels, n_voxels)
            Population noise covariance.
    """
    if not 1 <= true_rank < n_voxels:
        raise ValueError("true_rank must be between 1 and n_voxels - 1.")

    if n_images <= true_rank:
        raise ValueError("n_images must be greater than true_rank.")

    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2.")

    if not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("noise_scale must be positive and finite.")

    rng = np.random.default_rng(seed)

    signal_eigenvalues = np.geomspace(9.0,2.0,true_rank)
    noise_eigenvalues = noise_scale * np.linspace(2.0,0.2,n_voxels)

    signal_basis, _ = np.linalg.qr(rng.normal(size=(n_voxels, true_rank)))
    latent = rng.normal(size=(n_images, true_rank))
    scores = latent * np.sqrt(signal_eigenvalues)

    signal = scores @ signal_basis.T
    true_signal_covariance = (signal_basis * signal_eigenvalues) @ signal_basis.T

    noise_basis, _ = np.linalg.qr(rng.normal(size=(n_voxels, n_voxels)))
    true_noise_covariance = (noise_basis * noise_eigenvalues) @ noise_basis.T


    noise_cholesky = np.linalg.cholesky(true_noise_covariance)
    standard_noise = rng.normal(size=(n_images, n_repeats, n_voxels))

    noise = (standard_noise @ noise_cholesky.T).transpose(0, 2, 1) #(n_images, n_voxels, n_repeats)
    responses = signal[:, :, None] + noise #(n_images, n_voxels, n_repeats)

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



def LudiPQ_example():
    pass






def Zakum_example():
    pass




if __name__ == "__main__":
    pass
