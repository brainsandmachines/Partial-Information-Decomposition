"""Gaussian mutual-information calculation from a joint covariance matrix."""

from typing import Literal

import numpy as np
import torch

from pid.bias import mi_wishart_bias


def gaussian_mutual_information(
    covariance: torch.Tensor | np.ndarray,
    x_dim: int,
    *,
    n_samples: int | None = None,
    bias_correction: Literal["wishart"] | None = None,
) -> dict[str, float | str | None]:
    """Calculate Gaussian mutual information between two covariance blocks.

    Inputs:
        covariance: torch.Tensor or np.ndarray, positive-definite joint
            covariance ordered as ``[X, Y]`` with shape ``(D, D)``.
        x_dim: int, dimension of the leading ``X`` block; the remaining
            dimensions form ``Y``.
        n_samples: int or None, sample count used to estimate ``covariance``.
            Required only for Wishart correction.
        bias_correction: ``"wishart"`` or None. Wishart correction is valid
            only for an unbiased Gaussian sample covariance.

    Outputs:
        dict containing raw MI, estimated bias, and corrected MI in bits,
        together with the selected bias method. Corrected MI is not clipped.
    """

    covariance = torch.as_tensor(covariance, dtype=torch.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square two-dimensional matrix.")
    if not torch.isfinite(covariance).all():
        raise ValueError("covariance must contain only finite values.")
    if isinstance(x_dim, bool) or not isinstance(x_dim, (int, np.integer)):
        raise TypeError("x_dim must be an integer.")
    x_dim = int(x_dim)
    if not 0 < x_dim < covariance.shape[0]:
        raise ValueError("x_dim must split covariance into non-empty X and Y blocks.")
    if not torch.allclose(covariance, covariance.mT, rtol=1e-7, atol=1e-10):
        raise ValueError("covariance must be symmetric.")
    if bias_correction not in (None, "wishart"):
        raise ValueError("bias_correction must be None or 'wishart'.")

    y_dim = covariance.shape[0] - x_dim
    cov_x = covariance[:x_dim, :x_dim]  # (D, D) -> (x_dim, x_dim)
    cov_y = covariance[x_dim:, x_dim:]  # (D, D) -> (y_dim, y_dim)
    chol_x, info_x = torch.linalg.cholesky_ex(cov_x)
    chol_y, info_y = torch.linalg.cholesky_ex(cov_y)
    chol_joint, info_joint = torch.linalg.cholesky_ex(covariance)
    if any(int(info) != 0 for info in (info_x, info_y, info_joint)):
        min_eigenvalue = float(torch.linalg.eigvalsh(covariance).min())
        raise ValueError(
            "covariance and both marginal blocks must be positive definite; "
            f"minimum joint eigenvalue is {min_eigenvalue:.3e}."
        )

    logdet_x = 2.0 * torch.log(torch.diagonal(chol_x)).sum()
    logdet_y = 2.0 * torch.log(torch.diagonal(chol_y)).sum()
    logdet_joint = 2.0 * torch.log(torch.diagonal(chol_joint)).sum()
    raw_bits = float(0.5 * (logdet_x + logdet_y - logdet_joint) / np.log(2.0))
    bias_bits = 0.0
    if bias_correction == "wishart":
        if n_samples is None:
            raise ValueError("n_samples is required for Wishart bias correction.")
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
            raise TypeError("n_samples must be an integer.")
        n_samples = int(n_samples)
        if n_samples <= covariance.shape[0]:
            raise ValueError("Wishart correction requires n_samples > covariance dimension.")
        bias_bits = float(mi_wishart_bias([x_dim, y_dim], n_samples) / np.log(2.0))

    return {
        "raw_bits": raw_bits,
        "bias_bits": bias_bits,
        "corrected_bits": raw_bits - bias_bits,
        "bias_method": bias_correction,
    }
