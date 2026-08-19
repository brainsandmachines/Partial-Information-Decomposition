"""Optional covariance-shrinkage operations used by resampling corrections."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance


def ledoit_wolf_cov(samples: np.ndarray) -> np.ndarray:
    """Estimate a covariance matrix with Ledoit–Wolf shrinkage.

    Inputs:
        samples: np.ndarray shaped ``(n_samples, n_features)``.

    Outputs:
        np.ndarray covariance shaped ``(n_features, n_features)``.
    """

    return LedoitWolf().fit(samples).covariance_  # (N, D) -> (D, D)


def oracle_shrinkage_cov(
    samples: np.ndarray,
    assume_centered: bool = False,
    return_shrinkage: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Estimate covariance with Oracle Approximating Shrinkage.

    Inputs:
        samples: np.ndarray shaped ``(n_samples, n_features)``.
        assume_centered: bool indicating whether samples are already centered.
        return_shrinkage: bool selecting covariance-only or covariance/coefficient output.

    Outputs:
        np.ndarray covariance shaped ``(n_features, n_features)`` or a tuple
        containing that covariance and its float shrinkage coefficient.
    """

    estimator = OAS(assume_centered=assume_centered).fit(samples)
    if return_shrinkage:
        return estimator.covariance_, float(estimator.shrinkage_)
    return estimator.covariance_  # (N, D) -> (D, D)


def shrunk_cov(samples: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Estimate covariance with a fixed shrinkage coefficient.

    Inputs:
        samples: np.ndarray shaped ``(n_samples, n_features)``.
        alpha: float shrinkage coefficient in ``[0, 1]``.

    Outputs:
        np.ndarray covariance shaped ``(n_features, n_features)``.
    """

    return ShrunkCovariance(shrinkage=alpha).fit(samples).covariance_  # (N, D) -> (D, D)


def on_covariance(config: dict, covariance: torch.Tensor) -> dict[str, torch.Tensor]:
    """Apply the configured shrinkage method to one or more covariance matrices.

    Inputs:
        config: dict containing ``on_covariance`` and optional ``alpha``.
        covariance: torch.Tensor shaped ``(D, D)`` or ``(B, D, D)``.

    Outputs:
        dict with key ``cov`` containing the unchanged covariance or a batched
        torch.Tensor shaped ``(B, D, D)`` after shrinkage.
    """

    method = config["on_covariance"]
    if method == "False":
        return {"cov": covariance}
    if method not in {"ledoit_wolf", "oas", "shrunk_cov"}:
        raise ValueError(f"Unsupported covariance transformation: {method!r}.")

    covariance_batch = covariance
    if covariance_batch.ndim == 2:
        covariance_batch = covariance_batch.unsqueeze(0)  # (D, D) -> (1, D, D)

    transformed = []
    for matrix in covariance_batch:
        if not torch.isfinite(matrix).all():
            raise ValueError("Covariance matrix contains NaN or Inf values.")
        matrix_numpy = matrix.detach().cpu().numpy()  # torch (D, D) -> NumPy (D, D)
        if method == "ledoit_wolf":
            estimate = ledoit_wolf_cov(matrix_numpy)
        elif method == "oas":
            estimate = oracle_shrinkage_cov(matrix_numpy)
        else:
            estimate = shrunk_cov(matrix_numpy, float(config["alpha"]))
        transformed.append(
            torch.as_tensor(
                estimate,
                device=covariance_batch.device,
                dtype=covariance_batch.dtype,
            )
        )
    result = torch.stack(transformed, dim=0)  # B tensors (D, D) -> (B, D, D)
    if result.shape != covariance_batch.shape:
        raise ValueError(
            f"Expected output shape {covariance_batch.shape}, got {result.shape}."
        )
    return {"cov": result}
