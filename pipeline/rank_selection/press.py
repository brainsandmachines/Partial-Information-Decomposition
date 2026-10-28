"""Shared leave-one-voxel-out PRESS kernel for eigenvector cross-validation.

For a held-out response x (centered with training statistics) and a basis P
with orthonormal columns, predicting voxel j from the other voxels through the
first k basis vectors leaves the error r_j / (1 - h_j), where r = x - P P^T x
is the reconstruction residual and h_j = sum_k P_jk^2 the leverage. This is the
PRESS (leave-one-out) identity of least squares with design P (Bro et al.,
2008). ``direct_press`` computes the same error without the identity.
"""

from typing import Any

import numpy as np

TORCH_IMPORT_ERROR = None
try:
    import torch
except Exception as error:
    torch = None
    TORCH_IMPORT_ERROR = error


def to_numpy(array: Any) -> np.ndarray:
    """Copy a NumPy array or torch tensor to a NumPy array on the CPU.

    Input: array, np.ndarray or torch.Tensor of any shape.
    Output: np.ndarray with the same shape and values.
    """
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def is_orthonormal(P: Any, tolerance: float) -> bool:
    """Check whether the columns of a loading matrix are orthonormal.

    Inputs: P, np.ndarray or torch.Tensor (n_voxels, n_components); tolerance,
        float absolute tolerance on P^T P versus the identity.
    Output: bool.
    """
    gram = P.T @ P  # (n_voxels, n_components) -> (n_components, n_components)
    if torch is not None and isinstance(P, torch.Tensor):
        identity = torch.eye(P.shape[1], dtype=P.dtype, device=P.device)
        return bool(torch.allclose(gram, identity, rtol=1e-7, atol=tolerance))
    return bool(np.allclose(gram, np.eye(P.shape[1]), rtol=1e-7, atol=tolerance))


def eigenvector_press(
    heldout: Any,
    basis: Any,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-voxel-out PRESS for every component count 1..n_components.

    Inputs:
        heldout: np.ndarray or torch.Tensor (n_rows, n_voxels), held-out
            responses centered (and scaled, if used) with training statistics.
        basis: same backend, (n_voxels, n_components) with orthonormal columns,
            fitted without the held-out rows.
        tolerance: float; a count is unstable when some voxel's 1 - h_j is
            within tolerance of 0.
    Output:
        press: np.ndarray (n_components,), PRESS for k = 1..n_components summed
            over rows and voxels (meaningless where unstable).
        stable: np.ndarray of bool (n_components,).
    """
    n_components = basis.shape[1]
    # (n_rows, n_voxels) @ (n_voxels, n_components) -> (n_rows, n_components)
    scores = heldout @ basis
    residual = heldout * 1.0  # copy; (n_rows, n_voxels)
    leverage = basis[:, 0] * 0.0  # (n_voxels,)
    press, stable = [], []
    # Unstable counts divide by ~0; they are flagged, so silence NumPy's warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        for component_index in range(n_components):
            # (n_rows, 1) * (1, n_voxels) -> (n_rows, n_voxels)
            residual -= scores[:, component_index, None] * basis[None, :, component_index]
            leverage += basis[:, component_index] ** 2
            denominator = 1.0 - leverage  # (n_voxels,)
            stable.append((abs(denominator) > tolerance).all())
            press.append(((residual / denominator[None, :]) ** 2).sum())
    if torch is not None and isinstance(heldout, torch.Tensor):
        return to_numpy(torch.stack(press)), to_numpy(torch.stack(stable))
    return np.asarray(press, dtype=float), np.asarray(stable, dtype=bool)


def direct_press(x_test: np.ndarray, P: np.ndarray, eps: float) -> float:
    """PRESS of one held-out row by predicting each voxel from the others directly.

    Inputs: x_test, np.ndarray (n_voxels,), centered with training statistics;
        P, np.ndarray (n_voxels, n_components), any loadings (orthonormality not
        required); eps, float pseudoinverse cutoff.
    Output: float, sum over voxels of squared errors, where voxel j is
        predicted from the other voxels through a pseudoinverse fit of the
        scores on P with row j removed.
    """
    press = 0.0
    for j in range(P.shape[0]):
        x_i_minus_j = np.delete(x_test, j)  # (n_voxels,) -> (n_voxels - 1,)
        P_minus_j = np.delete(P, j, axis=0)  # (n_voxels, n_components) -> (n_voxels - 1, n_components)
        p_j = P[j, :]  # (n_voxels, n_components) -> (n_components,)
        gram = P_minus_j.T @ P_minus_j  # (n_voxels - 1, n_components) -> (n_components, n_components)
        t_hat = x_i_minus_j @ P_minus_j @ np.linalg.pinv(gram, rcond=eps)  # (n_voxels - 1,) -> (n_components,)
        error = x_test[j] - t_hat @ p_j
        press += error ** 2
    return float(press)
