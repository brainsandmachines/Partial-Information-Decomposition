"""Numerical eigenvector PCA cross-validation and loading estimators."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable
import torch 
import numpy as np

from simulations.pca.rank.eigenvector_pca_wrapper import (
    build_checkpoint_metadata,
    load_eigenvector_pca_checkpoint,
    write_eigenvector_pca_checkpoint,
)

SKLEARN_IMPORT_ERROR = None
try:
    from sklearn.decomposition import PCA
except Exception as error:
    PCA = None
    SKLEARN_IMPORT_ERROR = error

TORCH_IMPORT_ERROR = None
try:
    import torch
except Exception as error:
    torch = None
    TORCH_IMPORT_ERROR = error

RANK_SIMULATION_IMPORT_ERROR = None
try:
    from simulations.pca.rank.pca_simulation import generate_rank_simulation_data
except Exception as error:
    generate_rank_simulation_data = None
    RANK_SIMULATION_IMPORT_ERROR = error

ROWWISE_PCA_IMPORT_ERROR = None
try:
    from simulations.pca.rank.rowwise_pca import rowwise_loo_pca_variance_threshold
except Exception as error:
    rowwise_loo_pca_variance_threshold = None
    ROWWISE_PCA_IMPORT_ERROR = error

PCAFitFunction = Callable[[np.ndarray, int], np.ndarray]
COMPONENT_CHUNK_SIZE = 256


@dataclass
class EigenvectorPCACVResult:
    selected_n_components: int
    press: np.ndarray
    msep: np.ndarray
    n_samples: int
    n_features: int
    time: float
    max_components: int


def fit_pca_loadings_svd(
    X_train: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Fit PCA loadings with SVD.

    Input: X_train is np.ndarray, or torch.Tensor for the GPU path, with shape
        (n_train_samples, n_features); n_components is int.
    Output: loading matrix of the same type as X_train with shape
        (n_features, n_components).
    """
    if torch is not None and isinstance(X_train, torch.Tensor):
        # On CUDA, the QR-based gesvd driver is exact and faster than the
        # default Jacobi driver for wide float64 matrices.
        driver = "gesvd" if X_train.is_cuda else None
        # (n_train_samples, n_features) -> Vt: (min(n_train_samples, n_features), n_features)
        _, _, Vt = torch.linalg.svd(X_train, full_matrices=False, driver=driver)
    else:
        # (n_train_samples, n_features) -> Vt: (min(n_train_samples, n_features), n_features)
        _, _, Vt = np.linalg.svd(X_train, full_matrices=False)
    # (min(n_train_samples, n_features), n_features) -> (n_features, n_components)
    return Vt[:n_components, :].T


def _to_numpy(array: "np.ndarray | torch.Tensor") -> np.ndarray:
    """Copy a NumPy array or torch tensor to a NumPy array on the CPU.

    Input: array is np.ndarray or torch.Tensor of any shape.
    Output: np.ndarray with the same shape and values.
    """
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _check_loadings(
    P: "np.ndarray | torch.Tensor",
    n_features: int,
    n_components: int,
) -> None:
    """Check the type and shape of a loading matrix returned by pca_fit_fn.

    Input: P is the loading matrix; n_features and n_components are int, the
        expected shape (n_features, n_components).
    Output: None. Raises TypeError or ValueError when P is invalid.
    """
    is_tensor = torch is not None and isinstance(P, torch.Tensor)
    if not (isinstance(P, np.ndarray) or is_tensor):
        raise TypeError("pca_fit_fn must return a NumPy array or torch tensor.")
    if tuple(P.shape) != (n_features, n_components):
        raise ValueError(
            f"pca_fit_fn returned shape {tuple(P.shape)}, "
            f"but expected {(n_features, n_components)}."
        )


def _is_orthonormal(P: "np.ndarray | torch.Tensor", tolerance: float) -> bool:
    """Check whether the columns of a loading matrix are orthonormal.

    Input: P is np.ndarray or torch.Tensor with shape (n_features, n_components);
        tolerance is float absolute tolerance on P.T @ P versus the identity.
    Output: bool, True when P.T @ P matches the identity.
    """
    gram = P.T @ P  # (n_features, n_components) -> (n_components, n_components)
    if torch is not None and isinstance(P, torch.Tensor):
        identity = torch.eye(P.shape[1], dtype=P.dtype, device=P.device)
        return bool(torch.allclose(gram, identity, rtol=1e-7, atol=tolerance))
    return bool(np.allclose(gram, np.eye(P.shape[1]), rtol=1e-7, atol=tolerance))


def _direct_loo_press(x_test: np.ndarray, P: np.ndarray, eps: float) -> float:
    """Calculate one held-out sample's PRESS by predicting each feature directly.

    Input: x_test is np.ndarray with shape (n_features,), centered and scaled
        with training statistics; P is np.ndarray loading matrix with shape
        (n_features, n_components); eps is float pseudoinverse cutoff.
    Output: float sum of squared errors, where feature j is predicted from the
        other features through a pseudoinverse fit of the scores on P.
    """
    press = 0.0
    for j in range(P.shape[0]):
        x_i_minus_j = np.delete(x_test, j)  # (n_features,) -> (n_features - 1,)
        P_minus_j = np.delete(P, j, axis=0)  # (n_features, n_components) -> (n_features - 1, n_components)
        p_j = P[j, :]  # (n_features, n_components) -> (n_components,)
        gram = P_minus_j.T @ P_minus_j  # (n_features - 1, n_components) -> (n_components, n_components)
        t_hat = x_i_minus_j @ P_minus_j @ np.linalg.pinv(gram, rcond=eps)  # (n_features - 1,) -> (n_components,)
        error = x_test[j] - t_hat @ p_j
        press += error ** 2
    return float(press)


def _eigenvector_pca_cv_sample_press(
    X: "np.ndarray | torch.Tensor",
    sample_index: int,
    max_components: int,
    pca_fit_fn: PCAFitFunction,
    center: bool,
    scale: bool,
    include_zero_components: bool,
    method_pca: str | None,
    eps: float,
) -> np.ndarray:
    """Calculate one held-out sample's PRESS contribution.

    Input: X is np.ndarray, or torch.Tensor for the GPU path, with shape
        (n_samples, n_features); sample_index and max_components are int;
        pca_fit_fn is PCAFitFunction; center, scale, and
        include_zero_components are bool; method_pca is str or None; eps is float.
    Output: np.ndarray of shape (max_components + 1,). With method_pca="SVD"
        and orthonormal loadings, all component counts come from one SVD
        through running residuals and leverages. Otherwise each count uses the
        leverage formula, falling back to the direct per-feature
        pseudoinverse calculation for non-orthonormal loadings.
    """
    n_features = X.shape[1]
    sample_press = np.zeros(max_components + 1, dtype=float)
    tolerance = max(1e-8, eps)

    # 1. Hold out one sample.
    if torch is not None and isinstance(X, torch.Tensor):
        X_train = torch.cat((X[:sample_index], X[sample_index + 1:]))  # (n_samples, n_features) -> (n_samples - 1, n_features)
    else:
        X_train = np.delete(X, sample_index, axis=0)  # (n_samples, n_features) -> (n_samples - 1, n_features)
    x_test = X[sample_index]  # (n_samples, n_features) -> (n_features,)

    # 2. Center and scale with training statistics only.
    if center:
        train_mean = X_train.mean(0)
        X_train = X_train - train_mean
        x_test = x_test - train_mean

    if scale:
        if torch is not None and isinstance(X_train, torch.Tensor):
            train_std = X_train.std(0, correction=1)
        else:
            train_std = X_train.std(axis=0, ddof=1)
        train_std[train_std < eps] = 1.0
        X_train = X_train / train_std
        x_test = x_test / train_std

    sample_press[0] = float((x_test ** 2).sum()) if include_zero_components else np.nan
    use_full_svd = method_pca == "SVD"
    if use_full_svd:
        P_full = pca_fit_fn(X_train, max_components)
        _check_loadings(P_full, n_features, max_components)

        # 3. Fast path. SVD loadings are nested and orthonormal, so the
        #    reconstruction residual r and leverage h for f components are
        #    running sums over the first f loadings, and the error of
        #    predicting feature j from the others is r_j / (1 - h_j).
        if _is_orthonormal(P_full, tolerance):
            scores = x_test @ P_full  # (n_features,) -> (max_components,)
            residual = x_test  # (n_features,)
            leverage = x_test * 0.0  # (n_features,)
            for start in range(0, max_components, COMPONENT_CHUNK_SIZE):
                stop = min(start + COMPONENT_CHUNK_SIZE, max_components)
                P_chunk = P_full[:, start:stop]  # (n_features, max_components) -> (n_features, chunk)
                residuals = residual[:, None] - (scores[None, start:stop] * P_chunk).cumsum(1)  # (n_features,) -> (n_features, chunk)
                leverages = leverage[:, None] + (P_chunk * P_chunk).cumsum(1)  # (n_features,) -> (n_features, chunk)
                denominators = 1.0 - leverages
                with np.errstate(divide="ignore", invalid="ignore"):
                    sample_press[start + 1:stop + 1] = _to_numpy(
                        ((residuals / denominators) ** 2).sum(0)  # (n_features, chunk) -> (chunk,)
                    )
                is_stable = _to_numpy((abs(denominators) > tolerance).all(0))  # (n_features, chunk) -> (chunk,)
                for f in np.flatnonzero(~is_stable) + start + 1:
                    sample_press[f] = _direct_loo_press(
                        _to_numpy(x_test), _to_numpy(P_full[:, :f]), eps
                    )
                residual = residuals[:, -1]  # (n_features, chunk) -> (n_features,)
                leverage = leverages[:, -1]  # (n_features, chunk) -> (n_features,)
            return sample_press

    # 4. General path: refit for each component count, or slice
    #    non-orthonormal loadings.
    x_test = _to_numpy(x_test)
    for f in range(1, max_components + 1):
        if f % 10 == 0:
            print(f"Processing component {f} of {max_components}...✅")
        P = P_full[:, :f] if use_full_svd else pca_fit_fn(X_train, f)
        _check_loadings(P, n_features, f)
        P = _to_numpy(P)

        if _is_orthonormal(P, tolerance):
            raw_error = x_test - (x_test @ P) @ P.T  # (n_features,) -> (n_features,)
            leverage = np.sum(P * P, axis=1)  # (n_features, f) -> (n_features,)
            denominator = 1.0 - leverage
            if np.all(np.abs(denominator) > tolerance):
                loo_error = raw_error / denominator
                sample_press[f] = np.sum(loo_error ** 2)
                continue

        sample_press[f] = _direct_loo_press(x_test, P, eps)

    return sample_press


def eigenvector_pca_cv(
    X: np.ndarray,
    max_components: int | None = None,
    pca_fit_fn: PCAFitFunction | None = None,
    center: bool = True,
    scale: bool = False,
    include_zero_components: bool = True,
    method_pca: str | None = None,
    eps: float = 1e-12,
    checkpoint_csv_path: str | Path | None = None,
    device: str | None = None,
) -> EigenvectorPCACVResult:
    """Run eigenvector cross-validation for PCA component selection.

    Input: X is np.ndarray with shape (n_samples, n_features); max_components
        is int or None; pca_fit_fn is PCAFitFunction or None; center, scale, and
        include_zero_components are bool; method_pca is str or None; eps is
        float; checkpoint_csv_path is str, Path, or None for row-level resume;
        device is str or None. None runs in NumPy. A torch device such as
        "cuda" runs the SVD and PRESS in float64 torch on that device; it
        requires method_pca="SVD" and pca_fit_fn=fit_pca_loadings_svd. The
        device is not part of the checkpoint metadata, so a NumPy checkpoint
        can be resumed on a GPU.
    Output: EigenvectorPCACVResult with selected count, PRESS, MSEP, dimensions,
        time placeholder, and max_components.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

    n_samples, n_features = X.shape
    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")
    if n_features < 2:
        raise ValueError("Need at least 2 features.")

    max_allowed = min(n_samples - 2, n_features - 1)
    if max_components is None:
        max_components = max_allowed
    if max_components < 1:
        raise ValueError("max_components must be at least 1.")
    if max_components > max_allowed:
        raise ValueError(
            f"max_components={max_components} is too large. "
            f"Use max_components <= {max_allowed}."
        )

    if pca_fit_fn is None:
        pca_fit_fn = fit_pca_loadings_svd
        print("Using default PCA fitting function (SVD)!!!")

    X_backend = X
    if device is not None:
        if torch is None:
            raise ImportError(
                "torch is required when device is set."
            ) from TORCH_IMPORT_ERROR
        if method_pca != "SVD" or pca_fit_fn is not fit_pca_loadings_svd:
            raise ValueError(
                "device requires method_pca='SVD' and "
                "pca_fit_fn=fit_pca_loadings_svd."
            )
        X_backend = torch.as_tensor(X, dtype=torch.float64, device=device)

    completed_press: dict[int, np.ndarray] = {}

    if checkpoint_csv_path is not None:
        metadata = build_checkpoint_metadata(
            X,
            max_components,
            pca_fit_fn,
            center,
            scale,
            include_zero_components,
            method_pca,
            eps,
        )
        completed_press = load_eigenvector_pca_checkpoint(
            checkpoint_csv_path,
            metadata,
            max_components,
        )

    press = np.zeros(max_components + 1, dtype=float)
    for sample_press in completed_press.values():
        press += sample_press
    start = time.time()
    for i in range(n_samples):
        print(f"Processing sample {i + 1} of {n_samples}...")
        if i in completed_press:
            print(f"Skipping sample {i} (already completed)✅")
            continue

        sample_press = _eigenvector_pca_cv_sample_press(
            X_backend,
            i,
            max_components,
            pca_fit_fn,
            center,
            scale,
            include_zero_components,
            method_pca,
            eps,
        )
        press += sample_press
        completed_press[i] = sample_press

        if checkpoint_csv_path is not None:
            write_eigenvector_pca_checkpoint(
                checkpoint_csv_path,
                metadata,
                completed_press,
                max_components,
            )

            end = time.time()
            elapsed = end - start
            print("Checkpoint saved. Elapsed time: {:.2f} seconds.".format(elapsed))

    msep = press / (n_samples * n_features)
    selected_n_components = int(
        np.argmin(msep) if include_zero_components else np.argmin(msep[1:]) + 1
    )

    return EigenvectorPCACVResult(
        selected_n_components=selected_n_components,
        press=press,
        msep=msep,
        n_samples=n_samples,
        n_features=n_features,
        max_components=max_components,
        time=time.time() - start,
    )


def fit_pca_loadings_sklearn(
    X_train: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Fit PCA loadings with sklearn.

    Input: X_train is np.ndarray with shape (n_train_samples, n_features);
        n_components is int.
    Output: np.ndarray with shape (n_features, n_components).
    """
    if PCA is None:
        raise ImportError(
            "sklearn.decomposition.PCA is required for fit_pca_loadings_sklearn."
        ) from SKLEARN_IMPORT_ERROR

    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(X_train)
    return model.components_.T


def regular_PCA(X: np.ndarray, variance_threshold: float) -> np.ndarray:
    """Fit sklearn PCA using a variance threshold.

    Input: X is np.ndarray with shape (n_samples, n_features);
        variance_threshold is float.
    Output: np.ndarray with shape (n_features, n_components).
    """
    if PCA is None:
        raise ImportError(
            "sklearn.decomposition.PCA is required for regular_PCA."
        ) from SKLEARN_IMPORT_ERROR

    model = PCA(n_components=variance_threshold, svd_solver="full")
    model.fit(X)
    return model.components_.T


if __name__ == "__main__":
    if generate_rank_simulation_data is None:
        raise ImportError(
            "generate_rank_simulation_data is required for the example run."
        ) from RANK_SIMULATION_IMPORT_ERROR
    if rowwise_loo_pca_variance_threshold is None:
        raise ImportError(
            "rowwise_loo_pca_variance_threshold is required for the example run."
        ) from ROWWISE_PCA_IMPORT_ERROR

    # Example usage
    n_samples = 50
    n_features = 4
    rank = 2
    loading_corr = 0.9
    noise_std = 0.5
    random_state = 42

    data = generate_rank_simulation_data(
        n_samples=n_samples,
        n_features=n_features,
        rank=rank,
        loading_corr=loading_corr,
        noise_std=noise_std,
        random_state=random_state,
    )

    X = data["X"]
    result = eigenvector_pca_cv(
        X,
        max_components=3,
        pca_fit_fn=fit_pca_loadings_svd,
        center=True,
        scale=True,
    )

    print("Selected components:", result.selected_n_components)

    variance_threshold = 0.99
    print(f"\nRunning regular PCA with variance threshold of {variance_threshold}")
    P_regular = rowwise_loo_pca_variance_threshold(X, variance_threshold=variance_threshold)
    print("Number of Pcs chosen:", P_regular.shape[1])
