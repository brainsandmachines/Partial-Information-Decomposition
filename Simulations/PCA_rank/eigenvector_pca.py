"""Numerical eigenvector PCA cross-validation and loading estimators."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import numpy as np

from Simulations.PCA_rank.eigenvector_pca_wrapper import (
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

RANK_SIMULATION_IMPORT_ERROR = None
try:
    from Simulations.PCA_rank.pca_simulation import generate_rank_simulation_data
except Exception as error:
    generate_rank_simulation_data = None
    RANK_SIMULATION_IMPORT_ERROR = error

ROWWISE_PCA_IMPORT_ERROR = None
try:
    from Simulations.PCA_rank.rowwise_PCA import rowwise_loo_pca_variance_threshold
except Exception as error:
    rowwise_loo_pca_variance_threshold = None
    ROWWISE_PCA_IMPORT_ERROR = error

PCAFitFunction = Callable[[np.ndarray, int], np.ndarray]


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

    Input: X_train is np.ndarray with shape (n_train_samples, n_features);
        n_components is int.
    Output: np.ndarray loading matrix with shape (n_features, n_components).
    """
    _, _, Vt = np.linalg.svd(X_train, full_matrices=False)
    return Vt[:n_components, :].T


def _eigenvector_pca_cv_sample_press(
    X: np.ndarray,
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

    Input: X is np.ndarray; sample_index and max_components are int;
        pca_fit_fn is PCAFitFunction; center, scale, and
        include_zero_components are bool; method_pca is str or None; eps is float.
    Output: np.ndarray of shape (max_components + 1,). Orthonormal loadings use
        the vectorized leverage formula; non-orthonormal loadings fall back to
        the direct per-feature pseudoinverse calculation.
    """
    n_samples, n_features = X.shape
    sample_press = np.zeros(max_components + 1, dtype=float)
    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[sample_index] = False

    X_train = X[train_mask, :].copy()
    x_test = X[sample_index, :].copy()

    if center:
        train_mean = X_train.mean(axis=0)
        X_train = X_train - train_mean
        x_test = x_test - train_mean

    if scale:
        train_std = X_train.std(axis=0, ddof=1)
        train_std[train_std < eps] = 1.0
        X_train = X_train / train_std
        x_test = x_test / train_std

    sample_press[0] = np.sum(x_test ** 2) if include_zero_components else np.nan
    use_full_svd = method_pca == "SVD"
    if use_full_svd:
        P_full = pca_fit_fn(X_train, max_components)

    for f in range(1, max_components + 1):
        if f % 10 == 0:
            print(f"Processing component {f} of {max_components}...✅")
        P = P_full[:, :f] if use_full_svd else pca_fit_fn(X_train, f)
        if not isinstance(P, np.ndarray):
            raise TypeError("pca_fit_fn must return a NumPy array.")
        if P.shape != (n_features, f):
            raise ValueError(
                f"pca_fit_fn returned shape {P.shape}, "
                f"but expected {(n_features, f)}."
            )

        tolerance = max(1e-8, eps)
        is_orthonormal = np.allclose(
            P.T @ P,
            np.eye(f),
            rtol=1e-7,
            atol=tolerance,
        )
        if is_orthonormal:
            raw_error = x_test - (x_test @ P) @ P.T
            leverage = np.sum(P * P, axis=1)
            denominator = 1.0 - leverage
            if np.all(np.abs(denominator) > tolerance):
                loo_error = raw_error / denominator
                sample_press[f] = np.sum(loo_error ** 2)
                continue

        for j in range(n_features):
            x_i_minus_j = np.delete(x_test, j)
            P_minus_j = np.delete(P, j, axis=0)
            p_j = P[j, :]
            gram = P_minus_j.T @ P_minus_j
            t_hat = x_i_minus_j @ P_minus_j @ np.linalg.pinv(gram, rcond=eps)
            error = x_test[j] - t_hat @ p_j
            sample_press[f] += error ** 2

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
) -> EigenvectorPCACVResult:
    """Run eigenvector cross-validation for PCA component selection.

    Input: X is np.ndarray with shape (n_samples, n_features); max_components
        is int or None; pca_fit_fn is PCAFitFunction or None; center, scale, and
        include_zero_components are bool; method_pca is str or None; eps is
        float; checkpoint_csv_path is str, Path, or None for row-level resume.
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
            X,
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
