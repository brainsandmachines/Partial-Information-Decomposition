from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from Simulations.PCA_rank.rowwise_PCA import rowwise_loo_pca_variance_threshold
from pca_simulation import generate_rank_simulation_data
import numpy as np
from sklearn.decomposition import PCA


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
    """
    Default PCA fitting function using SVD.

    Input:
        X_train: array of shape (n_train_samples, n_features)
        n_components: number of PCA components

    Output:
        P: loading matrix of shape (n_features, n_components)
    """

    _, _, Vt = np.linalg.svd(X_train, full_matrices=False)

    P = Vt[:n_components, :].T

    return P


def eigenvector_pca_cv(
    X: np.ndarray,
    max_components: int | None = None,
    pca_fit_fn: PCAFitFunction | None = None,
    center: bool = True,
    scale: bool = False,
    include_zero_components: bool = True,
    method_pca: str = None,
    eps: float = 1e-12,
) -> EigenvectorPCACVResult:
    """
    Eigenvector cross-validation for PCA component selection.

    For each candidate number of components f:
        For each sample i:
            1. Remove row i from X.
            2. Fit PCA on X without row i.
            3. For each feature j:
                a. Remove x_ij from the held-out row.
                b. Estimate the PCA score from x_i,-j.
                c. Predict x_ij.
                d. Accumulate squared error.

    The PCA fitting function is injected through pca_fit_fn.

    Required signature:
        P = pca_fit_fn(X_train, n_components)

    where:
        X_train.shape == (n_train_samples, n_features)
        P.shape == (n_features, n_components)
    """
    assert method_pca is not None, "method_pca must be specified as 'SVD' or 'sklearn'."
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

    n_samples, n_features = X.shape

    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")

    if n_features < 2:
        raise ValueError("Need at least 2 features.")

    if max_components is None:
        max_components = min(n_samples - 2, n_features - 1)

    max_allowed = min(n_samples - 2, n_features - 1)

    if max_components < 1:
        raise ValueError("max_components must be at least 1.")

    if max_components > max_allowed:
        raise ValueError(
            f"max_components={max_components} is too large. "
            f"Use max_components <= {max_allowed}."
        )

    if pca_fit_fn is None:
        pca_fit_fn = fit_pca_loadings_svd
        print("Using default PCA fitting function (SVD).!!!")

    press = np.zeros(max_components + 1, dtype=float)

    for i in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False

        X_train = X[train_mask, :].copy()  # shape: (n_samples - 1, n_features)
        x_test = X[i, :].copy()            # shape: (n_features,)

        if center:
            train_mean = X_train.mean(axis=0)  # shape: (n_features,)
            X_train = X_train - train_mean
            x_test = x_test - train_mean

        if scale:
            train_std = X_train.std(axis=0, ddof=1)  # shape: (n_features,)
            train_std[train_std < eps] = 1.0
            X_train = X_train / train_std
            x_test = x_test / train_std

        # f = 0:
        # Predict each feature by the training mean.
        # After centering, this prediction is zero.
        if include_zero_components:
            press[0] += np.sum(x_test ** 2)
        else:
            press[0] = np.nan  # Ignore f=0 in the MSEP calculation.

        if method_pca == 'SVD':
            P_full = pca_fit_fn(X_train, max_components)  # shape: (n_features, max_components)
        for f in range(1, max_components + 1):

            if method_pca == 'SVD': 
                P = P_full[:, :f]  # shape: (n_features, f)
            else:
                P = pca_fit_fn(X_train, f)  # expected shape: (n_features, f)

            if not isinstance(P, np.ndarray):
                raise TypeError("pca_fit_fn must return a NumPy array.")

            expected_shape = (n_features, f)

            if P.shape != expected_shape:
                raise ValueError(
                    f"pca_fit_fn returned shape {P.shape}, "
                    f"but expected {expected_shape}."
                )
            
            for j in range(n_features):
                # Remove x_ij from the held-out row.
                x_i_minus_j = np.delete(x_test, j)      # shape: (n_features - 1,)

                # Remove loading row j.
                P_minus_j = np.delete(P, j, axis=0)     # shape: (n_features - 1, f)

                # Loading row for feature j.
                p_j = P[j, :]                           # shape: (f,)

                # Estimate score from all features except j:
                #
                # t_hat = x_i,-j P_-j (P_-j.T P_-j)^(-1)
                #
                gram = P_minus_j.T @ P_minus_j          # shape: (f, f)

                t_hat = (
                    x_i_minus_j
                    @ P_minus_j
                    @ np.linalg.pinv(gram, rcond=eps)
                )                                       # shape: (f,)

                # Predict x_ij from the estimated score.
                x_hat_ij = t_hat @ p_j                  # scalar

                error = x_test[j] - x_hat_ij
                press[f] += error ** 2

    msep = press / (n_samples * n_features)

    if include_zero_components:
        selected_n_components = int(np.argmin(msep))
    else:
        selected_n_components = int(np.argmin(msep[1:]) + 1)

    return EigenvectorPCACVResult(
        selected_n_components=selected_n_components,
        press=press,
        msep=msep,
        n_samples=n_samples,
        n_features=n_features,
        max_components=max_components,
        time = 0.0
    )





def fit_pca_loadings_sklearn(
    X_train: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """
    Fit PCA using sklearn.

    Input:
        X_train: shape (n_train_samples, n_features)
        n_components: number of components

    Output:
        P: shape (n_features, n_components)
    """

    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(X_train)

    P = model.components_.T

    return P

def regular_PCA(X: np.ndarray, variance_threshold: float) -> np.ndarray:
    """
    Fit PCA using sklearn.

    Input:
        X: shape (n_samples, n_features)
        variance_threshold: threshold for explained variance

    Output:
        P: shape (n_features, n_components)
    """

    model = PCA(n_components=variance_threshold, svd_solver="full")
    model.fit(X)

    P = model.components_.T

    return P


if __name__ == "__main__":
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