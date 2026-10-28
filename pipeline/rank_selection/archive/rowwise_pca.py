from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut


@dataclass
class RowwiseLOOVariancePCAResult:
    X_reconstructed: np.ndarray
    squared_errors: np.ndarray
    row_mse: np.ndarray
    total_press: float
    total_msep: float
    time: float
    n_components_per_fold: list[int]
    full_data_n_components: int
    full_data_loadings: np.ndarray
    variance_threshold: float


def rowwise_loo_pca_variance_threshold(
    X: np.ndarray,
    variance_threshold: float = 0.99,
) -> RowwiseLOOVariancePCAResult:
    """
    Row-wise leave-one-out PCA where the number of PCs is chosen by
    explained variance threshold inside each fold.

    For each held-out row:
        1. Fit PCA on all other rows.
        2. Let sklearn choose n_components by variance_threshold.
        3. Reconstruct the held-out row.
        4. Store reconstruction error and chosen number of PCs.

    Input:
        X:
            Data matrix of shape (n_samples, n_features).

        variance_threshold:
            Explained variance threshold, for example 0.99.

    Output:
        n_components_per_fold:
            Number of PCs chosen in each leave-one-out fold.

        full_data_n_components:
            Number of PCs chosen when PCA is fit on the full X.

        full_data_loadings:
            PCA loading matrix from full-data PCA.
            Shape: (n_features, full_data_n_components).
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D with shape (n_samples, n_features).")

    if not (0.0 < variance_threshold < 1.0):
        raise ValueError("variance_threshold must be between 0 and 1, e.g. 0.99.")

    n_samples, n_features = X.shape

    loo = LeaveOneOut()

    X_reconstructed = np.zeros_like(X)
    squared_errors = np.zeros_like(X)
    row_mse = np.zeros(n_samples)
    n_components_per_fold: list[int] = []

    for train_idx, test_idx in loo.split(X):
        X_train = X[train_idx]   # shape: (n_samples - 1, n_features)
        X_test = X[test_idx]     # shape: (1, n_features)

        model = PCA(
            n_components=variance_threshold,
            svd_solver="full",
        )

        model.fit(X_train)

        z_test = model.transform(X_test)
        X_test_reconstructed = model.inverse_transform(z_test)

        i = int(test_idx[0])

        X_reconstructed[i, :] = X_test_reconstructed[0]

        errors = X_test[0] - X_test_reconstructed[0]
        squared_errors[i, :] = errors**2
        row_mse[i] = np.mean(errors**2)

        n_components_per_fold.append(int(model.n_components_))

    total_press = float(np.sum(squared_errors))
    total_msep = float(np.mean(squared_errors))

    # Also fit PCA on the full data to get the final number of PCs
    # you would use for preprocessing.
    full_model = PCA(
        n_components=variance_threshold,
        svd_solver="full",
    )

    full_model.fit(X)

    full_data_n_components = int(full_model.n_components_)
    full_data_loadings = full_model.components_.T

    return RowwiseLOOVariancePCAResult(
        X_reconstructed=X_reconstructed,
        squared_errors=squared_errors,
        row_mse=row_mse,
        total_press=total_press,
        total_msep=total_msep,
        n_components_per_fold=n_components_per_fold,
        full_data_n_components=full_data_n_components,
        full_data_loadings=full_data_loadings,
        variance_threshold=variance_threshold,
        time=0.0
    )