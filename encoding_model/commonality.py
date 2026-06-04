"""Shared commonality analysis utilities."""

import numpy as np

from encoding_model.regression_metrics import (
    compute_lasso_cv_r2,
    compute_ols_cv_r2,
    compute_r2,
    compute_ridge_cv_r2,
)


def _ensure_2d(features):
    """Raise value error if features are not 2D:
    (n_samples, n_features). If features are 1D, raise an error with instructions to reshape."""
    features = np.asarray(features)
    if features.ndim == 1:
        raise ValueError("Features must be 2D. If you have a single feature, reshape it to (-1, 1).")
    return features


def _score_only(score_result):
    if isinstance(score_result, tuple):
        return score_result[1]
    return score_result


def commonality_analysis(
    features_A,
    features_B,
    target,
    method="standard",
    alphas=None,
    scale_by_target_variance=False,
    **_ignored_kwargs,
):
    """
    Decompose predictive power into unique, common, and unexplained components.

    Args:
        features_A (np.ndarray): First feature matrix, reported as X1.
        features_B (np.ndarray): Second feature matrix, reported as X2.
        target (np.ndarray): Target variable or target matrix.
        method (str): One of 'standard', 'ols_cv', 'ridge_cv', or 'lasso_cv'.
        alphas (array-like, optional): Ridge alpha values for method='ridge_cv'.
        scale_by_target_variance (bool): If True, return variance-scaled
            components. R2 values are never scaled.
        **_ignored_kwargs: Backwards compatibility for older callers that pass
            unused metadata such as snr.

    Returns:
        dict: R2 values and commonality components, without regression betas.
    """
    features_X1 = _ensure_2d(features_X1)
    features_X2 = _ensure_2d(features_X2)
    target = np.asarray(target)
    features_AB = np.hstack([features_X1, features_X2])

    if method == "standard":
        compute_r2_fn = compute_r2
    elif method == "ols_cv":
        compute_r2_fn = compute_ols_cv_r2
    elif method == "ridge_cv":
        compute_r2_fn = lambda X, y: compute_ridge_cv_r2(X, y, alphas)
    elif method == "lasso_cv":
        compute_r2_fn = compute_lasso_cv_r2
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'standard', 'ols_cv', 'ridge_cv', or 'lasso_cv'."
        )

    r2_X1 = _score_only(compute_r2_fn(features_A, target))
    r2_X2 = _score_only(compute_r2_fn(features_B, target))
    r2_X12 = _score_only(compute_r2_fn(features_AB, target))

    scale = 1.0
    if scale_by_target_variance:
        n = len(target)
        tss = np.sum((target - target.mean()) ** 2)
        scale = tss / (n - 1)

    unique_X1 = (r2_X12 - r2_X2) * scale
    unique_X2 = (r2_X12 - r2_X1) * scale
    common_X12 = (r2_X1 + r2_X2 - r2_X12) * scale
    unexplained = (1 - r2_X12) * scale

    return {
        "R²_X1": r2_X1,
        "R²_X2": r2_X2,
        "R²_X12": r2_X12,
        "unique_X1": unique_X1,
        "unique_X2": unique_X2,
        "common": common_X12,
        "unexplained": unexplained,
    }
