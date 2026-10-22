"""Commonality analysis for comparison with BROJA PID."""

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV


def commonality_analysis(
    features_X1,
    features_X2,
    target,
    method="standard",
    alphas=None,
    scale_by_target_variance=False,
):
    """Decompose predictive power into unique, common, and unexplained terms.

    Inputs:
        features_X1: array-like with shape (n_samples, n_features_1), first source.
        features_X2: array-like with shape (n_samples, n_features_2), second source.
        target: array-like with shape (n_samples, n_targets), prediction target.
        method: str, either ``standard`` for in-sample OLS or ``ridge_cv`` for
            generalized-cross-validation RidgeCV scores.
        alphas: array-like or None, ridge penalties used by ``ridge_cv``.
        scale_by_target_variance: bool, whether to scale decomposition terms by
            the sample variance of the target. R-squared values remain unscaled.

    Outputs:
        dict containing the three R-squared values, both unique contributions,
        the common contribution, and unexplained variance.
    """
    features_X1 = np.asarray(features_X1)
    features_X2 = np.asarray(features_X2)
    target = np.asarray(target)
    if features_X1.ndim != 2 or features_X2.ndim != 2:
        raise ValueError("Both feature arrays must be 2D (n_samples, n_features).")
    if target.ndim == 1:
        target = target[:, None]  # (n_samples,) -> (n_samples, 1)
    if target.ndim != 2:
        raise ValueError("target must be 1D or 2D.")
    if not (features_X1.shape[0] == features_X2.shape[0] == target.shape[0]):
        raise ValueError("Both sources and the target must have the same sample count.")

    features_X12 = np.hstack((features_X1, features_X2))
    # (n_samples, d1) + (n_samples, d2) -> (n_samples, d1 + d2)
    feature_sets = (features_X1, features_X2, features_X12)
    if method == "standard":
        scores = tuple(LinearRegression().fit(X, target).score(X, target) for X in feature_sets)
    elif method == "ridge_cv":
        penalties = np.logspace(-3, 3, 50) if alphas is None else alphas
        scores = tuple(
            RidgeCV(alphas=penalties, fit_intercept=True, scoring="r2", cv=None)
            .fit(X, target)
            .best_score_
            for X in feature_sets
        )
    else:
        raise ValueError("Unknown method; use 'standard' or 'ridge_cv'.")

    r2_X1, r2_X2, r2_X12 = scores
    scale = 1.0
    if scale_by_target_variance:
        scale = np.sum((target - target.mean()) ** 2) / (len(target) - 1)

    return {
        "R²_X1": r2_X1,
        "R²_X2": r2_X2,
        "R²_X12": r2_X12,
        "unique_X1": (r2_X12 - r2_X2) * scale,
        "unique_X2": (r2_X12 - r2_X1) * scale,
        "common": (r2_X1 + r2_X2 - r2_X12) * scale,
        "unexplained": (1 - r2_X12) * scale,
    }
