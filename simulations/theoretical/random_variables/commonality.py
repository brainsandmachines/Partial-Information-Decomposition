"""Commonality analysis and its regression scoring helpers."""

import numpy as np


def compute_ols_cv_r2(X, y, return_model=False):
    """
    Compute cross-validated R2 using leave-one-out cross-validation.

    Uses RidgeCV with near-zero regularization (alpha=1e-16), which is
    effectively OLS but leverages the efficient GCV formula.

    Args:
        X (np.ndarray): Design matrix WITHOUT intercept (shape: [n, p]).
        y (np.ndarray): Target variable.
        return_model (bool): If True, return the fitted RidgeCV model together
            with the score.

    Returns:
        float or tuple: Cross-validated R2, or (model, R2) when return_model is
        True.
    """
    from sklearn.linear_model import RidgeCV

    ridge_cv = RidgeCV(alphas=[1e-16], fit_intercept=True, scoring="r2", cv=None)
    ridge_cv.fit(X, y)
    if return_model:
        return ridge_cv, ridge_cv.best_score_
    return ridge_cv.best_score_


def compute_ridge_cv_r2(X, y, alphas=None, return_model=False):
    """
    Compute cross-validated R2 using RidgeCV with efficient LOO cross-validation.

    RidgeCV uses generalized cross-validation (GCV), an efficient approximation
    to leave-one-out CV for ridge regression.

    Args:
        X (np.ndarray): Design matrix WITHOUT intercept (shape: [n, p]).
        y (np.ndarray): Target variable.
        alphas (array-like, optional): Array of alpha values to try.
            Defaults to np.logspace(-3, 3, 50).
        return_model (bool): If True, return the fitted RidgeCV model together
            with the score.

    Returns:
        float or tuple: Best cross-validated R2, or (model, R2) when
        return_model is True.
    """
    from sklearn.linear_model import RidgeCV

    if alphas is None:
        alphas = np.logspace(-3, 3, 50)

    ridge_cv = RidgeCV(alphas=alphas, fit_intercept=True, scoring="r2", cv=None)
    ridge_cv.fit(X, y)

    if return_model:
        return ridge_cv, ridge_cv.best_score_
    return ridge_cv.best_score_


def compute_r2(X, y, return_model=False):
    """
    Compute in-sample R2 for OLS regression.

    Args:
        X (np.ndarray): Design matrix WITHOUT intercept (shape: [n, p]).
        y (np.ndarray): Target variable.
        return_model (bool): If True, return the fitted LinearRegression model
            together with the score.

    Returns:
        float or tuple: In-sample R2, or (model, R2) when return_model is True.
    """
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X, y)
    score = model.score(X, y)
    if return_model:
        return model, score
    return score


def compute_lasso_cv_r2(X, y):
    """
    Compute in-sample R2 after fitting multi-output LassoCV.

    Returns:
        tuple: (model, R2), matching the previous toy-example helper behavior.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.multioutput import MultiOutputRegressor

    base_lasso = LassoCV(
        n_alphas=100,
        fit_intercept=True,
        cv=5,
        max_iter=5000,
    )
    mo_lasso = MultiOutputRegressor(base_lasso)
    mo_lasso.fit(X, y)
    r2 = mo_lasso.score(X, y)
    return mo_lasso, r2


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
    features_X1,
    features_X2,
    target,
    method="standard",
    alphas=None,
    scale_by_target_variance=False,
):
    """
    Decompose predictive power into unique, common, and unexplained components.

    Args:
        features_X1 (np.ndarray): First feature matrix, reported as X1.
        features_X2 (np.ndarray): Second feature matrix, reported as X2.
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

    r2_X1 = _score_only(compute_r2_fn(features_X1, target))
    r2_X2 = _score_only(compute_r2_fn(features_X2, target))
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
