"""Shared regression scoring helpers for encoding and toy examples."""

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
