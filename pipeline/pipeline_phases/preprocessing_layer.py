import torch 
import numpy as np
import yaml
from pathlib import Path
import sys
from sklearn.linear_model import Ridge




"""This file will contain the preprocessing layer of the pipeline. For example: 
(Gaussian copulas, normalizing, standardizing, etc...)

For now it is empty, but in the future it will contain functions that will be applied to the features after feature extraction and before feature manipulation."""




def permute_rv(target,source1,source2,source1_perm=False,source2_perm=False,target_perm=False,rng_seed=56):
    """Permute the random variable rv according to the configuration provided in config.
    If rv is a tuple, all blocks are permuted with the same permutation.
    X is kept fixed, so any internal structure within X is preserved.
    
    input: 
        Experiment configuration
        rvs: tuple of random variables (source1, source2, target)
        source1: bool, whether to permute source1
        source2: bool, whether to permute source2
        target: bool, whether to permute target
    output:
        permuted_rvs: tuple of permuted random variables (source1, source2, target)
    """
    
    print("Running permuation on with RNG seed: ", rng_seed)
    n = target.shape[0]
    rng = torch.Generator()
    rng.manual_seed(rng_seed)

    idx = torch.randperm(n,generator=rng)

    source1 = source1[idx] if source1_perm and source1 is not None else source1
    source2 = source2[idx] if source2_perm and source2 is not None else source2
    target = target[idx] if target_perm and target is not None else target

    return source1 ,source2,target



from __future__ import annotations

import numpy as np
import torch

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error


def ridge_on_target_cv(
    source: torch.Tensor,
    target: torch.Tensor,
    alphas: list[float] | np.ndarray | None = None,
    outer_cv: int = 5,
    inner_cv: int = 5,
    scoring: str = "r2",
    shuffle: bool = True,
    random_state: int = 0,
):
    """
    Fit ridge regression fsrom source to target with nested cross-validation.

    For each outer training block:
        1. Use inner CV to choose the best alpha.
        2. Evaluate that alpha on the outer validation block.

    Then:
        3. Select the alpha from the best outer block.
        4. Refit Ridge on the full dataset using this alpha.
        5. Return predictions, or residuals if return_residuals=True.

    Inputs:
        target:
            torch.Tensor of shape (n_samples, n_target_features).
        source:
            torch.Tensor of shape (n_samples, n_source_features).
        alphas:
            Candidate ridge regularization strengths.
        outer_cv:
            Number of outer CV folds used to choose the best training block.
        inner_cv:
            Number of inner CV folds used to choose alpha inside each outer fold.
        scoring:
            "r2" or "neg_mse".
        return_residuals:
            If True, return target - prediction.
            If False, return prediction.

    Outputs:
        output:
            torch.Tensor of shape (n_samples, n_target_features).
        info:
            Dictionary with chosen alpha and CV diagnostics.
    """

    if alphas is None:
        alphas = np.logspace(-3, 3, 50)

    if target.shape[0] != source.shape[0]:
        raise ValueError(
            f"target and source must have same number of samples. "
            f"Got target={target.shape[0]}, source={source.shape[0]}."
        )

    device = target.device
    dtype = target.dtype

    X = source.detach().cpu().numpy()
    y = target.detach().cpu().numpy()

    outer_splitter = KFold(
        n_splits=outer_cv,
        shuffle=shuffle,
        random_state=random_state,
    )

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_splitter.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        ridge_cv = RidgeCV(
            alphas=alphas,
            cv=inner_cv,
            scoring="r2" if scoring == "r2" else "neg_mean_squared_error",
            fit_intercept=True,
        )

        ridge_cv.fit(X_train, y_train)

        y_val_pred = ridge_cv.predict(X_val)

        if scoring == "r2":
            val_score = r2_score(y_val, y_val_pred, multioutput="uniform_average")
        elif scoring == "neg_mse":
            val_score = -mean_squared_error(y_val, y_val_pred)
        else:
            raise ValueError("scoring must be either 'r2' or 'neg_mse'.")

        fold_results.append(
            {
                "fold": fold_idx,
                "best_alpha": float(ridge_cv.alpha_),
                "val_score": float(val_score),
            }
        )

    best_fold = max(fold_results, key=lambda d: d["val_score"])
    best_alpha = best_fold["best_alpha"]

    final_model = Ridge(alpha=best_alpha, fit_intercept=True)
    final_model.fit(X, y)

    y_pred = final_model.predict(X)


    res = y - y_pred


    output = torch.from_numpy(y_pred).to(device=device, dtype=dtype)

    info = {
        "best_alpha": best_alpha,
        "best_fold": best_fold,
        "fold_results": fold_results,
        "alphas": list(map(float, alphas)),
        "return_residuals": res,
    }

    return output, info