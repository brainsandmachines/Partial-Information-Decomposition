import torch 
import numpy as np
import yaml
from pathlib import Path
import sys
from sklearn.linear_model import Ridge
import numpy as np
import torch

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error



"""This file will contain the preprocessing layer of the pipeline. For example: 
(Gaussian copulas, normalizing, standardizing, etc...)
"""

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







def _to_numpy(x):
    """Convert torch.Tensor to NumPy only if needed."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _from_numpy_like(x_np, reference):
    """Return x_np as torch if reference is torch, otherwise NumPy."""
    if torch.is_tensor(reference):
        return torch.from_numpy(x_np).to(
            device=reference.device,
            dtype=reference.dtype,
        )
    return x_np


def ridge_train_to_test_prediction(
    source_train,
    target_train,
    source_test,
    target_test=None,
    alphas=None,
    inner_cv: int = 5,
    scoring: str = "r2",
    shuffle: bool = True,
    random_state: int = 0,
):
    """
    Fit ridge encoding model on training data and return held-out test predictions.

    This matches the PDF plan:
        source_train -> target_train is used for normalization, alpha selection, and fitting.
        source_test is used only to generate held-out predictions.
        target_test is optional and used only for diagnostics.

    Inputs:
        source_train:
            array-like of shape (n_train, n_source_features).
        target_train:
            array-like of shape (n_train, n_target_features).
        source_test:
            array-like of shape (n_test, n_source_features).
        target_test:
            optional array-like of shape (n_test, n_target_features).
        alphas:
            candidate ridge regularization strengths.
        inner_cv:
            number of CV folds used to choose alpha inside the training set.
        scoring:
            "r2" or "neg_mse".

    Outputs:
        test_prediction:
            same type as target_train.
            Shape: (n_test, n_target_features).
        info:
            dictionary with chosen alpha and diagnostics.
    """

    if alphas is None:
        alphas = np.logspace(-1, 8, 50)

    if scoring not in {"r2", "neg_mse"}:
        raise ValueError("scoring must be either 'r2' or 'neg_mse'.")

    X_train = _to_numpy(source_train)
    y_train = _to_numpy(target_train)
    X_test = _to_numpy(source_test)

    if target_test is not None:
        y_test = _to_numpy(target_test)
    else:
        y_test = None

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"source_train and target_train must have the same number of samples. "
            f"Got {X_train.shape[0]} and {y_train.shape[0]}."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"source_train and source_test must have the same number of features. "
            f"Got {X_train.shape[1]} and {X_test.shape[1]}."
        )

    if y_test is not None and X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"source_test and target_test must have the same number of samples. "
            f"Got {X_test.shape[0]} and {y_test.shape[0]}."
        )

    if inner_cv > X_train.shape[0]:
        raise ValueError(
            f"inner_cv={inner_cv} cannot be larger than the number of training samples "
            f"n_train={X_train.shape[0]}."
        )

    for name, arr in {
        "source_train": X_train,
        "target_train": y_train,
        "source_test": X_test,
    }.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or Inf values.")

    if y_test is not None and not np.isfinite(y_test).all():
        raise ValueError("target_test contains NaN or Inf values.")

    sklearn_scoring = "r2" if scoring == "r2" else "neg_mean_squared_error"

    inner_splitter = KFold(
        n_splits=inner_cv,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )

    model = make_pipeline(
        StandardScaler(),
        Ridge(fit_intercept=True, solver="svd"),
    )

    search = GridSearchCV(
        estimator=model,
        param_grid={"ridge__alpha": alphas},
        scoring=sklearn_scoring,
        cv=inner_splitter,
    )

    search.fit(X_train, y_train)


    y_test_pred = search.predict(X_test)

    test_prediction = _from_numpy_like(y_test_pred, target_train)

    info = {
        "prediction_type": "held_out_test",
        "best_alpha": float(search.best_params_["ridge__alpha"]),
        "alphas": list(map(float, alphas)),
        "inner_cv": inner_cv,
        "scoring": scoring,
    }

    if y_test is not None:
        if scoring == "r2":
            test_score = r2_score(
                y_test,
                y_test_pred,
                multioutput="uniform_average",
            )
        else:
            test_score = -mean_squared_error(y_test, y_test_pred)

        info["test_score"] = float(test_score)
    print(f"Best alpha: {search.best_params_['ridge__alpha']} with test score={test_score:.4f} - ☑️")

    return test_prediction, info