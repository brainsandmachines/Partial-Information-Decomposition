from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
import sys
import pandas as pd
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))


from Partial_Information_Decomposition.mi_functions import np_safe_logdet

import numpy as np
import pandas as pd
from typing import Any, Callable
from numpy.typing import ArrayLike


# ============================================================
# Permutation null debias
# ============================================================

def permutation_null_debias(
    X: ArrayLike | tuple[ArrayLike, ...],
    Y: ArrayLike | tuple[ArrayLike, ...],
    func: Callable[..., float],
    *,
    n_perm: int = 20,
    random_state: int | np.random.Generator | None = None,
    **func_kwargs: Any,
) -> dict[str, Any]:

    rng = np.random.default_rng(random_state)

    X = tuple(np.asarray(x) for x in X) if isinstance(X, tuple) else np.asarray(X)
    Y = tuple(np.asarray(y) for y in Y) if isinstance(Y, tuple) else np.asarray(Y)

    n = Y[0].shape[0] if isinstance(Y, tuple) else Y.shape[0]

    raw = float(func(X, Y, **func_kwargs))

    if n_perm == 0:
        return {
            "debiased": raw,
            "raw": raw,
            "perm_mean": 0.0,
            "perm_std": 0.0,
            "perm_se": 0.0,
            "perm_values": np.empty(0, dtype=float),
            "n_perm": n_perm,
        }

    perm_values = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        idx = rng.permutation(n)

        if isinstance(Y, tuple):
            Y_perm = tuple(y[idx] for y in Y)
        else:
            Y_perm = Y[idx]

        perm_values[i] = float(func(X, Y_perm, **func_kwargs))

    perm_mean = float(np.mean(perm_values))
    perm_std = float(np.std(perm_values, ddof=1)) if n_perm > 1 else 0.0
    perm_se = perm_std / np.sqrt(n_perm)

    return {
        "debiased": raw - perm_mean,
        "raw": raw,
        "perm_mean": perm_mean,
        "perm_std": perm_std,
        "perm_se": perm_se,
        "perm_values": perm_values,
        "n_perm": n_perm,
    }


# ============================================================
# Logdet Gaussian MI
# ============================================================

def safe_logdet(A: np.ndarray, eps: float = 1e-8) -> float:
    A = np.asarray(A)
    A = A + eps * np.eye(A.shape[0])

    sign, val = np.linalg.slogdet(A)

    if sign <= 0:
        raise ValueError("Matrix is not positive definite.")

    return float(val)


def sample_cov(X: np.ndarray) -> np.ndarray:
    return np.cov(X, rowvar=False, bias=False)


def gaussian_mi_logdet(X: np.ndarray, Y: np.ndarray, eps: float = 1e-8) -> float:
    """
    Plug-in Gaussian MI estimator:

        I(X;Y) = 1/2 [log|S_X| + log|S_Y| - log|S_XY|]
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    Sx = sample_cov(X)
    Sy = sample_cov(Y)
    Sxy = sample_cov(np.hstack([X, Y]))

    return 0.5 * (
        safe_logdet(Sx, eps)
        + safe_logdet(Sy, eps)
        - safe_logdet(Sxy, eps)
    )


def gaussian_mi_from_cov(Sigma: np.ndarray, dx: int) -> float:
    """
    True Gaussian MI from the population covariance.
    """
    Sx = Sigma[:dx, :dx]
    Sy = Sigma[dx:, dx:]
    Sxy = Sigma

    return 0.5 * (
        safe_logdet(Sx, eps=0.0)
        + safe_logdet(Sy, eps=0.0)
        - safe_logdet(Sxy, eps=0.0)
    )


# ============================================================
# General multivariate Gaussian simulation
# ============================================================

def random_orthonormal_matrix(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return an n x k matrix with orthonormal columns.
    """
    A = rng.normal(size=(n, k))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]


def make_population_cov(
    dx: int,
    dy: int,
    canonical_corrs: list[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Construct a population covariance:

        Cov(X) = I_dx
        Cov(Y) = I_dy
        Cov(X,Y) = U diag(canonical_corrs) V.T

    The values in canonical_corrs must be smaller than 1.
    """
    k = len(canonical_corrs)

    if k > min(dx, dy):
        raise ValueError("Number of canonical correlations cannot exceed min(dx, dy).")

    if np.max(np.abs(canonical_corrs)) >= 1:
        raise ValueError("Canonical correlations must be strictly smaller than 1.")

    U = random_orthonormal_matrix(dx, k, rng)
    V = random_orthonormal_matrix(dy, k, rng)

    R = np.diag(canonical_corrs)

    Cxy = U @ R @ V.T

    Sigma = np.block([
        [np.eye(dx), Cxy],
        [Cxy.T,     np.eye(dy)]
    ])

    return Sigma


def sample_multivariate_gaussian(
    n: int,
    Sigma: np.ndarray,
    dx: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample (X,Y) from the joint Gaussian.
    """
    d_total = Sigma.shape[0]

    Z = rng.multivariate_normal(
        mean=np.zeros(d_total),
        cov=Sigma,
        size=n,
    )

    X = Z[:, :dx]
    Y = Z[:, dx:]

    return X, Y


def run_multivariate_logdet_permutation_sim(
    *,
    n: int = 100,
    dx: int = 15,
    dy: int = 20,
    canonical_corrs: list[float] = [0.6, 0.4, 0.2],
    n_trials: int = 100,
    n_perm: int = 30,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)

    Sigma = make_population_cov(
        dx=dx,
        dy=dy,
        canonical_corrs=canonical_corrs,
        rng=rng,
    )

    true_mi = gaussian_mi_from_cov(Sigma, dx=dx)

    rows = []

    for trial in range(n_trials):
        X, Y = sample_multivariate_gaussian(
            n=n,
            Sigma=Sigma,
            dx=dx,
            rng=rng,
        )

        out = permutation_null_debias(
            X,
            Y,
            gaussian_mi_logdet,
            n_perm=n_perm,
            random_state=rng,
        )

        rows.append({
            "trial": trial,
            "true_mi": true_mi,
            "raw": out["raw"],
            "perm_mean": out["perm_mean"],
            "debiased": out["debiased"],
        })

    df = pd.DataFrame(rows)

    summary = df[["raw", "perm_mean", "debiased"]].agg(["mean", "std"])
    summary.loc["bias_vs_true"] = summary.loc["mean"] - true_mi

    print("=" * 70)
    print("Multivariate Gaussian logdet MI simulation")
    print(f"n = {n}")
    print(f"dx = {dx}")
    print(f"dy = {dy}")
    print(f"canonical correlations = {canonical_corrs}")
    print(f"true MI = {true_mi:.6f}")
    print("=" * 70)
    print(summary)

    return df, summary


# ============================================================
# Example 1: null case
# ============================================================

df_null, summary_null = run_multivariate_logdet_permutation_sim(
    n=1000,
    dx=300,
    dy=300,
    canonical_corrs=[0.0, 0.0, 0.0],
    n_trials=100,
    n_perm=100,
    seed=1,
)


# ============================================================
# Example 2: dependent multivariate case
# ============================================================

df_dep, summary_dep = run_multivariate_logdet_permutation_sim(
    n=1000,
    dx=150,
    dy=100,
    canonical_corrs=[0.2, 0.1, 0.9],
    n_trials=100,
    n_perm=100,
    seed=2,
)