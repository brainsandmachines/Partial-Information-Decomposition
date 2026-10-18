import numpy as np
import pytest

from simulations.theoretical.random_variables.commonality import commonality_analysis


def make_toy(n=50, p=20, t=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    W = rng.standard_normal((p, t))
    y = X @ W + 0.1 * rng.standard_normal((n, t))
    return rng, X, y


def test_commonality_analysis_keys_and_identity():
    rng, X, y = make_toy(n=60, p=10, t=1)
    A = X[:, :5]
    B = X[:, 5:]

    out = commonality_analysis(A, B, y, method="standard")
    assert set(out.keys()) == {
        "R²_X1",
        "R²_X2",
        "R²_X12",
        "unique_X1",
        "unique_X2",
        "common",
        "unexplained",
    }

    lhs = out["unique_X1"] + out["unique_X2"] + out["common"]
    assert np.isclose(lhs, out["R²_X12"])


def test_commonality_analysis_invalid_method_raises():
    rng, X, y = make_toy(n=30, p=10, t=1)
    with pytest.raises(ValueError):
        commonality_analysis(X[:, :5], X[:, 5:], y, method="not_a_method")
