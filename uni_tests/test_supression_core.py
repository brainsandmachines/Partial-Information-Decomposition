import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from encoding_model import suppression_core as se


# tests/test_suppression_effect.py

def make_toy(n=50, p=20, t=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    W = rng.standard_normal((p, t))
    y = X @ W + 0.1 * rng.standard_normal((n, t))
    return rng, X, y


def test_create_predictions_shapes_and_none():
    rng, X, y = make_toy()
    reg = LinearRegression().fit(X, y)

    y_lh, y_rh = se.create_predictions(reg, reg, X)
    assert y_lh.shape == y.shape
    assert y_rh.shape == y.shape

    y_lh, y_rh = se.create_predictions(None, reg, X)
    assert y_lh is None
    assert y_rh.shape == y.shape


def test_create_encoder_uses_subset_and_fits():
    rng, X, y = make_toy(p=30, t=2)
    model, X_sel = se.create_encoder(rng, X, y, n_features=7)

    assert X_sel.shape == (X.shape[0], 7)
    y_hat = model.predict(X_sel)
    assert y_hat.shape == y.shape
    assert np.isfinite(y_hat).all()


def test_permutate_models_shapes_and_not_identical():
    rng, X, _ = make_toy(n=40, p=12)
    X1, X2 = se.permutate_models(rng, X, suppression_strength=0.5)

    assert X1.shape == X.shape
    assert X2.shape == X.shape
    # Usually they should differ from original; not strict equality test, but sanity:
    assert not np.allclose(X1, X)
    assert not np.allclose(X2, X)



def test_noise_component_shapes_and_noise_only_in_X2_when_not_permuted():
    rng, X, _ = make_toy(n=50, p=18)
    X1, X2 = se.noise_component(rng, X, suppression_strength=0.6, permutation=False)

    assert X1.shape == X.shape
    assert X2.shape == X.shape
    # X2 should be random noise, not equal to X
    assert not np.allclose(X2, X)
    # X1 should differ from X because you add noise_strength * noise :contentReference[oaicite:4]{index=4}
    assert not np.allclose(X1, X)


def test_commonality_analysis_keys_and_identity():
    rng, X, y = make_toy(n=60, p=10, t=1)
    # make A and B different feature sets
    A = X[:, :5]
    B = X[:, 5:]

    out = se.commonality_analysis(A, B, y, method="standard")
    assert set(out.keys()) == {"R²_A", "R²_B", "R²_AB", "unique_A", "unique_B", "common", "unexplained"}

    # identity: unique_A + unique_B + common == R²_AB
    lhs = out["unique_A"] + out["unique_B"] + out["common"]
    assert np.isclose(lhs, out["R²_AB"])


def test_commonality_analysis_invalid_method_raises():
    rng, X, y = make_toy(n=30, p=10, t=1)
    with pytest.raises(ValueError):
        se.commonality_analysis(X[:, :5], X[:, 5:], y, method="not_a_method")
