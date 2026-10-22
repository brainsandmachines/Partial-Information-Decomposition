"""Checks for the RSA-versus-BROJA Gaussian simulation."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simulations.rsa_vs_pid.run_rsa_vs_pid import (
    CONFIG,
    TRIAL_FIELDS,
    run_trial,
    sample_shift_invariant_example,
    validate_config,
)

TIMING_FIELDS = {"rsa_seconds", "pid_seconds"}


def test_generator_shapes_and_reproducibility() -> None:
    """Check generator shapes, dtype, finiteness, and seeding; returns None."""
    first = sample_shift_invariant_example(np.random.default_rng(3), 50, 4, 0.1)
    second = sample_shift_invariant_example(np.random.default_rng(3), 50, 4, 0.1)
    other = sample_shift_invariant_example(np.random.default_rng(4), 50, 4, 0.1)
    for array in first:
        assert array.shape == (50, 4)
        assert array.dtype == np.float64
        assert np.isfinite(array).all()
    for array, repeated in zip(first, second):
        np.testing.assert_array_equal(array, repeated)
    assert not np.array_equal(first[2], other[2])


def test_rsatoolbox_correlation_rdm_uses_rows_as_stimuli() -> None:
    """Cross-check rsatoolbox's correlation RDM orientation against NumPy; returns None."""
    patterns = np.random.default_rng(0).standard_normal((7, 4))  # (n_stimuli, n_features)
    rdm = calc_rdm(Dataset(measurements=patterns), method="correlation")
    expected = (1 - np.corrcoef(patterns))[np.triu_indices(7, k=1)]
    # (7, 4) -> (7, 7) stimulus correlations -> (21,) upper-triangle dissimilarities
    assert rdm.n_cond == 7
    np.testing.assert_allclose(rdm.dissimilarities[0], expected, atol=1e-12)


def test_correlation_rdm_ignores_shifts_along_ones() -> None:
    """Check that adding c_i * 1 to each stimulus pattern leaves the RDM unchanged; returns None."""
    rng = np.random.default_rng(1)
    patterns = rng.standard_normal((30, 6))  # (n_stimuli, n_features)
    shifts = 5.0 * rng.standard_normal((30, 1))  # (n_stimuli, 1)
    shifted = patterns + shifts * np.ones(6)  # (30, 1) * (6,) -> (30, 6)
    original_rdm = calc_rdm(Dataset(measurements=patterns), method="correlation")
    shifted_rdm = calc_rdm(Dataset(measurements=shifted), method="correlation")
    np.testing.assert_allclose(original_rdm.dissimilarities, shifted_rdm.dissimilarities, atol=1e-12)


def test_run_trial_is_finite_consistent_and_reproducible() -> None:
    """Check one small trial's fields, PID identities, and seeded reproducibility; returns None."""
    config = {**CONFIG, "n_shuffles": 5, "n_resamples": 5}
    row = run_trial(dimension=3, n_samples=200, trial=0, seed=11, config=config)
    assert set(row) == set(TRIAL_FIELDS)
    for key, value in row.items():
        if isinstance(value, float):
            assert math.isfinite(value), key
    for suffix in ("", "_raw"):
        red, unq_x, unq_y, syn = (row[f"{key}{suffix}"] for key in ("red", "unq_x", "unq_y", "syn"))
        assert red + unq_x == pytest.approx(row[f"mi_tx{suffix}"], abs=1e-9)
        assert red + unq_y == pytest.approx(row[f"mi_ty{suffix}"], abs=1e-9)
        assert red + unq_x + unq_y + syn == pytest.approx(row[f"mi_txy{suffix}"], abs=1e-9)
    assert row["delta_rsa"] == pytest.approx(row["rsa_x"] - row["rsa_y"])
    assert row["delta_ui"] == pytest.approx(row["unq_x"] - row["unq_y"])

    repeated = run_trial(dimension=3, n_samples=200, trial=0, seed=11, config=config)
    for key in set(TRIAL_FIELDS) - TIMING_FIELDS:
        assert repeated[key] == row[key], key


def test_validate_config_requires_full_rank_sample_covariance() -> None:
    """Check that n < 3d + 1 is rejected and n = 3d + 1 is accepted; returns None."""
    with pytest.raises(ValueError, match="3d \\+ 1"):
        validate_config({**CONFIG, "dimensions": [5], "sample_sizes": [15]})
    validate_config({**CONFIG, "dimensions": [5], "sample_sizes": [16]})
