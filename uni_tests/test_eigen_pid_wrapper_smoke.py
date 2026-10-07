"""Regression checks for Eigen PID and its covariance input conventions."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from eigen_pid import debias_eigen_pid, eigen_pid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Partial_Information_Decomposition.PID_calc import pid_calc
from Simulations.PCA_Ridge.pid_feature_middleware import calculate_theoretical_pid
from Simulations.Theoretical_Examples.Covariance.cov_functions import make_direct_true_cov_from_config, change_covariance_order
from Simulations.Theoretical_Examples.Covariance.sample_simulation import simulation


def test_source_source_target_to_target_source_source() -> None:
    """Check the simulation covariance-order helper; no inputs or return value."""
    matrix = torch.tensor([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=torch.float64)
    # Nested list (3, 3) -> torch covariance (3, 3).
    reordered = change_covariance_order(matrix, [2, 0, 1], [1, 1, 1])
    # (3, 3) [X1, X2, T] -> (3, 3) [T, X1, X2]
    expected = matrix[[2, 0, 1]][:, [2, 0, 1]]  # (3, 3) -> reordered (3, 3)
    torch.testing.assert_close(reordered, expected)


@pytest.mark.parametrize("method", [None, "eigen", "eigen_pid"])
def test_eigen_pid_contract(method: str | None) -> None:
    """Check aliases, output keys, bits and source order; returns None."""
    generator = torch.Generator().manual_seed(56)
    a = torch.randn((6, 6), dtype=torch.float64, generator=generator)  # draws -> (6, 6)
    cov = a @ a.T + torch.eye(6, dtype=torch.float64)  # (6, 6) -> (6, 6) SPD covariance
    config = {"dt": 2, "dx1": 3, "dx2": 1, "bias_correction": False}
    pid, mi = pid_calc(config=config, covariance=cov, method=method)
    direct = eigen_pid(cov, 2, 3, 1)
    for old_key, new_key in (("red", "red"), ("unq1", "uix"), ("unq2", "uiy"), ("syn", "syn")):
        assert isinstance(pid[old_key], float)
        assert pid[old_key] == pytest.approx(float(direct[new_key]), abs=1e-10)
    assert pid["method"] == "eigen_pid"
    assert mi["bi_mi_1"] == pytest.approx(pid["red"] + pid["unq1"], abs=1e-10)
    assert mi["bi_mi_2"] == pytest.approx(pid["red"] + pid["unq2"], abs=1e-10)
    assert mi["tri_mi"] == pytest.approx(sum(pid[k] for k in ("red", "unq1", "unq2", "syn")), abs=1e-10)
    assert pid["obj"] - pid["union_info"] == pytest.approx(pid["obj_bias"])
    indices = [0, 1, 5, 2, 3, 4]
    swapped_cov = cov[indices][:, indices]  # (6, 6) [T, X1, X2] -> (6, 6) [T, X2, X1]
    swapped, _ = pid_calc(config={**config, "dx1": 1, "dx2": 3}, covariance=swapped_cov)
    for first, second in (("unq1", "unq2"), ("unq2", "unq1"), ("red", "red"), ("syn", "syn")):
        assert pid[first] == pytest.approx(swapped[second], abs=1e-10)


@pytest.mark.parametrize("method", ["tilde", "delta", "thin", "thin_pid", "flow", "flow_pid", "idep"])
def test_removed_backends_rejected(method: str) -> None:
    """Check that an obsolete method string raises instead of relabeling results."""
    with pytest.raises(ValueError, match="Only Eigen PID"):
        pid_calc(method=method)


def test_sample_covariance_equivalence() -> None:
    """Check sample/covariance routes and population bias handling; returns None."""
    generator = torch.Generator().manual_seed(17)
    data = torch.randn((128, 6), generator=generator, dtype=torch.float64)  # draws -> (128, 6)
    sources, target = [data[:, 2:5], data[:, 5:]], [data[:, :2]]
    # (128, 6) -> [(128, 3), (128, 1)] sources and [(128, 2)] target
    sampled, sampled_mi = pid_calc(sources=sources, target=target)
    covariance = torch.cov(data.T)  # (128, 6) -> (6, 6)
    population, population_mi = pid_calc(
        config={"dt": 2, "dx1": 3, "dx2": 1, "bias_correction": True},
        covariance=covariance.numpy(),  # torch (6, 6) -> NumPy (6, 6)
    )
    for key in ("red", "unq1", "unq2", "syn", "obj", "union_info", "obj_bias"):
        assert sampled[key] == pytest.approx(population[key], abs=1e-10)
    assert sampled_mi == pytest.approx(population_mi, abs=1e-10)


def test_sample_bias_preserves_formula() -> None:
    """Check saved pre-migration correction values; no inputs or return value."""
    generator = torch.Generator().manual_seed(81)
    data = torch.randn((128, 6), dtype=torch.float64, generator=generator)  # draws -> (128, 6)
    data[:, 2:5] += data[:, :1]  # (128, 3) + (128, 1) -> (128, 3)
    data[:, 5:] += data[:, 1:2]  # (128, 1) + (128, 1) -> (128, 1)
    pid, mi = pid_calc(
        config={"bias_correction": True},
        sources=[data[:, 2:5], data[:, 5:]], target=[data[:, :2]],
        # (128, 6) -> [(128, 3), (128, 1)] and [(128, 2)]
    )
    # Recorded from the old Eigen wrapper's compatibility correction before
    # migration: guards against changing the estimator's bias-correction policy.
    expected = [0.9144428674569413, 0.4508435702655605, 1.3694611478166754,
                1.3642539032395231, 0.9134103329739627, 0.44981103578258186,
                0.0010325344829786864, 0.00520724457715227]
    actual = [mi["bi_mi_1"], mi["bi_mi_2"], mi["tri_mi"], pid["union_info"],
              pid["unq1"], pid["unq2"], pid["red"], pid["syn"]]
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("correction_method", ["permutation", "combined"])
def test_objective_corrections(correction_method: str) -> None:
    """Check seeded objective corrections and PID identities; returns None."""
    generator = torch.Generator().manual_seed(12)
    data = torch.randn((128, 4), dtype=torch.float64, generator=generator)  # draws -> (128, 4)
    sources, target = [data[:, 1:3], data[:, 3:]], [data[:, :1]]
    # (128, 4) -> [(128, 2), (128, 1)] and [(128, 1)]
    config = {
        "bias_correction": False, "param_bias_method": correction_method,
        "device": "cpu", "rng_seed": 42, "n_perm": 3,
        "n_resamples": 3, "n_shuffles": 3, "bias_seed": 51,
    }
    pid, mi = pid_calc(config=config, sources=sources, target=target, param_bias=True)
    repeated, repeated_mi = pid_calc(config=config, sources=sources, target=target, param_bias=True)
    for key in ("red", "unq1", "unq2", "syn", "obj", "union_info", "obj_bias"):
        assert pid[key] == pytest.approx(repeated[key], abs=1e-12)
    assert mi == pytest.approx(repeated_mi)
    assert mi["bi_mi_1"] == pytest.approx(pid["red"] + pid["unq1"])
    assert mi["bi_mi_2"] == pytest.approx(pid["red"] + pid["unq2"])
    assert mi["tri_mi"] == pytest.approx(sum(pid[k] for k in ("red", "unq1", "unq2", "syn")))
    if correction_method == "combined":
        direct = debias_eigen_pid(data, 1, 2, 1, n_resamples=3, n_shuffles=3, seed=51)
        assert pid["syn"] == pytest.approx(direct["combined"]["syn"])
        assert mi["tri_mi"] == pytest.approx(direct["combined"]["mixy"])
        assert pid["obj_bias"] == pytest.approx(direct["union_bias"]["combined"])
        assert pid["extra"]["mi_bias"] == direct["mi_bias"]
        mismatched_cov = torch.eye(4, dtype=torch.float64)  # scalar dimension -> (4, 4)
        with pytest.raises(ValueError, match="covariance of the supplied samples"):
            pid_calc(config=config, sources=sources, target=target,
                     covariance=mismatched_cov, param_bias=True)


def test_covariance_experiment_consumers(tmp_path: Path) -> None:
    """Run a tiny covariance experiment in tmp_path and check PCA result mapping."""
    config = {
        "dt": 1, "dx1": 2, "dx2": 3, "p_scale": 0.1, "q_scale": 0.5, "r_scale": 0.2,
        "n_samples": 128, "n_trials": 2, "seed": 19, "device": "cpu",
        "bias_correction": False, "param_bias": False, "output_dir": str(tmp_path),
    }
    results = simulation(config, ["eigen_pid"], experiment_name="migration_check")
    covariance = make_direct_true_cov_from_config(config)  # config -> (6, 6) [X1, X2, T]
    ordered = change_covariance_order(covariance, [2, 0, 1], [2, 3, 1])
    # (6, 6) [X1, X2, T] -> (6, 6) [T, X1, X2]
    expected = eigen_pid(ordered, 1, 2, 3)
    theoretical, _ = results["eigen_pid"]["theoretical"]
    assert theoretical["unq1"] == pytest.approx(float(expected["uix"]), abs=1e-10)
    pca_truth = calculate_theoretical_pid(covariance, [2, 3, 1], "eigen_pid")
    assert pca_truth["unq1"] == pytest.approx(theoretical["unq1"], abs=1e-10)
    assert len(pca_truth) == 7
    assert len(list(tmp_path.rglob("*.csv"))) == 1


def test_source_transform_once() -> None:
    """Check the existing on_rvs hook is applied once with updated dimensions."""
    generator = torch.Generator().manual_seed(29)
    data = torch.randn((128, 5), dtype=torch.float64, generator=generator)  # draws -> (128, 5)
    sources, target = [data[:, 1:3], data[:, 3:]], [data[:, :1]]
    transformed = [sources[0][:, :1], sources[1]]  # [(128, 2), (128, 2)] -> [(128, 1), (128, 2)]
    transform = Mock(return_value=transformed)
    pid, mi = pid_calc(sources=sources, target=target, on_rvs=transform)
    transform.assert_called_once_with(sources)
    expected, expected_mi = pid_calc(sources=transformed, target=target)
    for key in ("red", "unq1", "unq2", "syn"):
        assert pid[key] == pytest.approx(expected[key])
    assert mi == pytest.approx(expected_mi)
