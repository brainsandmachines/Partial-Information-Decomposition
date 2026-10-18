"""Regression checks for Eigen PID and its covariance input conventions."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import torch
from eigen_pid import debias_eigen_pid, eigen_pid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pid.calculation import pid_calc
from pid.utils import reorder_cov_blocks
from simulations.pca.ridge.pid_feature_middleware import calculate_theoretical_pid


def test_source_source_target_to_target_source_source() -> None:
    """Check the simulation covariance-order helper; no inputs or return value."""
    matrix = torch.tensor([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=torch.float64)
    # Nested list (3, 3) -> torch covariance (3, 3).
    reordered = reorder_cov_blocks(
        matrix,
        {"X1": 1, "X2": 1, "T": 1},
        ["X1", "X2", "T"],
        ["T", "X1", "X2"],
    )  # (3, 3) [X1, X2, T] -> (3, 3) [T, X1, X2]
    # (3, 3) [X1, X2, T] -> (3, 3) [T, X1, X2]
    expected = matrix[[2, 0, 1]][:, [2, 0, 1]]  # (3, 3) -> reordered (3, 3)
    torch.testing.assert_close(reordered, expected)
    theoretical = calculate_theoretical_pid(matrix, [1, 1, 1], "eigen_pid")
    direct = eigen_pid(expected, 1, 1, 1)
    assert theoretical["unq1"] == pytest.approx(float(direct["uix"]), abs=1e-10)


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
    """Check that uncorrected sample and covariance routes agree; returns None."""
    generator = torch.Generator().manual_seed(17)
    data = torch.randn((128, 6), generator=generator, dtype=torch.float64)  # draws -> (128, 6)
    sources, target = [data[:, 2:5], data[:, 5:]], [data[:, :2]]
    # (128, 6) -> [(128, 3), (128, 1)] sources and [(128, 2)] target
    sampled, sampled_mi = pid_calc(sources=sources, target=target)
    covariance = torch.cov(data.T)  # (128, 6) -> (6, 6)
    population, population_mi = pid_calc(
        config={"dt": 2, "dx1": 3, "dx2": 1, "bias_correction": False},
        covariance=covariance.numpy(),  # torch (6, 6) -> NumPy (6, 6)
    )
    for key in ("red", "unq1", "unq2", "syn", "obj", "union_info", "obj_bias"):
        assert sampled[key] == pytest.approx(population[key], abs=1e-10)
    assert sampled_mi == pytest.approx(population_mi, abs=1e-10)


def test_bias_correction_uses_combined_eigen_pid() -> None:
    """Check the seeded package correction and PID identities; returns None."""
    generator = torch.Generator().manual_seed(12)
    data = torch.randn((128, 4), dtype=torch.float64, generator=generator)  # draws -> (128, 4)
    sources, target = [data[:, 1:3], data[:, 3:]], [data[:, :1]]
    # (128, 4) -> [(128, 2), (128, 1)] and [(128, 1)]
    config = {
        "bias_correction": True,
        "n_resamples": 3,
        "n_shuffles": 3,
        "bias_seed": 51,
    }
    pid, mi = pid_calc(config=config, sources=sources, target=target)
    repeated, repeated_mi = pid_calc(config=config, sources=sources, target=target)
    direct = debias_eigen_pid(data, 1, 2, 1, n_resamples=3, n_shuffles=3, seed=51)
    for key in ("red", "unq1", "unq2", "syn", "obj", "union_info", "obj_bias"):
        assert pid[key] == pytest.approx(repeated[key], abs=1e-12)
    assert mi == pytest.approx(repeated_mi)
    for project_key, package_key in (
        ("red", "red"), ("unq1", "uix"), ("unq2", "uiy"), ("syn", "syn")
    ):
        assert pid[project_key] == pytest.approx(direct["combined"][package_key])
    assert mi["tri_mi"] == pytest.approx(direct["combined"]["mixy"])
    assert pid["obj_bias"] == pytest.approx(direct["union_bias"]["combined"])
    assert pid["extra"]["bias_method"] == "combined"
    assert pid["extra"]["mi_bias"] == direct["mi_bias"]
    assert mi["bi_mi_1"] == pytest.approx(pid["red"] + pid["unq1"])
    assert mi["bi_mi_2"] == pytest.approx(pid["red"] + pid["unq2"])
    assert mi["tri_mi"] == pytest.approx(sum(pid[k] for k in ("red", "unq1", "unq2", "syn")))


def test_bias_correction_requires_samples() -> None:
    """Check that covariance-only input cannot request resampling correction."""
    covariance = torch.eye(3, dtype=torch.float64)  # scalar dimensions -> (3, 3)
    with pytest.raises(ValueError, match="requires source and target samples"):
        pid_calc(
            config={"dt": 1, "dx1": 1, "dx2": 1, "bias_correction": True},
            covariance=covariance,
        )


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
