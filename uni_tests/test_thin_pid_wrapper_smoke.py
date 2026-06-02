"""Smoke tests for the flow-pid Thin-PID wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "library_wrappers"))
PID_ENV_PYTHON = Path("/home/ohadshee/anaconda3/envs/PID_env/bin/python")
FLOW_PYTHON = str(PID_ENV_PYTHON if PID_ENV_PYTHON.exists() else Path(sys.executable))

from wrapper_utils import parse_sizes, source_source_target_to_target_source_source


def test_parse_sizes() -> None:
    """Shared size parsing should accept the wrapper dimension convention."""
    assert parse_sizes("1, 2,3") == (1, 2, 3)
    with pytest.raises(SystemExit):
        # argparse converts ArgumentTypeError into SystemExit when wired into a parser;
        # call through a tiny parser to check the actual CLI behavior.
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--sizes", type=parse_sizes)
        parser.parse_args(["--sizes", "1,0,1"])


def test_source_source_target_to_target_source_source() -> None:
    """The shared reorder helper should move target blocks before source blocks."""
    matrix = np.array(
        [
            [1.0, 0.1, 0.2],
            [0.1, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ]
    )
    reordered = source_source_target_to_target_source_source(matrix, (1, 1, 1))
    expected = matrix[np.ix_([2, 0, 1], [2, 0, 1])]
    assert np.allclose(reordered, expected)


def test_thin_pid_wrapper_help() -> None:
    """The wrapper CLI should parse and display help."""
    result = subprocess.run(
        [sys.executable, "library_wrappers/Thin_PID.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Run flow-pid Thin-PID" in result.stdout


def test_flow_pid_wrapper_help() -> None:
    """The Flow-PID wrapper CLI should parse without importing training dependencies."""
    result = subprocess.run(
        [sys.executable, "library_wrappers/Flow_PID.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Run flow-pid Flow-PID" in result.stdout


def test_flow_pid_wrapper_simple_gaussian_runs(tmp_path: Path) -> None:
    """The Flow-PID wrapper should run the shared simple Gaussian example."""
    output = tmp_path / "flow.csv"
    result = subprocess.run(
        [
            FLOW_PYTHON,
            "library_wrappers/Flow_PID.py",
            "--example",
            "simple-gaussian",
            "--num-samples",
            "16",
            "--n-flows",
            "1",
            "--n-epochs",
            "1",
            "--batch-size",
            "8",
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert "Flow" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("script", "output_name"),
    [
        ("Delta_PID.py", "delta_simple.csv"),
        ("Tilde_PID.py", "tilde_simple.csv"),
        ("Thin_PID.py", "thin_simple.csv"),
    ],
)
def test_covariance_wrappers_run_simple_gaussian_example(script: str, output_name: str, tmp_path: Path) -> None:
    """The covariance wrappers should expose the shared simple Gaussian example."""
    result = subprocess.run(
        [
            sys.executable,
            f"library_wrappers/{script}",
            "--example",
            "simple-gaussian",
            "--output",
            str(tmp_path / output_name),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / output_name).exists()


@pytest.mark.parametrize(
    ("script", "output_name"),
    [
        ("Delta_PID.py", "delta.csv"),
        ("Tilde_PID.py", "tilde.csv"),
        ("Thin_PID.py", "thin.csv"),
    ],
)
def test_python_pid_wrappers_run_default(script: str, output_name: str, tmp_path: Path) -> None:
    """The Python covariance wrappers should run on the default evil-twin CSV."""
    result = subprocess.run(
        [sys.executable, f"library_wrappers/{script}", "--output", str(tmp_path / output_name)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / output_name).exists()


@pytest.mark.parametrize(
    ("script", "output_name"),
    [
        ("Delta_PID.py", "delta_simple_gaussian.csv"),
        ("Tilde_PID.py", "tilde_simple_gaussian.csv"),
        ("Thin_PID.py", "thin_simple_gaussian.csv"),
    ],
)
def test_covariance_wrappers_no_args_run_simple_gaussian(script: str, output_name: str, tmp_path: Path) -> None:
    """Running a covariance wrapper file directly should execute a small self-check."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "library_wrappers" / script)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "SimpleGaussian" in result.stdout
    assert not (tmp_path / output_name).exists()


def test_flow_wrapper_no_args_runs_simple_gaussian(tmp_path: Path) -> None:
    """Running Flow_PID.py directly should execute a short simple-Gaussian check."""
    result = subprocess.run(
        [FLOW_PYTHON, str(REPO_ROOT / "library_wrappers" / "Flow_PID.py")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Flow PID - SimpleGaussian" in result.stdout
    assert not (tmp_path / "flow_simple_gaussian.csv").exists()
