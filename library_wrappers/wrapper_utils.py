"""Shared utilities for the Python PID covariance wrappers."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from typing import Callable

import numpy as np


BASE_PID_COLUMNS = [
    "unique_source1",
    "unique_source2",
    "redundancy",
    "synergy",
    "I_source1_target",
    "I_source2_target",
    "joint_mutual_information",
    "interaction_information",
]


THIN_PID_COLUMNS = [
    "unique_source1",
    "unique_source2",
    "redundancy",
    "synergy",
    "I_source1_target",
    "I_source2_target",
    "joint_mutual_information",
    "union_information",
    "optimization_objective",
    "interaction_information",
]

SIMPLE_GAUSSIAN_SIZES = (1, 1, 1)
SIMPLE_GAUSSIAN_COVARIANCE = np.array(
    [
        [1.5, 0.6, 1.0],
        [0.6, 0.86, 0.6],
        [1.0, 0.6, 1.0],
    ],
    dtype=float,
)
SIMPLE_GAUSSIAN_CASE = "SimpleGaussian"


def parse_sizes(value: str) -> tuple[int, int, int]:
    """Parse source1,source2,target dimensions and reject invalid values."""
    try:
        sizes = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--sizes must look like 1,1,1") from exc
    if len(sizes) != 3 or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--sizes must look like 1,1,1")
    return sizes


def read_matrix(path: Path) -> np.ndarray:
    """Read a square covariance or correlation matrix from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"input matrix CSV does not exist: {path}")
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square: {path}")
    return matrix


def validate_covariance(matrix: np.ndarray, expected_shape: tuple[int, int]) -> None:
    """Validate shape and basic covariance/correlation symmetry."""
    if matrix.shape != expected_shape:
        raise ValueError(f"matrix shape must be {expected_shape}, got {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("matrix must be symmetric within atol=1e-8")


def source_source_target_to_target_source_source(matrix: np.ndarray, sizes: tuple[int, int, int]) -> np.ndarray:
    """Reorder a [source1, source2, target] matrix to [target, source1, source2]."""
    n1, n2, nt = sizes
    total = n1 + n2 + nt
    validate_covariance(matrix, (total, total))
    order = list(range(n1 + n2, total)) + list(range(0, n1)) + list(range(n1, n1 + n2))
    return matrix[np.ix_(order, order)]


def write_simple_gaussian_covariance(path: Path) -> None:
    """Write the shared simple [source1, source2, target] Gaussian covariance example."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, SIMPLE_GAUSSIAN_COVARIANCE, delimiter=",")


def simple_gaussian_samples(num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate raw samples from the shared simple Gaussian example.

    Returns arrays in flow-pid order: target/M, source1/X, source2/Y.
    """
    if num_samples < 4:
        raise ValueError("--num-samples must be at least 4")
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        np.zeros(len(SIMPLE_GAUSSIAN_SIZES)),
        SIMPLE_GAUSSIAN_COVARIANCE,
        size=num_samples,
    )
    x = samples[:, 0:1]
    y = samples[:, 1:2]
    m = samples[:, 2:3]
    return m, x, y


@contextmanager
def covariance_example_context(args: argparse.Namespace):
    """Temporarily attach the shared simple covariance example to wrapper args."""
    if getattr(args, "example", None) != "simple-gaussian":
        yield
        return

    with tempfile.TemporaryDirectory(prefix="simple_gaussian_pid_") as temp_dir:
        matrix_csv = Path(temp_dir) / "simple_gaussian_covariance_1_1_1.csv"
        write_simple_gaussian_covariance(matrix_csv)
        args.matrix_csv = matrix_csv
        args.sizes = SIMPLE_GAUSSIAN_SIZES
        args.case = SIMPLE_GAUSSIAN_CASE
        yield


def pid_result_row(
    values: tuple[float, ...],
    case: str,
    pid_definition: str,
    *,
    include_union_objective: bool = False,
) -> dict[str, object]:
    """Convert a Gaussian PID tuple into the local one-row CSV schema."""
    imx, imy, imxy, fourth, fifth, uix, uiy, ri, si = values[:9]
    row = {
        "case": case,
        "pid_definition": pid_definition,
        "unique_source1": uix,
        "unique_source2": uiy,
        "redundancy": ri,
        "synergy": si,
        "I_source1_target": imx,
        "I_source2_target": imy,
        "joint_mutual_information": imxy,
        "interaction_information": si - ri,
    }
    if include_union_objective:
        row["union_information"] = fourth
        row["optimization_objective"] = fifth
    return row


def write_pid_row(row: dict[str, object], path: Path, columns: list[str]) -> None:
    """Write one standardized PID result row to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "pid_definition", *columns])
        writer.writeheader()
        writer.writerow(row)


def print_pid_result(row: dict[str, object], columns: list[str]) -> None:
    """Print one PID result in a compact debug table."""
    labels = {
        "case": "Case",
        "pid_definition": "Definition",
        "unique_source1": "Unique source 1",
        "unique_source2": "Unique source 2",
        "redundancy": "Redundancy",
        "synergy": "Synergy",
        "I_source1_target": "I(source1; target)",
        "I_source2_target": "I(source2; target)",
        "joint_mutual_information": "I(source1, source2; target)",
        "union_information": "Union information",
        "optimization_objective": "Optimization objective",
        "interaction_information": "Interaction information",
    }
    title = f"{row['pid_definition']} PID - {row['case']}"
    print()
    print(title)
    print("=" * len(title))
    print("Example model: source1 and source2 are noisy copies of one Gaussian target.")
    print()
    for column in columns:
        value = row.get(column, "")
        if value == "":
            continue
        if isinstance(value, (float, int, np.floating)):
            value = f"{float(value): .8f}"
        print(f"{labels[column]:30} {value}")
    print()


def load_module(module_name: str, path: Path) -> types.ModuleType:
    """Load one source file as a module without importing package __init__ files."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_covariance_pid_wrapper(
    args: argparse.Namespace,
    solver: Callable[..., tuple[float, ...]] | None = None,
    *,
    pid_definition: str,
    columns: list[str],
    solver_loader: Callable[[], Callable[..., tuple[float, ...]]] | None = None,
    solver_kwargs: dict[str, object] | None = None,
    include_union_objective: bool = False,
    verbose_history: bool = False,
    read_message: bool = False,
    call_message: str | None = None,
    written_message: str | None = None,
) -> int:
    """Run the common covariance-wrapper flow and write a one-row PID CSV."""
    try:
        n1, n2, nt = args.sizes
        matrix_path = args.matrix_csv.expanduser()
        output_path = args.output.expanduser() if args.output is not None else None

        if read_message:
            print(f"Reading covariance/correlation matrix from {matrix_path}")
        matrix = read_matrix(matrix_path)
        cov = source_source_target_to_target_source_source(matrix, args.sizes)

        if call_message:
            print(call_message)
        if solver is None:
            if solver_loader is None:
                raise ValueError("run_covariance_pid_wrapper needs solver or solver_loader")
            solver = solver_loader()
        values = solver(cov, nt, n1, n2, **(solver_kwargs or {}))
        if verbose_history:
            values, obj_hist = values
            print(f"Optimizer objective history length: {len(obj_hist)}")

        row = pid_result_row(
            values,
            args.case,
            pid_definition,
            include_union_objective=include_union_objective,
        )
        if args.output is None:
            print_pid_result(row, columns)
        else:
            write_pid_row(row, output_path, columns)
            message = (
                written_message.format(output=output_path, raw_output=args.output)
                if written_message
                else f"Wrote {args.output}"
            )
            print(message)
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
