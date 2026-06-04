#!/usr/bin/env python3
"""Wrapper for the gpid Gaussian tilde PID definition."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    from .wrapper_utils import parse_sizes
except ImportError:  # pragma: no cover - script-style import fallback
    from wrapper_utils import parse_sizes

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpid" / "src"))

import numpy as np
from gpid import tilde_pid


DEFAULT_MATRIX_CSV = Path(__file__).resolve().parent / "evil_twin_whitened_correlation_1_1_1.csv"
DEFAULT_OUTPUT = "tilde_evil_twin.csv"

PID_COLUMNS = [
    "unique_source1",
    "unique_source2",
    "redundancy",
    "synergy",
    "I_source1_target",
    "I_source2_target",
    "joint_mutual_information",
    "interaction_information",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gpid tilde PID on a covariance CSV.")
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX_CSV)
    parser.add_argument("--sizes", type=parse_sizes, default=(1, 1, 1))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--case", default="InputCov")
    parser.add_argument("--sample-size", type=int, help="Optional sample size for unbiased correction.")
    return parser.parse_args()


def read_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square: {path}")
    return matrix


def source_source_target_to_target_source_source(matrix: np.ndarray, sizes: tuple[int, int, int]) -> np.ndarray:
    n1, n2, nt = sizes
    total = n1 + n2 + nt
    if matrix.shape != (total, total):
        raise ValueError(f"matrix shape must be {(total, total)}, got {matrix.shape}")
    order = list(range(n1 + n2, total)) + list(range(0, n1)) + list(range(n1, n1 + n2))
    return matrix[np.ix_(order, order)]


def result_row(values: tuple[float, ...], case: str) -> dict[str, object]:
    imx, imy, imxy, _union_info, _obj, uix, uiy, ri, si = values[:9]
    return {
        "case": case,
        "pid_definition": "Tilde",
        "unique_source1": uix,
        "unique_source2": uiy,
        "redundancy": ri,
        "synergy": si,
        "I_source1_target": imx,
        "I_source2_target": imy,
        "joint_mutual_information": imxy,
        "interaction_information": si - ri,
    }


def write_row(row: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "pid_definition", *PID_COLUMNS])
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    try:
        args = parse_args()
        n1, n2, nt = args.sizes
        matrix = read_matrix(args.matrix_csv.expanduser())
        cov = source_source_target_to_target_source_source(matrix, args.sizes)
        values = tilde_pid.exact_gauss_tilde_pid(
            cov,
            nt,
            n1,
            n2,
            unbiased=args.sample_size is not None,
            sample_size=args.sample_size,
        )
        write_row(result_row(values, args.case), args.output.expanduser())
        print(f"Wrote {args.output}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
