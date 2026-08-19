#!/usr/bin/env python3
"""Wrapper for the gpid Gaussian tilde PID definition."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from .wrapper_utils import (
        BASE_PID_COLUMNS,
        add_gpid_src_to_path,
        covariance_example_context,
        parse_sizes,
        run_covariance_pid_wrapper,
    )
except ImportError:  # pragma: no cover - script-style import fallback
    from wrapper_utils import (
        BASE_PID_COLUMNS,
        add_gpid_src_to_path,
        covariance_example_context,
        parse_sizes,
        run_covariance_pid_wrapper,
    )

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

GPID_SRC = add_gpid_src_to_path()

from gpid import tilde_pid

DEFAULT_MATRIX_CSV = Path(__file__).resolve().parent / "evil_twin_whitened_correlation_1_1_1.csv"
DEFAULT_OUTPUT = "tilde_evil_twin.csv"
SIMPLE_OUTPUT = "tilde_simple_gaussian.csv"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gpid tilde PID on a covariance CSV.",
        epilog="Example: python library_wrappers/Tilde_PID.py --example simple-gaussian --output /tmp/tilde_simple.csv",
    )
    parser.add_argument("--example", choices=("simple-gaussian",), help="Run a small built-in Gaussian example.")
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX_CSV)
    parser.add_argument("--sizes", type=parse_sizes, default=(1, 1, 1))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--case", default="InputCov")
    parser.add_argument("--sample-size", type=int, help="Optional sample size for unbiased correction.")
    args = parser.parse_args()
    if args.example == "simple-gaussian" and args.output == Path(DEFAULT_OUTPUT):
        args.output = Path(SIMPLE_OUTPUT)
    return args


def simple_example_args() -> argparse.Namespace:
    """Small debug example: source1 and source2 are noisy copies of one target."""
    return argparse.Namespace(
        example="simple-gaussian",
        matrix_csv=DEFAULT_MATRIX_CSV,
        sizes=(1, 1, 1),
        output=None,
        case="InputCov",
        sample_size=None,
    )


def main() -> int:
    args = simple_example_args() if len(sys.argv) == 1 else parse_args()
    with covariance_example_context(args):
        return run_covariance_pid_wrapper(
            args,
            tilde_pid.exact_gauss_tilde_pid,
            pid_definition="Tilde",
            columns=BASE_PID_COLUMNS,
            solver_kwargs={
                "unbiased": args.sample_size is not None,
                "sample_size": args.sample_size,
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
