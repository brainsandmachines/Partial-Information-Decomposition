#!/usr/bin/env python3
"""Wrapper for warrenzha/flow-pid's exact Gaussian Thin-PID definition."""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from wrapper_utils import (
    THIN_PID_COLUMNS,
    covariance_example_context,
    load_module,
    parse_sizes,
    run_covariance_pid_wrapper,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FLOW_PID_ROOT = REPO_ROOT / "external" / "flow-pid"
DEFAULT_MATRIX_CSV = Path(__file__).resolve().parent / "evil_twin_whitened_correlation_1_1_1.csv"
DEFAULT_OUTPUT = "thin_evil_twin.csv"
SIMPLE_OUTPUT = "thin_simple_gaussian.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running Thin-PID on a covariance/correlation CSV.

    The input matrix is assumed to be ordered as [source1, source2, target],
    matching the other local wrappers. The wrapped flow-pid function expects
    [target, source1, source2], so this wrapper reorders blocks before calling
    the original solver.
    """
    parser = argparse.ArgumentParser(
        description="Run flow-pid Thin-PID on a covariance CSV.",
        epilog="Example: python library_wrappers/Thin_PID.py --example simple-gaussian --output /tmp/thin_simple.csv",
    )
    parser.add_argument("--example", choices=("simple-gaussian",), help="Run a small built-in Gaussian example.")
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX_CSV)
    parser.add_argument("--sizes", type=parse_sizes, default=(1, 1, 1))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--case", default="InputCov")
    parser.add_argument("--sample-size", type=int, help="Optional sample size for unbiased correction.")
    parser.add_argument("--verbose", action="store_true", help="Return and report optimizer history length.")
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
        verbose=False,
    )


def load_exact_gauss_thin_pid():
    """Load and return flow-pid's original ``exact_gauss_thin_pid`` function.

    The loader provides a tiny ``utils`` module containing only ``whiten`` and
    ``pinv``, the two utility symbols used by ``pid/thin_pid.py``. This avoids
    importing unrelated flow/distribution modules while leaving the upstream
    Thin-PID source unchanged.
    """
    if not FLOW_PID_ROOT.exists():
        raise ImportError(f"flow-pid repository not found at {FLOW_PID_ROOT}")

    linalg = load_module("_flow_pid_linalg", FLOW_PID_ROOT / "utils" / "linalg.py")
    estimate_channel = load_module(
        "_flow_pid_estimate_channel",
        FLOW_PID_ROOT / "utils" / "estimate_channel.py",
    )

    utils_shim = types.ModuleType("utils")
    utils_shim.whiten = estimate_channel.whiten
    utils_shim.pinv = linalg.pinv
    sys.modules["utils"] = utils_shim

    pid_pkg = types.ModuleType("pid")
    pid_pkg.__path__ = [str(FLOW_PID_ROOT / "pid")]
    sys.modules["pid"] = pid_pkg

    thin_pid = load_module("pid.thin_pid", FLOW_PID_ROOT / "pid" / "thin_pid.py")
    return thin_pid.exact_gauss_thin_pid


def main() -> int:
    """Load input, run flow-pid Thin-PID, and save the result CSV."""
    args = simple_example_args() if len(sys.argv) == 1 else parse_args()
    with covariance_example_context(args):
        return run_covariance_pid_wrapper(
            args,
            pid_definition="Thin",
            columns=THIN_PID_COLUMNS,
            solver_loader=load_exact_gauss_thin_pid,
            solver_kwargs={
                "verbose": args.verbose,
                "unbiased": args.sample_size is not None,
                "sample_size": args.sample_size,
            },
            include_union_objective=True,
            verbose_history=args.verbose,
            read_message=True,
            call_message="Calling external/flow-pid/pid/thin_pid.py: exact_gauss_thin_pid",
            written_message="Wrote Thin-PID result to {output}",
        )


if __name__ == "__main__":
    raise SystemExit(main())
