#!/usr/bin/env python3
"""Wrapper for warrenzha/flow-pid's normalizing-flow PID estimator."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np

from wrapper_utils import (
    BASE_PID_COLUMNS,
    SIMPLE_GAUSSIAN_CASE,
    load_module,
    parse_sizes,
    pid_result_row,
    print_pid_result,
    simple_gaussian_samples,
    write_pid_row,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FLOW_PID_ROOT = REPO_ROOT / "external" / "flow_pid"
DEFAULT_OUTPUT = "flow_pid_result.csv"
SIMPLE_OUTPUT = "flow_simple_gaussian.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Flow-PID on raw sample matrices.

    Combined sample CSVs are expected in local wrapper order
    [source1, source2, target]. Separate CSV inputs use direct flow-pid order:
    target/M, source1/X, source2/Y. Rows are samples and columns are features.
    """
    parser = argparse.ArgumentParser(
        description="Run flow-pid Flow-PID on raw sample CSV data.",
        epilog=(
            "Example: python library_wrappers/Flow_PID.py --example simple-gaussian "
            "--n-flows 1 --n-epochs 1 --batch-size 8 --output /tmp/flow_simple.csv"
        ),
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--samples-csv", type=Path, help="No-header sample CSV ordered source1,source2,target.")
    input_group.add_argument(
        "--example",
        choices=("simple-gaussian", "toy"),
        help="Run a built-in small Gaussian sample example.",
    )
    parser.add_argument("--sizes", type=parse_sizes, default=(1, 1, 1))
    parser.add_argument("--m-csv", type=Path, help="Optional target/M sample CSV; rows are samples.")
    parser.add_argument("--x-csv", type=Path, help="Optional source1/X sample CSV; rows are samples.")
    parser.add_argument("--y-csv", type=Path, help="Optional source2/Y sample CSV; rows are samples.")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--case", default="InputSamples")
    parser.add_argument("--n-flows", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=128, help="Built-in example sample count.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.example in ("simple-gaussian", "toy") and args.output == Path(DEFAULT_OUTPUT):
        args.output = Path(SIMPLE_OUTPUT)
    return args


def simple_example_args() -> argparse.Namespace:
    """Small debug example: train Flow-PID on samples from the shared Gaussian case."""
    return argparse.Namespace(
        samples_csv=None,
        example="simple-gaussian",
        sizes=(1, 1, 1),
        m_csv=None,
        x_csv=None,
        y_csv=None,
        output=None,
        case="InputSamples",
        n_flows=1,
        n_epochs=1,
        batch_size=8,
        lr=2e-4,
        device="cpu",
        seed=0,
        num_samples=32,
        verbose=False,
    )


def load_flow_pid():
    """Load and return flow-pid's original ``flow_pid`` function.

    The loader avoids importing broad package ``__init__`` files so this wrapper
    only pulls in the model and utility modules needed by `pid/flow_pid.py`.
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

    flows = load_module("_flow_pid_models_flows", FLOW_PID_ROOT / "models" / "flows.py")
    models_shim = types.ModuleType("models")
    models_shim.CartesianProductFlow = flows.CartesianProductFlow
    sys.modules["models"] = models_shim

    pid_pkg = types.ModuleType("pid")
    pid_pkg.__path__ = [str(FLOW_PID_ROOT / "pid")]
    sys.modules["pid"] = pid_pkg

    flow_pid_module = load_module("pid.flow_pid", FLOW_PID_ROOT / "pid" / "flow_pid.py")
    return flow_pid_module.flow_pid


def read_samples(path: Path, expected_columns: int | None = None) -> np.ndarray:
    """Read a two-dimensional sample CSV with rows as observations."""
    if not path.exists():
        raise FileNotFoundError(f"sample CSV does not exist: {path}")
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError(f"sample CSV must be two-dimensional: {path}")
    if expected_columns is not None and data.shape[1] != expected_columns:
        raise ValueError(f"{path} must have {expected_columns} columns, got {data.shape[1]}")
    return data


def split_combined_samples(samples: np.ndarray, sizes: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split [source1, source2, target] samples into flow-pid's (target, source1, source2)."""
    n1, n2, nt = sizes
    total = n1 + n2 + nt
    if samples.shape[1] != total:
        raise ValueError(f"samples must have {total} columns for --sizes {sizes}, got {samples.shape[1]}")
    x = samples[:, :n1]
    y = samples[:, n1:n1 + n2]
    m = samples[:, n1 + n2:]
    return m, x, y


def load_input_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load target/M, source1/X, and source2/Y sample arrays."""
    n1, n2, nt = args.sizes
    separate_paths = (args.m_csv, args.x_csv, args.y_csv)
    if any(path is not None for path in separate_paths):
        if args.samples_csv is not None or args.example is not None:
            raise ValueError("separate --m-csv/--x-csv/--y-csv inputs cannot be combined with --samples-csv or --example")
        if not all(path is not None for path in separate_paths):
            raise ValueError("separate input mode requires --m-csv, --x-csv, and --y-csv")
        m = read_samples(args.m_csv.expanduser(), nt)
        x = read_samples(args.x_csv.expanduser(), n1)
        y = read_samples(args.y_csv.expanduser(), n2)
    elif args.example in ("simple-gaussian", "toy"):
        m, x, y = simple_gaussian_samples(args.num_samples, args.seed)
    elif args.samples_csv is not None:
        samples = read_samples(args.samples_csv.expanduser())
        m, x, y = split_combined_samples(samples, args.sizes)
    else:
        raise ValueError("provide --samples-csv, --example toy, or all of --m-csv/--x-csv/--y-csv")

    sample_counts = {m.shape[0], x.shape[0], y.shape[0]}
    if len(sample_counts) != 1:
        raise ValueError(f"m/x/y sample counts must match, got {m.shape[0]}, {x.shape[0]}, {y.shape[0]}")
    return m, x, y


def validate_training_args(args: argparse.Namespace) -> None:
    """Validate Flow-PID training hyperparameters."""
    if args.n_flows < 1:
        raise ValueError("--n-flows must be positive")
    if args.n_epochs < 1:
        raise ValueError("--n-epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")


def main() -> int:
    """Load raw samples, train flow-pid's estimator, and save PID components."""
    try:
        args = simple_example_args() if len(sys.argv) == 1 else parse_args()
        validate_training_args(args)
        if args.example in ("simple-gaussian", "toy") and args.case == "InputSamples":
            args.case = SIMPLE_GAUSSIAN_CASE
        m, x, y = load_input_arrays(args)

        print(f"Loaded samples: M{m.shape}, X{x.shape}, Y{y.shape}")
        print("Calling external/flow_pid/pid/flow_pid.py: flow_pid")
        flow_pid = load_flow_pid()
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="flow_pid_training_") as temp_dir:
            try:
                os.chdir(temp_dir)
                values = flow_pid(
                    m,
                    x,
                    y,
                    n_flows=args.n_flows,
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    encoder=None,
                    verbose=args.verbose,
                    ret_t_sigt=False,
                    device=args.device,
                )
            finally:
                os.chdir(original_cwd)

        row = pid_result_row(values, args.case, "Flow")
        if args.output is None:
            print_pid_result(row, BASE_PID_COLUMNS)
        else:
            write_pid_row(row, args.output.expanduser(), BASE_PID_COLUMNS)
            print(f"Wrote Flow-PID result to {args.output.expanduser()}")
        return 0
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
