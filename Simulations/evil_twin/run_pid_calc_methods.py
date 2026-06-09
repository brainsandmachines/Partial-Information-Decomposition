"""Run PID_calc methods on the evil-twin covariance example across seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Simulations.evil_twin.evil_twin_pid_batch_utils import DEFAULT_METHODS, run_evil_twin_pid_sweep


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evil-twin PID sweep.

    Inputs:
        No inputs.

    Outputs:
        argparse.Namespace, parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100)), help="Seeds to run.")
    parser.add_argument("--n", type=int, default=500, help="Number of samples per seed.")
    parser.add_argument("--p", type=int, default=30, help="Dimension of each random variable.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help="PID_calc methods to run.")
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_results/evil_twin_pid"))
    parser.add_argument("--csv-prefix", default="evil_twin_pid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flow-epochs", type=int, default=50)
    parser.add_argument("--flow-verbose", action="store_true")
    return parser.parse_args()


def main() -> dict:
    """Run the command-line evil-twin PID sweep.

    Inputs:
        No inputs.

    Outputs:
        dict, nested in-memory results keyed by seed, method, and twin.
    """
    args = parse_args()
    return run_evil_twin_pid_sweep(
        seeds=args.seeds,
        n=args.n,
        p=args.p,
        methods=tuple(args.methods),
        output_dir=args.output_dir,
        device=args.device,
        flow_epochs=args.flow_epochs,
        flow_verbose=args.flow_verbose,
        csv_prefix=args.csv_prefix,
    )


if __name__ == "__main__":
    main()
