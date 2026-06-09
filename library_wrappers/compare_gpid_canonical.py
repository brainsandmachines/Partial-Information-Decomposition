#!/usr/bin/env python3
"""Compare GPID canonical examples by direct GPID calls and PID_calc wrappers."""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import logging
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external.gpid.src.gpid.estimate import approx_pid_from_cov
from external.gpid.src.gpid.tilde_pid import exact_gauss_tilde_pid
from Partial_Information_Decomposition.PID_calc import delta_wrapper, pid_tilde_wrapper


COMPONENTS = ("imx", "imy", "imxy", "uix", "uiy", "ri", "si")
GPID_METHODS = {
    "tilde": exact_gauss_tilde_pid,
    "delta": lambda cov, dm, dx, dy: approx_pid_from_cov(cov, dm, dx, dy, verbose=False),
}


def canonical_examples() -> list[dict[str, object]]:
    """Build the canonical covariance examples from external/gpid/scripts.

    Inputs:
        None.

    Outputs:
        list[dict[str, object]], example metadata plus covariance matrices in
        GPID/PID_calc order [target, source1, source2].
    """
    dm = dx = dy = 1
    cases: list[dict[str, object]] = []

    def add(desc: str, case_id: int, cov: np.ndarray, sigma: float = np.nan, rho: float = np.nan) -> None:
        """Append one canonical example to the local case list.

        Inputs:
            desc: str, canonical family name.
            case_id: int, index inside the family.
            cov: np.ndarray, covariance matrix in [target, source1, source2] order.
            sigma: float, sigma_y__x value when relevant.
            rho: float, source-noise correlation when relevant.

        Outputs:
            None.
        """
        cases.append({"desc": desc, "case_id": case_id, "dm": dm, "dx": dx, "dy": dy, "sigma_y__x": sigma, "rho": rho, "cov": cov})

    sigm = np.eye(1)
    hx = np.array([[1]])
    hyx = np.array([[1]])
    hy = hyx @ hx
    covx = hx @ sigm @ hx.T + np.eye(1)
    for i, sigma2 in enumerate(np.r_[0, np.logspace(0, 2, 9)]):
        sigy_x = sigma2 * np.eye(1)
        cov = np.block([[sigm, sigm @ hx.T, sigm @ hy.T], [hx @ sigm, covx, covx @ hyx.T], [hy @ sigm, hyx @ covx, hyx @ covx @ hyx.T + sigy_x]])
        add("uix+ri", i, cov, sigma=float(sigma2))

    rho_vals = np.r_[np.linspace(0, 1, 10, endpoint=False), 0.99]
    for desc, hy in (("uix+si", np.array([[0]])), ("ri+si", np.array([[1]]))):
        for i, rho in enumerate(rho_vals):
            sigw = np.array([[rho]])
            cov = np.block(
                [
                    [sigm, sigm @ hx.T, sigm @ hy.T],
                    [hx @ sigm, hx @ sigm @ hx.T + np.eye(1), hx @ sigm @ hy.T + sigw],
                    [hy @ sigm, hy @ sigm @ hx.T + sigw.T, hy @ sigm @ hy.T + np.eye(1)],
                ]
            )
            add(desc, i, cov, rho=float(rho))

    return cases


def unpack_gpid(values: tuple[float, ...]) -> dict[str, float]:
    """Convert one GPID return tuple into named components.

    Inputs:
        values: tuple[float, ...], GPID estimator output.

    Outputs:
        dict[str, float], values keyed by imx, imy, imxy, uix, uiy, ri, and si.
    """
    imx, imy, imxy = values[:3]
    uix, uiy, ri, si = values[-4:]
    return dict(zip(COMPONENTS, map(float, (imx, imy, imxy, uix, uiy, ri, si))))


def unpack_wrapper(pid: dict[str, float], mi: dict[str, float]) -> dict[str, float]:
    """Convert one PID_calc wrapper output into named components.

    Inputs:
        pid: dict[str, float], wrapper PID dictionary.
        mi: dict[str, float], wrapper mutual information dictionary.

    Outputs:
        dict[str, float], values keyed by imx, imy, imxy, uix, uiy, ri, and si.
    """
    values = (mi["bi_mi_1"], mi["bi_mi_2"], mi["tri_mi"], pid["unq1"], pid["unq2"], pid["red"], pid["syn"])
    return dict(zip(COMPONENTS, map(float, values)))


def pid_calc(method: str, cov: np.ndarray, dm: int, dx: int, dy: int) -> dict[str, float]:
    """Run one canonical covariance through the matching PID_calc wrapper.

    Inputs:
        method: str, either "tilde" or "delta".
        cov: np.ndarray, covariance matrix in [target, source1, source2] order.
        dm: int, target/message dimension.
        dx: int, first source dimension.
        dy: int, second source dimension.

    Outputs:
        dict[str, float], wrapper result keyed by component name.
    """
    config = {"dt": dm, "dx1": dx, "dx2": dy, "n_samples": 0, "bias_correction": False}
    cov_tensor = torch.as_tensor(cov, dtype=torch.float64)
    wrapper = pid_tilde_wrapper if method == "tilde" else delta_wrapper
    pid, mi = wrapper(config, sources=None, target=None, covariance=cov_tensor, rng=torch.Generator().manual_seed(56), on_rvs=None)
    return unpack_wrapper(pid, mi)


def compare() -> list[dict[str, object]]:
    """Compare direct GPID calls with PID_calc wrappers for all canonical examples.

    Inputs:
        None.

    Outputs:
        list[dict[str, object]], long-form rows with direct values, wrapper values,
        and absolute differences.
    """
    rows = []
    for ex in canonical_examples():
        cov = np.asarray(ex["cov"], dtype=float)
        dims = int(ex["dm"]), int(ex["dx"]), int(ex["dy"])
        for method, gpid_func in GPID_METHODS.items():
            direct = unpack_gpid(gpid_func(cov, *dims))
            wrapped = pid_calc(method, cov, *dims)
            for component in COMPONENTS:
                rows.append(
                    {
                        "desc": ex["desc"],
                        "case_id": ex["case_id"],
                        "sigma_y__x": ex["sigma_y__x"],
                        "rho": ex["rho"],
                        "method": method,
                        "component": component,
                        "direct_gpid": direct[component],
                        "pid_calc_wrapper": wrapped[component],
                        "abs_diff": abs(direct[component] - wrapped[component]),
                    }
                )
    return rows


def print_summary(rows: list[dict[str, object]]) -> None:
    """Print the maximum absolute difference per method and component.

    Inputs:
        rows: list[dict[str, object]], comparison rows from compare.

    Outputs:
        None.
    """
    summary = {}
    for row in rows:
        key = row["method"], row["component"]
        summary[key] = max(summary.get(key, 0.0), float(row["abs_diff"]))

    print(f"{'method':<8} {'component':<10} {'max_abs_diff':>14}")
    for (method, component), value in sorted(summary.items()):
        print(f"{method:<8} {component:<10} {value:>14.12g}")


def main() -> int:
    """Run the canonical GPID comparison command.

    Inputs:
        None.

    Outputs:
        int, process exit code where 0 means every difference is within tolerance.
    """
    parser = argparse.ArgumentParser(description="Compare direct GPID and PID_calc on canonical examples.")
    parser.add_argument("--output", type=Path, help="Optional CSV path for all comparison rows.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Allowed absolute difference.")
    parser.add_argument("--verbose", action="store_true", help="Show GPID/CVXPY solver logs and warnings.")
    args = parser.parse_args()

    previous_logging_disabled = logging.root.manager.disable
    if not args.verbose:
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)
    quiet_output = redirect_stdout(io.StringIO()) if not args.verbose else nullcontext()
    quiet_errors = redirect_stderr(io.StringIO()) if not args.verbose else nullcontext()
    try:
        with quiet_output, quiet_errors:
            rows = compare()
    finally:
        logging.disable(previous_logging_disabled)

    print_summary(rows)
    max_diff = max(float(row["abs_diff"]) for row in rows)
    print(f"\nmax_abs_diff: {max_diff:.12g}")
    print(f"within_atol_{args.atol}: {max_diff <= args.atol}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote: {args.output}")

    return 0 if max_diff <= args.atol else 1


if __name__ == "__main__":
    raise SystemExit(main())
