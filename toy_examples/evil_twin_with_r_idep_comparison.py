#!/usr/bin/env python3
"""Evil twin PID experiment with an added comparison to R idepGM().

This is based on the evil twin code you pasted. The new pieces are:

- import r_idep_wrapper.run_idep_from_covariance
- evil_twin_r_idep(): calls JWKay/PID/IdepGauss.R through Rscript
- print_idep_comparison_against_r(): compares Python PID atoms to R Idep atoms

The R call receives the full empirical covariance matrix ordered as
[X1, X2, T], matching idepGM(sizes, mat).
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import torch

from r_idep_wrapper import ATOMS, RIdePResult, run_idep_from_covariance

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from gpid import tilde_pid
from Partial_Information_Decomposition.Idep_multivariate_gauss import (
    Idep_multivariate_gauss,
)
from Partial_Information_Decomposition.PID_util import (
    create_cov_matrix,
    residual_rvs,
)
from utils import Tee


def empirical_covariance_matrix_torch(X1, X2, T, correction=1):
    """Compute empirical covariance matrix of (X1, X2, T) using torch."""

    Z = torch.cat([X1, X2, T], dim=1)
    return torch.cov(Z.T, correction=correction)


def check_evil_twin_covariances_torch(data, atol=5e-2, rtol=5e-2, verbose=True):
    """Check whether Sonic and Shadow empirical covariance matrices match."""

    X1_so, X2_so, T_so = data["sonic"]
    X1_sh, X2_sh, T_sh = data["shadow"]

    Sigma_so = empirical_covariance_matrix_torch(X1_so, X2_so, T_so)
    Sigma_sh = empirical_covariance_matrix_torch(X1_sh, X2_sh, T_sh)

    diff = Sigma_so - Sigma_sh
    max_abs_diff = torch.max(torch.abs(diff))
    are_equal = torch.allclose(Sigma_so, Sigma_sh, atol=atol, rtol=rtol)

    if verbose:
        print("Sonic empirical covariance:")
        print(Sigma_so)

        print("\nShadow empirical covariance:")
        print(Sigma_sh)

        print("\nDifference: Sonic - Shadow")
        print(diff)

        print("\nMax absolute difference:")
        print(max_abs_diff.item())

        print("\nAre covariance matrices close?")
        print(f"{are_equal} {'[OK]' if are_equal else '[NO]'}")

    return {
        "Sigma_sonic": Sigma_so,
        "Sigma_shadow": Sigma_sh,
        "difference": diff,
        "max_abs_difference": max_abs_diff,
        "are_equal": are_equal,
    }


def evil_twin_example_torch(generator, n, p, device="cpu", dtype=torch.float64):
    """Torch version of the evil twin example."""

    def randn_scaled(var_total):
        scale = torch.sqrt(torch.tensor(var_total / p, dtype=dtype, device=device))
        return scale * torch.randn(
            n, p, generator=generator, device=device, dtype=dtype
        )

    # Twin Sonic
    R_so = randn_scaled(0.5)
    N_so = randn_scaled(2.5)
    U1_so = randn_scaled(2.5)
    U2_so = randn_scaled(0.5)
    N_t_so = randn_scaled(1.0)

    X1_so = R_so + N_so + U1_so
    X2_so = R_so + N_so + U2_so
    T_so = R_so + U1_so + U2_so + N_t_so

    # Twin Shadow
    R_sh = randn_scaled(1.0)
    N_sh = randn_scaled(2.0)
    U1_sh = randn_scaled(2.0)
    E1_sh = randn_scaled(0.5)
    E2_sh = randn_scaled(0.5)
    E_t_sh = randn_scaled(0.5)
    N_t_sh = randn_scaled(1.0)

    X1_sh = R_sh + N_sh + U1_sh + E1_sh
    X2_sh = R_sh + N_sh + E2_sh
    T_sh = R_sh + U1_sh + N_t_sh + E_t_sh

    return {
        "sonic": (X1_so, X2_so, T_so),
        "shadow": (X1_sh, X2_sh, T_sh),
    }


def evil_twin_idep(data, on_rvs: callable = None):
    """Compute PID with the local Python Idep_multivariate_gauss code."""

    data_sonic = data["sonic"]
    data_shadow = data["shadow"]
    config = {
        "n_samples": data_sonic[0].shape[0],
        "dx1": data_sonic[0].shape[1],
        "dx2": data_sonic[1].shape[1],
        "dt": data_sonic[2].shape[1],
    }
    sources_sonic = list(data_sonic[0:2])
    target_sonic = [data_sonic[2]]
    sources_shadow = list(data_shadow[0:2])
    target_shadow = [data_shadow[2]]

    if on_rvs is not None:
        print("\nApplying on_rvs transformation to sources...")
        sources_sonic = on_rvs(sources_sonic)
        sources_shadow = on_rvs(sources_shadow)

    idep_sonic = Idep_multivariate_gauss(config,
       sources = sources_sonic, targets = target_sonic, bias_correction=False
    )
    pid_so, mi_so = idep_sonic.idep()

    idep_shadow = Idep_multivariate_gauss(
        config,
        sources = sources_shadow, targets = target_shadow, bias_correction=False
    )
    pid_sh, mi_sh = idep_shadow.idep()

    return {
        "sonic": {"pid": pid_so, "mi": mi_so},
        "shadow": {"pid": pid_sh, "mi": mi_sh},
    }


def evil_twin_r_idep(
    data,
    *,
    rscript: str | None = None,
    local_idep: str | None = "IdepGauss.R",
    correction: int = 1,
) -> dict[str, RIdePResult]:
    """Compute R idepGM() from each case's empirical covariance matrix."""

    results: dict[str, RIdePResult] = {}

    for case_name in ("sonic", "shadow"):
        X1, X2, T = data[case_name]
        sizes = (X1.shape[1], X2.shape[1], T.shape[1])

        # Important: this full Sigma is ordered as [X1, X2, T].
        # R idepGM() will internally extract blocks and whiten them.
        Sigma = empirical_covariance_matrix_torch(X1, X2, T, correction=correction)

        results[case_name] = run_idep_from_covariance(
            Sigma,
            sizes,
            rscript=rscript,
            local_idep=local_idep,
        )

    return results


def evil_twin_tilde_pid(data):
    """Compute PID using the BROJA-style Gaussian tilde PID code."""

    data_sonic = data["sonic"]
    data_shadow = data["shadow"]

    # Local code uses [predictor1, predictor2, target].
    # gpid assumes [target, predictor1, predictor2].
    data_sonic_reordered = [data_sonic[2], data_sonic[0], data_sonic[1]]
    data_shadow_reordered = [data_shadow[2], data_shadow[0], data_shadow[1]]

    dict_cov_sonic = create_cov_matrix(data_sonic_reordered)
    dict_cov_shadow = create_cov_matrix(data_shadow_reordered)

    cov_sonic = dict_cov_sonic["full_cov"].cpu().numpy()
    cov_shadow = dict_cov_shadow["full_cov"].cpu().numpy()

    dm_sonic = data_sonic_reordered[0].shape[1]
    dx_sonic = data_sonic_reordered[1].shape[1]
    dy_sonic = data_sonic_reordered[2].shape[1]

    dm_shadow = data_shadow_reordered[0].shape[1]
    dx_shadow = data_shadow_reordered[1].shape[1]
    dy_shadow = data_shadow_reordered[2].shape[1]

    # (imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si)
    output_so = tilde_pid.exact_gauss_tilde_pid(
        cov_sonic,
        dm_sonic,
        dx_sonic,
        dy_sonic,
        unbiased=True,
        sample_size=data_sonic_reordered[0].shape[0],
    )
    output_sh = tilde_pid.exact_gauss_tilde_pid(
        cov_shadow,
        dm_shadow,
        dx_shadow,
        dy_shadow,
        unbiased=True,
        sample_size=data_shadow_reordered[0].shape[0],
    )

    pid_so = {
        "red": output_so[7],
        "unq1": output_so[5],
        "unq2": output_so[6],
        "syn": output_so[8],
    }
    pid_sh = {
        "red": output_sh[7],
        "unq1": output_sh[5],
        "unq2": output_sh[6],
        "syn": output_sh[8],
    }

    mi_so = {
        "I(M1;T)": output_so[0],
        "I(M2;T)": output_so[1],
        "I(M1,M2;T)": output_so[2],
    }
    mi_sh = {
        "I(M1;T)": output_sh[0],
        "I(M2;T)": output_sh[1],
        "I(M1,M2;T)": output_sh[2],
    }

    return {
        "sonic": {"pid": pid_so, "mi": mi_so},
        "shadow": {"pid": pid_sh, "mi": mi_sh},
    }


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _normalize_pid_atoms(pid: Any) -> dict[str, float] | None:
    """Normalize common PID dict/vector layouts to R atom names."""

    if isinstance(pid, Mapping):
        lowered = {str(key).lower(): value for key, value in pid.items()}
        candidates = {
            "unique_X1": ("unique_x1", "unq1", "uix", "unique1", "unique_1"),
            "unique_X2": ("unique_x2", "unq2", "uiy", "unique2", "unique_2"),
            "redundancy": ("redundancy", "red", "ri"),
            "synergy": ("synergy", "syn", "si"),
        }
        normalized = {}
        for atom, keys in candidates.items():
            for key in keys:
                if key in lowered:
                    normalized[atom] = _scalar(lowered[key])
                    break
        return normalized if set(normalized) == set(ATOMS) else None

    if isinstance(pid, (list, tuple)) and len(pid) == 4:
        return {atom: _scalar(value) for atom, value in zip(ATOMS, pid)}

    if hasattr(pid, "detach") and pid.numel() == 4:
        values = pid.detach().cpu().reshape(-1).tolist()
        return {atom: float(value) for atom, value in zip(ATOMS, values)}

    return None


def print_r_idep_results(r_results: Mapping[str, RIdePResult]) -> None:
    print("\n" + "=" * 60)
    print("R IDEPGAUSS.R RESULTS".center(60))
    print("=" * 60)

    for name in ("sonic", "shadow"):
        result = r_results[name]
        print(f"\n{name.title()}:")
        print("  R Idep PID:")
        for atom in ATOMS:
            print(f"    {atom:12s}: {result.idep[atom]:.6f}")
        print("  R MMI PID:")
        for atom in ATOMS:
            print(f"    {atom:12s}: {result.mmi[atom]:.6f}")


def print_idep_comparison_against_r(
    label: str,
    python_results: Mapping[str, Mapping[str, Any]],
    r_results: Mapping[str, RIdePResult],
) -> None:
    print("\n" + "=" * 60)
    print(label.center(60))
    print("=" * 60)

    for case_name in ("sonic", "shadow"):
        py_pid = _normalize_pid_atoms(python_results[case_name]["pid"])
        r_pid = r_results[case_name].idep

        print(f"\n{case_name.title()}:")
        if py_pid is None:
            print("  Could not normalize Python PID output automatically.")
            print(f"  Raw Python PID: {python_results[case_name]['pid']}")
            print(f"  R Idep PID: {r_pid}")
            continue

        for atom in ATOMS:
            py_value = py_pid[atom]
            r_value = r_pid[atom]
            print(
                f"  {atom:12s}: Python {py_value: .6f} | "
                f"R {r_value: .6f} | diff {py_value - r_value: .6f}"
            )


def print_pid_results(title: str, pid_results: Mapping[str, Mapping[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)

    for name in ("Sonic", "Shadow"):
        key = name.lower()
        pid = pid_results[key]["pid"]
        mi = pid_results[key]["mi"]

        print(f"\n{name}:")
        print("  PID Decomposition:")
        print(f"    Redundancy (Red):     {_scalar(pid['red']):.4f}")
        print(f"    Unique 1 (Unq1):      {_scalar(pid['unq1']):.4f}")
        print(f"    Unique 2 (Unq2):      {_scalar(pid['unq2']):.4f}")
        print(f"    Synergy (Syn):        {_scalar(pid['syn']):.4f}")
        print("  Mutual Information:")
        print(f"    I(M1;T):              {_scalar(mi['I(M1;T)']):.4f}")
        print(f"    I(M2;T):              {_scalar(mi['I(M2;T)']):.4f}")
        print(f"    I(M1,M2;T):           {_scalar(mi['I(M1,M2;T)']):.4f}")


def print_sonic_shadow_comparison(pid_results: Mapping[str, Mapping[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("COMPARISON: SONIC vs SHADOW".center(60))
    print("=" * 60)

    sonic_pid = pid_results["sonic"]["pid"]
    shadow_pid = pid_results["shadow"]["pid"]
    sonic_mi = pid_results["sonic"]["mi"]
    shadow_mi = pid_results["shadow"]["mi"]

    print("\nPID Decomposition Differences:")
    for key, label in (
        ("red", "Redundancy (Red)"),
        ("unq1", "Unique 1 (Unq1)"),
        ("unq2", "Unique 2 (Unq2)"),
        ("syn", "Synergy (Syn)"),
    ):
        sonic_value = _scalar(sonic_pid[key])
        shadow_value = _scalar(shadow_pid[key])
        print(
            f"  {label:21s}: {sonic_value:.4f} - {shadow_value:.4f} = "
            f"{sonic_value - shadow_value:.4f}"
        )

    print("\nMutual Information Differences:")
    for key in ("I(M1;T)", "I(M2;T)", "I(M1,M2;T)"):
        sonic_value = _scalar(sonic_mi[key])
        shadow_value = _scalar(shadow_mi[key])
        print(
            f"  {key:21s}: {sonic_value:.4f} - {shadow_value:.4f} = "
            f"{sonic_value - shadow_value:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--rscript", default=None)
    parser.add_argument("--local-idep", default="IdepGauss.R")
    parser.add_argument("--skip-r-idep", action="store_true")
    parser.add_argument("--skip-python-idep", action="store_true")
    parser.add_argument("--log", default="tilde_Evil_Twin_pidv.log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log = open(args.log, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)

    torch.set_default_dtype(torch.float64)

    g = torch.Generator().manual_seed(args.seed)
    data = evil_twin_example_torch(g, args.n, args.p)

    check_evil_twin_covariances_torch(
        data,
        atol=args.atol,
        rtol=args.rtol,
    )

    res_rv = None
    # Example if you want residualized predictors:
    # res_rv = partial(residual_rvs, target_index=1)

    python_idep_results = None
    if not args.skip_python_idep:
        python_idep_results = evil_twin_idep(data, on_rvs=res_rv)

    gpid_results = evil_twin_tilde_pid(data)
    print_pid_results("TILDE PID RESULTS", gpid_results)
    print_sonic_shadow_comparison(gpid_results)

    if not args.skip_r_idep:
        r_results = evil_twin_r_idep(
            data,
            rscript=args.rscript,
            local_idep=args.local_idep,
        )
        print_r_idep_results(r_results)

        if python_idep_results is not None:
            print_idep_comparison_against_r(
                "PYTHON IDEP vs R IDEPGAUSS.R",
                python_idep_results,
                r_results,
            )

        print_idep_comparison_against_r(
            "TILDE PID vs R IDEPGAUSS.R",
            gpid_results,
            r_results,
        )

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
