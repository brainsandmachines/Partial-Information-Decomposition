"""Equicorrelation examples for the Idep multivariate Gaussian solvers."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from Partial_Information_Decomposition.Idep.Idep_multivariate_gauss import (
    Idep_multivariate_gauss,
)
from Partial_Information_Decomposition.jacknife_Idep_multivariate_gauss import (
    JackknifeIdepMultivariateGauss,
)
from Partial_Information_Decomposition.Idep_Simulations.parallel_Idep_multivariate_gauss import (
    para_Idep_multivariate_gauss,
)


EXAMPLES = [
    ((3, 4, 3), (-0.15, 0.15, 0.15), {"unq1": 0.1227, "unq2": 0.1865, "red": 0.0406, "syn": 2.4772}),
    ((4, 4, 2), (-0.2, -0.2, 0.3), {"unq1": 0.0893, "unq2": 0.7293, "red": 0.1889, "syn": 0.0087}),
    ((4, 2, 4), (-0.1, 0.15, -0.2), {"unq1": 0.2336, "unq2": 0.1899, "red": 0.0883, "syn": 0.0345}),
]


def ones(n: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.ones((n, 1), device=device, dtype=dtype)


def equicorr_blocks(
    n0: int,
    n1: int,
    n2: int,
    p: float,
    q: float,
    r: float,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the P, Q, R equicorrelation blocks from Eq. 63 in the paper."""
    P = p * (ones(n0, device, dtype) @ ones(n1, device, dtype).T)
    Q = q * (ones(n0, device, dtype) @ ones(n2, device, dtype).T)
    R = r * (ones(n1, device, dtype) @ ones(n2, device, dtype).T)
    return P, Q, R


def build_full_cov(
    n0: int,
    n1: int,
    n2: int,
    P: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build the full Model 8 covariance/correlation matrix."""
    I0 = torch.eye(n0, device=device, dtype=dtype)
    I1 = torch.eye(n1, device=device, dtype=dtype)
    I2 = torch.eye(n2, device=device, dtype=dtype)

    row1 = torch.cat([I0, P, Q], dim=1)
    row2 = torch.cat([P.T, I1, R], dim=1)
    row3 = torch.cat([Q.T, R.T, I2], dim=1)
    return torch.cat([row1, row2, row3], dim=0)


def scalarize_mapping(values: Mapping[str, object]) -> dict[str, float]:
    scalarized = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                value = value.item()
        scalarized[key] = float(value)
    return scalarized


def run_serial_example(n0: int, n1: int, n2: int, p: float, q: float, r: float):
    P, Q, R = equicorr_blocks(n0, n1, n2, p, q, r)
    sigma = build_full_cov(n0, n1, n2, P, Q, R)
    config = {"n_samples": 0, "dx1": n0, "dx2": n1, "dt": n2}
    solver = Idep_multivariate_gauss(
        config=config,
        sources=None,
        targets=None,
        cov_matrix=sigma,
        base_e=True,
        bias_correction=False,
    )
    return solver.idep(cov_matrix=sigma)


def run_jackknife_example(n0: int, n1: int, n2: int, p: float, q: float, r: float):
    P, Q, R = equicorr_blocks(n0, n1, n2, p, q, r)
    sigma = build_full_cov(n0, n1, n2, P, Q, R)
    solver = JackknifeIdepMultivariateGauss(
        sources=None,
        targets=None,
        cov_matrix=sigma,
        bias_correction=False,
        verbose=False,
    )
    solver.dim_m1, solver.dim_m2, solver.dim_t = n0, n1, n2
    solver.I0 = torch.eye(n0)
    solver.I1 = torch.eye(n1)
    solver.I2 = torch.eye(n2)
    solver.P, solver.Q, solver.R = P, Q, R
    solver.cov_matrix = sigma
    return solver.idep(cov_matrix=sigma)


def run_parallel_example(n0: int, n1: int, n2: int, p: float, q: float, r: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P, Q, R = equicorr_blocks(n0, n1, n2, p, q, r, device=device)
    sigma = build_full_cov(n0, n1, n2, P, Q, R, device=device).unsqueeze(0)
    solver = para_Idep_multivariate_gauss(
        N=1,
        device=device,
        sources=None,
        targets=None,
        cov_matrix=sigma,
        dims=[n0, n1, n2],
        bias_correction=False,
    )
    pid, mi = solver.idep()
    return scalarize_mapping(pid), scalarize_mapping(mi)


def main() -> None:
    ln2 = float(torch.log(torch.tensor(2.0, dtype=torch.float64)))

    for (n0, n1, n2), (p, q, r), expected_bits in EXAMPLES:
        pid, _ = run_serial_example(n0, n1, n2, p, q, r)
        got = {key: pid[key] for key in ["unq1", "unq2", "red", "syn"]}
        expected = {key: value * ln2 for key, value in expected_bits.items()}

        print("\n========================================")
        print(f"Example (n0,n1,n2)=({n0},{n1},{n2}), (p,q,r)=({p},{q},{r})")
        print("Got:      ", {key: f"{value:.3f}" for key, value in got.items()})
        print("Expected: ", {key: f"{value:.3f}" for key, value in expected.items()})


if __name__ == "__main__":
    main()
