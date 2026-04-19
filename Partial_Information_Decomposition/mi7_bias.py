import torch
from typing import Dict, Any


def raw_m7_bias_corrections_from_m7_dict(
    cov_dict: Dict[str, torch.Tensor],
    n_samples: int,
    eps: float = 1e-10,
    check_m7_block: bool = True,
) -> Dict[str, Any]:
    """
    Compute raw-M7 bias corrections directly from an M7 covariance dict.

    Required keys:
        cov_x0, cov_x1, cov_x2,
        cross_x0_x2, cross_x1_x2,
        joint_x0_x1

    Returns:
        BC_central
        BC_first_order
        BC_second_order
        tau
        sigma
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")

    required = [
        "cov_x0", "cov_x1", "cov_x2",
        "cross_x0_x2", "cross_x1_x2",
        "joint_x0_x1",
    ]
    missing = [k for k in required if k not in cov_dict]
    if missing:
        raise KeyError(f"Missing keys in M7 cov_dict: {missing}")

    def sym(M: torch.Tensor) -> torch.Tensor:
        return 0.5 * (M + M.T)

    def inv_spd(M: torch.Tensor) -> torch.Tensor:
        M = sym(M)
        eye = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
        try:
            L = torch.linalg.cholesky(M)
        except RuntimeError:
            L = torch.linalg.cholesky(M + eps * eye)
        return torch.cholesky_inverse(L)

    def invsqrt_spd(M: torch.Tensor) -> torch.Tensor:
        M = sym(M)
        evals, evecs = torch.linalg.eigh(M)
        evals = torch.clamp(evals, min=eps)
        return sym(evecs @ torch.diag(evals.rsqrt()) @ evecs.T)

    def block_diag(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        a, b = A.shape[0], B.shape[0]
        out = torch.zeros((a + b, a + b), dtype=A.dtype, device=A.device)
        out[:a, :a] = A
        out[a:, a:] = B
        return out

    def wishart_logdet_bias(df: int, dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if df <= dim - 1:
            raise ValueError(f"Need df > dim - 1; got df={df}, dim={dim}")
        r = torch.arange(1, dim + 1, dtype=dtype, device=device)
        return torch.digamma((torch.tensor(float(df), dtype=dtype, device=device) - r + 1.0) / 2.0).sum() + \
            dim * torch.log(torch.tensor(2.0 / float(df), dtype=dtype, device=device))

    S00 = cov_dict["cov_x0"]
    S11 = cov_dict["cov_x1"]
    STT = cov_dict["cov_x2"]
    S0T = cov_dict["cross_x0_x2"]
    S1T = cov_dict["cross_x1_x2"]
    joint_x0_x1 = sym(cov_dict["joint_x0_x1"])

    d0 = S00.shape[0]
    d1 = S11.shape[0]
    dt = STT.shape[0]
    p = d0 + d1
    m = n_samples - 1

    STT_inv = inv_spd(STT)

    # Reconstructed M7 joint block from T-links
    S01_m7 = S0T @ STT_inv @ S1T.T
    joint_expected = sym(torch.cat([
        torch.cat([S00, S01_m7], dim=1),
        torch.cat([S01_m7.T, S11], dim=1)
    ], dim=0))

    rel_err = torch.norm(joint_x0_x1 - joint_expected) / (torch.norm(joint_expected) + eps)

    Psi0 = sym(S00 - S0T @ STT_inv @ S0T.T)
    Psi1 = sym(S11 - S1T @ STT_inv @ S1T.T)

    B0 = S0T @ STT_inv
    B1 = S1T @ STT_inv

    Psi0_inv = inv_spd(Psi0)
    Psi1_inv = inv_spd(Psi1)

    A0 = B0.T @ Psi0_inv @ B0
    A1 = B1.T @ Psi1_inv @ B1

    Psi = block_diag(Psi0, Psi1)
    B = torch.cat([B0, B1], dim=0)
    Psi_invsqrt = invsqrt_spd(Psi)

    Omega = sym(m * (Psi_invsqrt @ B @ STT @ B.T @ Psi_invsqrt))
    tau = torch.trace(Omega)
    sigma = torch.trace(Omega @ Omega)

    # Exact central pieces
    BC_Psi0 = wishart_logdet_bias(df=m - dt, dim=d0, dtype=S00.dtype, device=S00.device)
    BC_Psi1 = wishart_logdet_bias(df=m - dt, dim=d1, dtype=S00.dtype, device=S00.device)

    # E[-log Lambda_J] in the central case
    BC_Lambda0 = (
        wishart_logdet_bias(df=m, dim=p, dtype=S00.dtype, device=S00.device)
        - wishart_logdet_bias(df=m - dt, dim=p, dtype=S00.dtype, device=S00.device)
    )

    # FIXED SIGNS
    BC_central = BC_Psi0 + BC_Psi1 + BC_Lambda0
    BC_first = BC_central + tau / m
    BC_second = BC_central + tau / m + ((tau * tau) / m - sigma) / (2.0 * (m - 1) * (m + 2))

    return {
        "m": m,
        "p": p,
        "dt": dt,
        "Psi0": Psi0,
        "Psi1": Psi1,
        "B0": B0,
        "B1": B1,
        "A0": A0,
        "A1": A1,
        "Omega": Omega,
        "tau": tau,
        "sigma": sigma,
        "BC_Psi0": BC_Psi0,
        "BC_Psi1": BC_Psi1,
        "BC_Lambda0": BC_Lambda0,
        "BC_central": BC_central,
        "BC_first_order": BC_first,
        "BC_second_order": BC_second,
        "m7_relative_error_joint_x0_x1": rel_err,
    }


def raw_m7_bias_corrected_logdet_from_m7_dict(
    cov_dict: Dict[str, torch.Tensor],
    n_samples: int,
    order: int = 2,
    eps: float = 1e-10,
) -> Dict[str, Any]:
    """
    Compute corrected logdet of the raw M7 joint block from an M7 dict.

    order:
        0 -> central
        1 -> first-order
        2 -> second-order
    """
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2, got {order}")

    def sym(M: torch.Tensor) -> torch.Tensor:
        return 0.5 * (M + M.T)

    def safe_logdet_spd(M: torch.Tensor) -> torch.Tensor:
        M = sym(M)
        eye = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
        sign, val = torch.linalg.slogdet(M + eps * eye)
        if sign <= 0:
            raise ValueError("Matrix is not positive definite enough for logdet.")
        return val

    joint_block = sym(cov_dict["joint_x0_x1"])
    raw_logdet = safe_logdet_spd(joint_block)

    bc = raw_m7_bias_corrections_from_m7_dict(
        cov_dict=cov_dict,
        n_samples=n_samples,
        eps=eps,
    )

    if order == 0:
        BC = bc["BC_central"]
    elif order == 1:
        BC = bc["BC_first_order"]
    else:
        BC = bc["BC_second_order"]

    return {
        "raw_m7_joint_block": joint_block,
        "raw_logdet": raw_logdet,
        "bias_correction": BC,
        "bias_corrected_logdet": raw_logdet - BC,
        **bc,
    }