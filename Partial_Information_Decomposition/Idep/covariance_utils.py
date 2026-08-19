"""Reusable covariance construction and shrinkage helpers for Idep code."""

from __future__ import annotations

import torch

from Partial_Information_Decomposition.PID_util import (
    create_cov_matrix,
    para_whiten_block,
    whiten_block,
)
from Partial_Information_Decomposition.mi_functions import calcualte_mi

def build_m8_terms(
    config: dict,
    covariance_blocks: dict,
    whiten: str = "whiten_ver",
    para: bool = False,
) -> dict[str, torch.Tensor]:
    """Construct M8 covariance terms from covariance blocks.

    Inputs:
        config: dict containing ``n0``, ``n1``, ``n2``, and optional ``device``.
        covariance_blocks: dict returned by a PID covariance-block helper.
        whiten: str in ``{'False', 'whiten_ver', 'True'}`` selecting normalization.
        para: bool indicating batched ``(B, D, D)`` construction.

    Outputs:
        dict containing P, Q, R cross-blocks and the reconstructed M8 ``Sigma``.
    """

    n0, n1, n2 = config["n0"], config["n1"], config["n2"]
    device = config.get("device", "cpu")
    cross_1_t_key = "cross_x1_t" if "cross_x1_t" in covariance_blocks else "cross_x1_xt"
    cross_2_t_key = "cross_x2_t" if "cross_x2_t" in covariance_blocks else "cross_x2_xt"
    target_covariance_key = "cov_t" if "cov_t" in covariance_blocks else "cov_xt"
    if whiten == "False":
        p_block = covariance_blocks["cross_x1_x2"]
        q_block = covariance_blocks[cross_1_t_key]
        r_block = covariance_blocks[cross_2_t_key]
    elif whiten == "whiten_ver":
        whitening_function = para_whiten_block if para else whiten_block
        p_block = whitening_function(
            covariance_blocks["cov_x1"],
            covariance_blocks["cross_x1_x2"],
            covariance_blocks["cov_x2"],
        )
        q_block = whitening_function(
            covariance_blocks["cov_x1"],
            covariance_blocks[cross_1_t_key],
            covariance_blocks[target_covariance_key],
        )
        r_block = whitening_function(
            covariance_blocks["cov_x2"],
            covariance_blocks[cross_2_t_key],
            covariance_blocks[target_covariance_key],
        )
    elif whiten == "True":
        p_block = covariance_blocks["cross_x1_x2"]
        q_block = covariance_blocks[cross_1_t_key]
        r_block = covariance_blocks[cross_2_t_key]
    else:
        raise ValueError(f"Unsupported whiten mode: {whiten!r}.")

    if not para:
        row_1 = torch.cat([torch.eye(n0, device=device), p_block, q_block], dim=1)  # block widths -> (n0, D)
        row_2 = torch.cat([p_block.T, torch.eye(n1, device=device), r_block], dim=1)  # block widths -> (n1, D)
        row_3 = torch.cat([q_block.T, r_block.T, torch.eye(n2, device=device)], dim=1)  # block widths -> (n2, D)
        sigma = (
            torch.cat([row_1, row_2, row_3], dim=0)  # rows (n0, D), (n1, D), (n2, D) -> (D, D)
            if whiten != "False"
            else covariance_blocks["full_cov"]
        )
    else:
        if covariance_blocks["full_cov"].ndim != 3:
            raise ValueError("Parallel M8 construction expects full_cov shaped (B, D, D).")
        batch_size = covariance_blocks["cov_x1"].shape[0]
        row_1 = torch.cat(
            [torch.eye(n0, device=device).repeat(batch_size, 1, 1), p_block, q_block],
            dim=2,
        )  # three batched blocks -> (B, n0, D)
        row_2 = torch.cat(
            [p_block.mT, torch.eye(n1, device=device).repeat(batch_size, 1, 1), r_block],
            dim=2,
        )  # three batched blocks -> (B, n1, D)
        row_3 = torch.cat(
            [q_block.mT, r_block.mT, torch.eye(n2, device=device).repeat(batch_size, 1, 1)],
            dim=2,
        )  # three batched blocks -> (B, n2, D)
        sigma = (
            torch.cat([row_1, row_2, row_3], dim=1)  # batched block rows -> (B, D, D)
            if whiten != "False"
            else covariance_blocks["full_cov"]
        )
    return {"P": p_block, "Q": q_block, "R": r_block, "Sigma": sigma}


def build_m7_terms(
    config: dict,
    covariance_blocks: dict,
    whiten: str = "whiten_ver",
    para: bool = False,
) -> dict[str, torch.Tensor]:
    """Construct M7 covariance terms from covariance blocks.

    Inputs:
        config: dict containing source/target dimensions and optional device.
        covariance_blocks: dict returned by a PID covariance-block helper.
        whiten: str in ``{'False', 'whiten_ver', 'True'}`` selecting normalization.
        para: bool indicating batched ``(B, D, D)`` construction.

    Outputs:
        dict containing M7 P, Q, R cross-blocks and reconstructed ``Sigma``.
    """

    dx1, dx2, dt = config["dx1"], config["dx2"], config["dt"]
    device = config.get("device", "cpu")
    cross_1_t_key = "cross_x1_t" if "cross_x1_t" in covariance_blocks else "cross_x1_xt"
    cross_2_t_key = "cross_x2_t" if "cross_x2_t" in covariance_blocks else "cross_x2_xt"
    target_covariance_key = "cov_t" if "cov_t" in covariance_blocks else "cov_xt"
    if whiten == "False":
        q_block = covariance_blocks[cross_1_t_key]
        r_block = covariance_blocks[cross_2_t_key]
        target_covariance = covariance_blocks[target_covariance_key]
    elif whiten == "whiten_ver":
        q_block = para_whiten_block(
            covariance_blocks["cov_x1"],
            covariance_blocks[cross_1_t_key],
            covariance_blocks[target_covariance_key],
        )
        r_block = para_whiten_block(
            covariance_blocks["cov_x2"],
            covariance_blocks[cross_2_t_key],
            covariance_blocks[target_covariance_key],
        )
        target_covariance = torch.eye(dt, device=device)
    elif whiten == "True":
        q_block = covariance_blocks[cross_1_t_key]
        r_block = covariance_blocks[cross_2_t_key]
        target_covariance = covariance_blocks[target_covariance_key]
    else:
        raise ValueError(f"Unsupported whiten mode: {whiten!r}.")

    if not para:
        covariance_1 = torch.eye(dx1, device=device) if whiten != "False" else covariance_blocks["cov_x1"]
        covariance_2 = torch.eye(dx2, device=device) if whiten != "False" else covariance_blocks["cov_x2"]
        target_covariance = torch.eye(dt, device=device) if whiten != "False" else target_covariance
        target_inverse = torch.linalg.inv(target_covariance).to(dtype=q_block.dtype, device=device)
        p_block = q_block @ target_inverse @ r_block.T
        row_1 = torch.cat([covariance_1, p_block, q_block], dim=1)  # block widths -> (dx1, D)
        row_2 = torch.cat([p_block.T, covariance_2, r_block], dim=1)  # block widths -> (dx2, D)
        row_3 = torch.cat([q_block.T, r_block.T, target_covariance], dim=1)  # block widths -> (dt, D)
        sigma = torch.cat([row_1, row_2, row_3], dim=0)  # block rows -> (D, D)
    else:
        if covariance_blocks["full_cov"].ndim != 3:
            raise ValueError("Parallel M7 construction expects full_cov shaped (B, D, D).")
        batch_size = covariance_blocks["cov_x1"].shape[0]
        covariance_1 = torch.eye(dx1, device=device).repeat(batch_size, 1, 1) if whiten != "False" else covariance_blocks["cov_x1"]
        covariance_2 = torch.eye(dx2, device=device).repeat(batch_size, 1, 1) if whiten != "False" else covariance_blocks["cov_x2"]
        target_covariance = torch.eye(dt, device=device).repeat(batch_size, 1, 1) if whiten != "False" else target_covariance
        target_inverse = torch.linalg.inv(target_covariance).to(dtype=q_block.dtype, device=device)
        p_block = q_block @ target_inverse @ r_block.mT
        row_1 = torch.cat([covariance_1, p_block, q_block], dim=2)  # batched blocks -> (B, dx1, D)
        row_2 = torch.cat([p_block.mT, covariance_2, r_block], dim=2)  # batched blocks -> (B, dx2, D)
        row_3 = torch.cat([q_block.mT, r_block.mT, target_covariance], dim=2)  # batched blocks -> (B, dt, D)
        sigma = torch.cat([row_1, row_2, row_3], dim=1)  # batched block rows -> (B, D, D)
    return {"P": p_block, "Q": q_block, "R": r_block, "Sigma": sigma}


def create_cov_m8(
    config: dict,
    p_block: torch.Tensor,
    q_block: torch.Tensor,
    r_block: torch.Tensor,
) -> torch.Tensor:
    """Create an M8 covariance from P, Q, and R cross-blocks.

    Inputs:
        config: dict containing ``dx1``, ``dx2``, ``dt``, ``device``, and ``ver``.
        p_block: torch.Tensor shaped ``(dx1, dx2)``.
        q_block: torch.Tensor shaped ``(dx1, dt)``.
        r_block: torch.Tensor shaped ``(dx2, dt)``.

    Outputs:
        torch.Tensor covariance shaped ``(dx1 + dx2 + dt, dx1 + dx2 + dt)``.
    """

    dx1, dx2, dt = config["dx1"], config["dx2"], config["dt"]
    version = config.get("ver", "raw")
    if version == "red":
        q_block = p_block @ r_block
        if q_block.shape != (dx1, dt):
            raise ValueError("Shape mismatch for Q in red version.")
    if version == "only_unq1_zero":
        raise NotImplementedError(
            "The legacy only_unq1_zero covariance called an undefined optimizer."
        )
    row_1 = torch.cat([torch.eye(dx1, device=config["device"]), p_block, q_block], dim=1)  # blocks -> (dx1, D)
    row_2 = torch.cat([p_block.T, torch.eye(dx2, device=config["device"]), r_block], dim=1)  # blocks -> (dx2, D)
    row_3 = torch.cat([q_block.T, r_block.T, torch.eye(dt, device=config["device"])], dim=1)  # blocks -> (dt, D)
    return torch.cat([row_1, row_2, row_3], dim=0)  # block rows -> (D, D)


def create_m7_cov(
    config: dict,
    cov_m8: torch.Tensor,
    whitening_normalize: bool = True,
) -> torch.Tensor:
    """Construct the corresponding M7 covariance from an M8 covariance.

    Inputs:
        config: dict containing ``dx1``, ``dx2``, ``dt``, and ``device``.
        cov_m8: torch.Tensor covariance ordered ``[X1, X2, T]``.
        whitening_normalize: bool selecting identity diagonal blocks.

    Outputs:
        torch.Tensor M7 covariance with the same ``(D, D)`` shape as ``cov_m8``.
    """

    blocks = create_cov_matrix(
        Sigma=cov_m8,
        dims=[config["dx1"], config["dx2"], config["dt"]],
    )
    diagonal_1 = torch.eye(config["dx1"], device=config["device"]) if whitening_normalize else blocks["cov_x1"]
    diagonal_2 = torch.eye(config["dx2"], device=config["device"]) if whitening_normalize else blocks["cov_x2"]
    diagonal_t = torch.eye(config["dt"], device=config["device"]) if whitening_normalize else blocks["cov_t"]
    if whitening_normalize:
        q_block = whiten_block(blocks["cov_x1"], blocks["cross_x1_t"], blocks["cov_t"])
        r_block = whiten_block(blocks["cov_x2"], blocks["cross_x2_t"], blocks["cov_t"])
        p_block = q_block @ r_block.T
    else:
        q_block = blocks["cross_x1_t"]
        r_block = blocks["cross_x2_t"]
        p_block = q_block @ torch.linalg.inv(blocks["cov_t"]) @ r_block.T
    row_1 = torch.cat([diagonal_1, p_block, q_block], dim=1)  # blocks -> (dx1, D)
    row_2 = torch.cat([p_block.T, diagonal_2, r_block], dim=1)  # blocks -> (dx2, D)
    row_3 = torch.cat([q_block.T, r_block.T, diagonal_t], dim=1)  # blocks -> (dt, D)
    return torch.cat([row_1, row_2, row_3], dim=0)  # block rows -> (D, D)


def make_random_true_cov(
    config: dict,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a positive-definite M8 covariance and corresponding M7 model.

    Inputs:
        config: dict defining dimensions, block scales, device, regime, and retries.
        rng: optional torch.Generator controlling random block construction.

    Outputs:
        tuple[torch.Tensor, torch.Tensor] containing M8 and M7 covariances,
        each shaped ``(dx1 + dx2 + dt, dx1 + dx2 + dt)``.
    """

    q_scale = float(config["q_scale"])
    r_scale = float(config["r_scale"])
    dx1, dx2, dt = config["dx1"], config["dx2"], config["dt"]
    device = config["device"]
    mode = config.get("mode", "exact_m7")
    delta = float(config.get("delta", 0.0))
    max_tries = int(config.get("max_tries", 1000))
    delta_margin = float(config.get("delta_margin", 1e-3))
    alpha = float(config.get("alpha", 0.5))
    version = config.get("ver", "raw")
    last_covariances: tuple[torch.Tensor, torch.Tensor] | None = None

    for _ in range(max_tries):
        a_matrix = torch.randn((dx1, dt), generator=rng, dtype=torch.float64, device=device)  # dimensions -> (dx1, dt)
        b_matrix = torch.randn((dx2, dt), generator=rng, dtype=torch.float64, device=device)  # dimensions -> (dx2, dt)
        c_matrix = torch.randn((dx1, dx2), generator=rng, dtype=torch.float64, device=device)  # dimensions -> (dx1, dx2)
        a_norm = torch.linalg.norm(a_matrix, ord=2)
        b_norm = torch.linalg.norm(b_matrix, ord=2)
        c_norm = torch.linalg.norm(c_matrix, ord=2)
        if a_norm == 0 or b_norm == 0 or c_norm == 0:
            continue
        q_block = q_scale * a_matrix / a_norm
        r_block = r_scale * b_matrix / b_norm
        p_m7 = q_block @ r_block.T
        if mode == "exact_m7":
            p_block = p_m7
        elif mode in {"m8_side", "m7_side"}:
            p_block = alpha * p_m7 + delta * c_matrix / c_norm
        else:
            raise ValueError(f"Unknown covariance mode: {mode!r}.")

        cov_m8 = create_cov_m8(config, p_block, q_block, r_block)
        if torch.min(torch.linalg.eigvalsh(cov_m8)) <= 1e-10:
            continue
        cov_m7 = create_m7_cov(config, cov_m8, whitening_normalize=True)
        if torch.min(torch.linalg.eigvalsh(cov_m7)) <= 1e-10:
            continue
        last_covariances = (cov_m8, cov_m7)
        if mode == "exact_m7":
            return last_covariances
        try:
            terms_m8 = {"P": p_block, "Q": q_block, "R": r_block, "Sigma": cov_m8}
            terms_m7 = {"P": p_m7, "Q": q_block, "R": r_block, "Sigma": cov_m7}
            mi_m8 = calcualte_mi(config, terms_m8, term="mi_tri")["mi_tri"]
            mi_m7 = calcualte_mi(config, terms_m7, term="mi_tri")["mi_tri"]
            delta_mi = mi_m8 - mi_m7
        except RuntimeError:
            continue
        if version == "m8_side" and delta_mi < -delta_margin:
            return last_covariances
        if version == "m7_side" and delta_mi > delta_margin:
            return last_covariances

    if last_covariances is None:
        raise RuntimeError("Failed to construct positive-definite M7/M8 covariances.")
    return last_covariances
