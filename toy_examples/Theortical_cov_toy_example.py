import numpy as np
import torch
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from Partial_Information_Decomposition.PID_util import whiten_block


# ============================================================
# 1. Utilities
# ============================================================

def make_spd_matrix(p, rng, eig_min=0.5, eig_max=2.0):
    """
    Create a random symmetric positive definite covariance matrix.
    """
    A = rng.standard_normal((p, p))
    Q, _ = np.linalg.qr(A)

    eigvals = rng.uniform(eig_min, eig_max, size=p)

    Sigma = Q @ np.diag(eigvals) @ Q.T
    Sigma = 0.5 * (Sigma + Sigma.T)

    return Sigma


def check_spd(Sigma, name="Sigma", tol=1e-10):
    """
    Check whether a matrix is symmetric positive definite.
    """
    Sigma = np.asarray(Sigma)

    if not np.allclose(Sigma, Sigma.T, atol=tol):
        raise ValueError(f"{name} is not symmetric.")

    eigvals = np.linalg.eigvalsh(Sigma)

    if eigvals.min() <= 0:
        raise ValueError(
            f"{name} is not positive definite. "
            f"Minimum eigenvalue = {eigvals.min()}"
        )


# ============================================================
# 2. Theoretical covariance
# ============================================================

def theoretical_covariance_multivariate(
    Sigma_R,
    Sigma_U,
    Sigma_N,
    Sigma_eps,
    order=("X1", "X2", "Y"),
):
    """
    Theoretical covariance for the multivariate generative process:

        Y  = R + U + eps
        X1 = R + U + N
        X2 = R     + N

    where

        R   ~ N(0, Sigma_R)
        U   ~ N(0, Sigma_U)
        N   ~ N(0, Sigma_N)
        eps ~ N(0, Sigma_eps)

    and all latent variables are mutually independent.

    Default output order is [X1, X2, Y].
    """

    Sigma_R = np.asarray(Sigma_R)
    Sigma_U = np.asarray(Sigma_U)
    Sigma_N = np.asarray(Sigma_N)
    Sigma_eps = np.asarray(Sigma_eps)

    p = Sigma_R.shape[0]

    for name, Sigma in {
        "Sigma_R": Sigma_R,
        "Sigma_U": Sigma_U,
        "Sigma_N": Sigma_N,
        "Sigma_eps": Sigma_eps,
    }.items():
        if Sigma.shape != (p, p):
            raise ValueError(f"{name} must have shape {(p, p)}.")
        check_spd(Sigma, name=name)

    cov_blocks = {
        # Marginal covariance blocks
        ("X1", "X1"): Sigma_R + Sigma_U + Sigma_N,
        ("X2", "X2"): Sigma_R + Sigma_N,
        ("Y",  "Y"):  Sigma_R + Sigma_U + Sigma_eps,

        # Cross-covariance blocks
        ("X1", "X2"): Sigma_R + Sigma_N,
        ("X1", "Y"):  Sigma_R + Sigma_U,
        ("X2", "Y"):  Sigma_R,
    }

    # Add symmetric blocks
    for (a, b), block in list(cov_blocks.items()):
        cov_blocks[(b, a)] = block.T

    Sigma_full = np.block([
        [cov_blocks[(a, b)] for b in order]
        for a in order
    ])

    return Sigma_full, cov_blocks


# ============================================================
# 3. Simulation from the same process
# ============================================================

def simulate_multivariate_process(
    n,
    Sigma_R,
    Sigma_U,
    Sigma_N,
    Sigma_eps,
    rng=None,
):
    """
    Simulate:

        Y  = R + U + eps
        X1 = R + U + N
        X2 = R     + N
    """

    if rng is None:
        rng = np.random.default_rng()

    p = Sigma_R.shape[0]
    mean = np.zeros(p)

    R = rng.multivariate_normal(mean, Sigma_R, size=n)
    U = rng.multivariate_normal(mean, Sigma_U, size=n)
    N = rng.multivariate_normal(mean, Sigma_N, size=n)
    eps = rng.multivariate_normal(mean, Sigma_eps, size=n)

    Y = R + U + eps
    X1 = R + U + N
    X2 = R + N

    return X1, X2, Y


# ============================================================
# 4. Extract covariance blocks
# ============================================================

def extract_covariance_blocks(Sigma_full, p):
    """
    Extract covariance blocks assuming order [X1, X2, Y].
    """

    idx_X1 = slice(0, p)
    idx_X2 = slice(p, 2 * p)
    idx_Y = slice(2 * p, 3 * p)

    blocks = {
        "Sigma_X1X1": Sigma_full[idx_X1, idx_X1],
        "Sigma_X2X2": Sigma_full[idx_X2, idx_X2],
        "Sigma_YY": Sigma_full[idx_Y, idx_Y],

        "Sigma_X1X2": Sigma_full[idx_X1, idx_X2],
        "Sigma_X1Y": Sigma_full[idx_X1, idx_Y],
        "Sigma_X2Y": Sigma_full[idx_X2, idx_Y],
    }

    return blocks


# ============================================================
# 5. Whitening procedure
# ============================================================

def whiten_theoretical_covariance_blocks(
    Sigma_full,
    p,
    device="cpu",
    dtype=torch.float64,
):
    """
    Given the full covariance matrix of [X1, X2, Y],
    compute the whitened cross-covariance blocks:

        P = whitened Cov(X1, X2)
        Q = whitened Cov(X1, Y)
        R = whitened Cov(X2, Y)

    using whiten_block from PID_util.

    In Gaussian Idep notation:

        P = K_{X1,X2}
        Q = K_{X1,Y}
        R = K_{X2,Y}

    Under the M7 constraint:

        P_M7 = Q @ R.T
    """

    Sigma = torch.as_tensor(Sigma_full, dtype=dtype, device=device)

    idx_X1 = slice(0, p)
    idx_X2 = slice(p, 2 * p)
    idx_Y = slice(2 * p, 3 * p)

    Sigma_X1X1 = Sigma[idx_X1, idx_X1]
    Sigma_X2X2 = Sigma[idx_X2, idx_X2]
    Sigma_YY = Sigma[idx_Y, idx_Y]

    Sigma_X1X2 = Sigma[idx_X1, idx_X2]
    Sigma_X1Y = Sigma[idx_X1, idx_Y]
    Sigma_X2Y = Sigma[idx_X2, idx_Y]

    P = whiten_block(Sigma_X1X1, Sigma_X1X2, Sigma_X2X2)
    Q = whiten_block(Sigma_X1X1, Sigma_X1Y, Sigma_YY)
    R = whiten_block(Sigma_X2X2, Sigma_X2Y, Sigma_YY)

    P_M7 = Q @ R.T

    I = torch.eye(p, dtype=dtype, device=device)

    Sigma_white = torch.cat(
        [
            torch.cat([I,   P,   Q], dim=1),
            torch.cat([P.T, I,   R], dim=1),
            torch.cat([Q.T, R.T, I], dim=1),
        ],
        dim=0,
    )

    Sigma_white_M7 = torch.cat(
        [
            torch.cat([I,      P_M7,   Q], dim=1),
            torch.cat([P_M7.T, I,      R], dim=1),
            torch.cat([Q.T,    R.T,    I], dim=1),
        ],
        dim=0,
    )

    return {
        "Sigma_X1X1": Sigma_X1X1,
        "Sigma_X2X2": Sigma_X2X2,
        "Sigma_YY": Sigma_YY,

        "Sigma_X1X2": Sigma_X1X2,
        "Sigma_X1Y": Sigma_X1Y,
        "Sigma_X2Y": Sigma_X2Y,

        "P_X1X2": P,
        "Q_X1Y": Q,
        "R_X2Y": R,

        "P_M7": P_M7,

        "Sigma_white": Sigma_white,
        "Sigma_white_M7": Sigma_white_M7,

        "M7_gap_P_minus_QRt": P - P_M7,
        "M7_gap_norm": torch.linalg.norm(P - P_M7).item(),
    }


# ============================================================
# 6. Validation through simulation
# ============================================================

def validate_multivariate_covariance(
    n=1_000_000,
    p=5,
    seed=123,
    device="cpu",
):
    """
    Validate the theoretical covariance by simulation,
    then compute whitened covariance blocks.
    """

    rng = np.random.default_rng(seed)

    Sigma_R = make_spd_matrix(p, rng, eig_min=0.5, eig_max=2.0)
    Sigma_U = make_spd_matrix(p, rng, eig_min=0.5, eig_max=2.0)
    Sigma_N = make_spd_matrix(p, rng, eig_min=0.1, eig_max=1.0)
    Sigma_eps = make_spd_matrix(p, rng, eig_min=0.1, eig_max=1.0)

    Sigma_theoretical, cov_blocks = theoretical_covariance_multivariate(
        Sigma_R=Sigma_R,
        Sigma_U=Sigma_U,
        Sigma_N=Sigma_N,
        Sigma_eps=Sigma_eps,
    )

    X1, X2, Y = simulate_multivariate_process(
        n=n,
        Sigma_R=Sigma_R,
        Sigma_U=Sigma_U,
        Sigma_N=Sigma_N,
        Sigma_eps=Sigma_eps,
        rng=rng,
    )

    Z = np.concatenate([X1, X2, Y], axis=1)

    Sigma_empirical = np.cov(Z, rowvar=False, bias=False)

    abs_err = np.abs(Sigma_empirical - Sigma_theoretical)

    whitened_theoretical = whiten_theoretical_covariance_blocks(
        Sigma_full=Sigma_theoretical,
        p=p,
        device=device,
        dtype=torch.float64,
    )

    whitened_empirical = whiten_theoretical_covariance_blocks(
        Sigma_full=Sigma_empirical,
        p=p,
        device=device,
        dtype=torch.float64,
    )

    print("====================================================")
    print("Covariance validation")
    print("====================================================")
    print("p:", p)
    print("n:", n)
    print("Sigma_theoretical shape:", Sigma_theoretical.shape)
    print("Sigma_empirical shape:", Sigma_empirical.shape)
    print("Max absolute covariance error:", abs_err.max())
    print("Mean absolute covariance error:", abs_err.mean())

    print("\n====================================================")
    print("Whitened theoretical blocks")
    print("====================================================")
    print("P = K_X1X2 shape:", whitened_theoretical["P_X1X2"].shape)
    print("Q = K_X1Y  shape:", whitened_theoretical["Q_X1Y"].shape)
    print("R = K_X2Y  shape:", whitened_theoretical["R_X2Y"].shape)
    print("||P - Q R.T|| theoretical:", whitened_theoretical["M7_gap_norm"])

    print("\n====================================================")
    print("Whitened empirical blocks")
    print("====================================================")
    print("||P - Q R.T|| empirical:", whitened_empirical["M7_gap_norm"])

    return {
        "Sigma_R": Sigma_R,
        "Sigma_U": Sigma_U,
        "Sigma_N": Sigma_N,
        "Sigma_eps": Sigma_eps,

        "Sigma_theoretical": Sigma_theoretical,
        "Sigma_empirical": Sigma_empirical,
        "cov_blocks": cov_blocks,
        "abs_err": abs_err,

        "whitened_theoretical": whitened_theoretical,
        "whitened_empirical": whitened_empirical,

        "X1": X1,
        "X2": X2,
        "Y": Y,
    }


# ============================================================
# 7. Run example
# ============================================================

if __name__ == "__main__":

    results = validate_multivariate_covariance(
        n=1_000_000,
        p=5,
        seed=123,
        device="cpu",
    )

    P = results["whitened_theoretical"]["P_X1X2"]
    Q = results["whitened_theoretical"]["Q_X1Y"]
    R = results["whitened_theoretical"]["R_X2Y"]
    P_M7 = results["whitened_theoretical"]["P_M7"]

    print("\nP:")
    print(P)

    print("\nQ:")
    print(Q)

    print("\nR:")
    print(R)

    print("\nP_M7 = Q @ R.T:")
    print(P_M7)

    print("\nP - P_M7:")
    print(P - P_M7)