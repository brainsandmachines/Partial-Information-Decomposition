
import torch

def make_only_unq1_zero_cov(config):
    device = config["device"]
    dtype = torch.float64

    dx1 = config["dx1"]
    dx2 = config["dx2"]
    dt = config["dt"]

    q = config.get("q_active", 0.028)
    c = config.get("c_eps", 1.1)
    active_dim = config.get("active_dim",10)

    eps = c * q**2
    if eps >= 1:
        raise ValueError("Need c * q^2 < 1")

    r = (1.0 - eps) ** 0.5

    Q = torch.zeros((dx1, dt), dtype=dtype, device=device)
    R = torch.zeros((dx2, dt), dtype=dtype, device=device)
    P = torch.zeros((dx1, dx2), dtype=dtype, device=device)

    for a in range(active_dim):
        Q[a, a] = q
        R[a, a] = r
        P[a, a] = 0.0

    I0 = torch.eye(dx1, dtype=dtype, device=device)
    I1 = torch.eye(dx2, dtype=dtype, device=device)
    I2 = torch.eye(dt, dtype=dtype, device=device)

    row1 = torch.cat([I0, P, Q], dim=1)
    row2 = torch.cat([P.T, I1, R], dim=1)
    row3 = torch.cat([Q.T, R.T, I2], dim=1)

    Sigma_m8 = torch.cat([row1, row2, row3], dim=0)

    eigmin = torch.min(torch.linalg.eigvalsh(Sigma_m8))
    if eigmin <= 1e-10:
        raise ValueError(f"Covariance not PD. min eig={eigmin.item()}")


    return P,Q,R