import torch


def logdet_wishart_bias(df: int, d: int) -> float:
    """
    Exact finite-sample bias for log|S| when S is the unbiased sample covariance
    from Gaussian data and (df) * S ~ Wishart_d(Sigma, df).

    Returns
    -------
    bias : float
        E[log|S|] - log|Sigma|
    """
    if df <= d - 1:
        raise ValueError(f"Need df > d-1. Got df={df}, d={d}.")

    i = torch.arange(1, d + 1, dtype=torch.float64)
    term = torch.special.digamma((df - i + 1) / 2.0)

    bias = torch.sum(term) + d * torch.log(torch.tensor(2.0 / df, dtype=torch.float64))

    return bias.item()

def mi_wishart_bias(dims: list, n_samples: int):
    """
    Bias correction for Gaussian mutual information estimates
    from unbiased sample covariance.

    Assumes order: [X1, X2, T]
    and torch.cov(..., correction=1), so df = n_samples - 1.

    Returns biases in nats.
    """
    df = n_samples - 1

    if len(dims) == 2:
        d1, d2 = dims

        bias_x1 = logdet_wishart_bias(df, d1)
        bias_x2 = logdet_wishart_bias(df, d2)
        bias_x1x2 = logdet_wishart_bias(df, d1 + d2)

        bias_mi = 0.5 * (bias_x1 + bias_x2 - bias_x1x2)
        return bias_mi

    if len(dims) == 3:
        d1, d2, dt = dims

        bias_x1 = logdet_wishart_bias(df, d1)
        bias_x2 = logdet_wishart_bias(df, d2)
        bias_t = logdet_wishart_bias(df, dt)

        bias_x1x2 = logdet_wishart_bias(df, d1 + d2)
        bias_x1t = logdet_wishart_bias(df, d1 + dt)
        bias_x2t = logdet_wishart_bias(df, d2 + dt)
        bias_x1x2t = logdet_wishart_bias(df, d1 + d2 + dt)

        # Bias of I(X1; T)
        bias_mi_1_t = 0.5 * (bias_x1 + bias_t - bias_x1t)

        # Bias of I(X2; T)
        bias_mi_2_t = 0.5 * (bias_x2 + bias_t - bias_x2t)

        # Bias of I((X1, X2); T)
        bias_tri_mi = 0.5 * (bias_x1x2 + bias_t - bias_x1x2t)

        # Optional: bias of I(X1; X2), if you need it
        bias_mi_12 = 0.5 * (bias_x1 + bias_x2 - bias_x1x2)

        return {
            "bias_mi_1_t": bias_mi_1_t,
            "bias_mi_2_t": bias_mi_2_t,
            "bias_tri_mi": bias_tri_mi,
            "bias_mi_12": bias_mi_12,
        }

    raise ValueError(f"dims must have length 2 or 3. Got len(dims)={len(dims)}.")
