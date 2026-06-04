"""Torch Sonic/Shadow evil-twin covariance example."""

import torch


def empirical_covariance_matrix_torch(X1, X2, T, correction=1):
    """
    Compute the empirical covariance matrix of (X1, X2, T).

    Args:
        X1: First source tensor with shape (n_samples, n_features).
        X2: Second source tensor with shape (n_samples, n_features).
        T: Target tensor with shape (n_samples, n_features).
        correction: Correction used by torch.cov. The default, 1, gives the
            unbiased sample covariance.

    Returns:
        torch.Tensor: Covariance matrix with shape (3 * n_features, 3 * n_features).
    """
    Z = torch.cat([X1, X2, T], dim=1)
    return torch.cov(Z.T, correction=correction)


def check_evil_twin_covariances_torch(data, atol=5e-2, rtol=5e-2, verbose=True):
    """
    Compare the empirical covariance matrices of Sonic and Shadow.

    Args:
        data: Output of evil_twin_example_torch, with "sonic" and "shadow"
            entries containing (X1, X2, T).
        atol: Absolute tolerance for torch.allclose.
        rtol: Relative tolerance for torch.allclose.
        verbose: If True, print the covariance matrices and diagnostics.

    Returns:
        dict: Sonic covariance, Shadow covariance, their difference, the maximum
        absolute difference, and whether they are close under the tolerances.
    """
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
        print(are_equal)

    return {
        "Sigma_sonic": Sigma_so,
        "Sigma_shadow": Sigma_sh,
        "difference": diff,
        "max_abs_difference": max_abs_diff,
        "are_equal": are_equal,
    }


def evil_twin_example_torch(generator, n, p, device="cpu", dtype=torch.float64):
    """
    Generate the Sonic and Shadow evil-twin Gaussian examples.

    For p=1 this is the scalar construction. For p>1, each latent block is
    scaled so its total variance matches the scalar construction.

    Args:
        generator: torch.Generator controlling randomness.
        n: Number of samples.
        p: Number of features per random variable.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        dict: "sonic" and "shadow" entries, each containing (X1, X2, T).
    """

    def randn_scaled(var_total):
        scale = torch.sqrt(torch.tensor(var_total / p, dtype=dtype, device=device))
        return scale * torch.randn(
            n,
            p,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    R_so = randn_scaled(0.5)
    N_so = randn_scaled(2.5)
    U1_so = randn_scaled(2.5)
    U2_so = randn_scaled(0.5)
    N_t_so = randn_scaled(1.0)

    X1_so = R_so + N_so + U1_so
    X2_so = R_so + N_so + U2_so
    T_so = R_so + U1_so + U2_so + N_t_so

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


def run_covariance_comparison(
    n=1000,
    p=300,
    seed=0,
    device="cpu",
    dtype=torch.float64,
    atol=5e-2,
    rtol=5e-2,
    verbose=True,
):
    """
    Generate Sonic/Shadow samples and compare their empirical covariances.

    This is the small demonstration entry point for showing that Sonic and
    Shadow are designed to have matching covariance structure.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    data = evil_twin_example_torch(
        generator=generator,
        n=n,
        p=p,
        device=device,
        dtype=dtype,
    )
    result = check_evil_twin_covariances_torch(
        data,
        atol=atol,
        rtol=rtol,
        verbose=verbose,
    )
    result["data"] = data
    return result


if __name__ == "__main__":
    run_covariance_comparison()
