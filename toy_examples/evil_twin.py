# import torch
# import numpy as np











# def evil_twin_example(rng,n,p):
#     """
#     Create an "evil twin" example where X and Y are identical copies of S, but with some noise.
#     This should yield a high unique information for both X and Y, and zero shared information.
#     """
#     #Twin sonic (Unique for both)
#     R_so = rng.standard_normal((n, p))
#     R_so *= (0.5 / p) ** 0.5 #Var(R_so) = 0.5

#     N = rng.standard_normal((n, p))
#     N *= (2.5 / p) ** 0.5 #Var(N) = 2.5

#     N_t = rng.standard_normal((n, p))

#     U1_so = rng.standard_normal((n, p))
#     U1_so *= (2.5 / p) ** 0.5 #Var(U1_so) = 2.5

#     U2_so = rng.standard_normal((n, p))
#     U2_so *= (0.5 / p) ** 0.5 #Var(U2_so) = 0.5

#     X1_so = R_so + N + U1_so
#     X2_so = R_so + N + U2_so

#     T_so = R_so + U1_so + U2_so + N_t



#     # Twin Shadow (No unique for X2)
#     R_sh = rng.standard_normal((n, p))
#     R_sh *= (1 / p) ** 0.5 #Var(R_sh) = 1
    
#     N_sh = rng.standard_normal((n, p))
#     N_sh *= (2 / p) ** 0.5 #Var(N_sh) = 2
#     E1_sh = rng.standard_normal((n, p))
#     E1_sh *= (0.5 / p) ** 0.5

#     E2_sh = rng.standard_normal((n, p))
#     E2_sh *= (0.5 / p) ** 0.5

#     U1_sh = rng.standard_normal((n, p))
#     U1_sh *= (2 / p) ** 0.5 #Var(U1_sh) = 2

#     N_t_sh = rng.standard_normal((n, p))
#     E_t = rng.standard_normal((n, p))
#     E_t *= (0.5 / p) ** 0.5

#     X1_sh = R_sh + N_sh + U1_sh + E1_sh
#     X2_sh = R_sh + N_sh + E2_sh
#     T_sh = R_sh + U1_sh + N_t_sh +E_t

#     return {'sonic': (X1_so, X2_so, T_so), 'shadow': (X1_sh, X2_sh, T_sh)}


# def main():
#     rng = np.random.default_rng(0)
#     n = 10000
#     p = 10
#     data = evil_twin_example(rng, n, p)
    


import sys

import torch
from pathlib import Path



root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))  
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss
from utils import Tee
log = open("Evil_Twin_pidvsvp.log", "w")

sys.stdout = Tee(sys.stdout, log)
sys.stderr = Tee(sys.stderr, log)


def empirical_covariance_matrix_torch(X1, X2, T, correction=1):
    """
    Compute empirical covariance matrix of (X1, X2, T) using torch.

    Parameters
    ----------
    X1, X2, T : torch.Tensor
        Each should have shape (n, p).

    correction : int
        correction=1 gives the unbiased sample covariance,
        equivalent to dividing by n - 1.

    Returns
    -------
    Sigma_hat : torch.Tensor
        Covariance matrix of shape (3p, 3p).
        If p=1, this is a 3x3 matrix.
    """

    Z = torch.cat([X1, X2, T], dim=1)  # shape (n, 3p)

    Sigma_hat = torch.cov(Z.T, correction=correction)

    return Sigma_hat


def check_evil_twin_covariances_torch(data, atol=5e-2, rtol=5e-2, verbose=True):
    """
    Check whether the empirical covariance matrices of Sonic and Shadow match.

    Parameters
    ----------
    data : dict
        Output of evil_twin_example(rng, n, p), with keys:
            data["sonic"] = (X1_so, X2_so, T_so)
            data["shadow"] = (X1_sh, X2_sh, T_sh)

    atol, rtol : float
        Tolerances for torch.allclose.

    verbose : bool
        If True, print covariance matrices and diagnostics.

    Returns
    -------
    result : dict
        Contains the covariance matrices, their difference,
        max absolute difference, and equality check.
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
        emoji = "✅" if are_equal else "❌"
        print(f"{are_equal}{emoji}")


    return {
        "Sigma_sonic": Sigma_so,
        "Sigma_shadow": Sigma_sh,
        "difference": diff,
        "max_abs_difference": max_abs_diff,
        "are_equal": are_equal,
    }


def evil_twin_example_torch(generator, n, p, device="cpu", dtype=torch.float64):
    """
    Torch version of the evil twin example.

    For p=1, this is the scalar construction.
    For p>1, the scaling makes each latent block have total variance equal
    to the number written in the comments.
    """

    def randn_scaled(var_total):
        return torch.sqrt(torch.tensor(var_total / p, dtype=dtype, device=device)) * torch.randn(
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

def evil_twin_idep(rng,data):

    data_sonic = data["sonic"]
    data_shadow = data["shadow"]

    sources_sonic = list(data_sonic[0:2])  # (X1_so, X2_so)
    target_sonic = [data_sonic[2]]       # T_so
    sources_shadow = list(data_shadow[0:2])  # (X1_sh, X2_sh)
    target_shadow = [data_shadow[2]]     # T_sh

    idep_sonic = Idep_multivariate_gauss(rng,sources_sonic, target_sonic,bias_correction=False)
    pid_so,mi_so = idep_sonic.idep()
    idep_shadow = Idep_multivariate_gauss(rng,sources_shadow, target_shadow,bias_correction=False)
    pid_sh,mi_sh = idep_shadow.idep()

    return {
        "sonic": {"pid": pid_so, "mi": mi_so},
        "shadow": {"pid": pid_sh, "mi": mi_sh},
    }
torch.set_default_dtype(torch.float64)

g = torch.Generator().manual_seed(0)

n = 10000000
p = 5

data = evil_twin_example_torch(g, n, p)

result = check_evil_twin_covariances_torch(
    data,
    atol=5e-2,
    rtol=5e-2,
)



idep_results = evil_twin_idep(g,data)

print("\n" + "="*60)
print("PARTIAL INFORMATION DECOMPOSITION (PID) RESULTS".center(60))
print("="*60)

for name in ["Sonic", "Shadow"]:
    key = name.lower()
    pid = idep_results[key]["pid"]
    mi = idep_results[key]["mi"]
    
    print(f"\n{name}:")
    print(f"  PID Decomposition:")
    print(f"    Redundancy (Red):     {pid['red']:.4f}")
    print(f"    Unique 1 (Unq1):      {pid['unq1']:.4f}")
    print(f"    Unique 2 (Unq2):      {pid['unq2']:.4f}")
    print(f"    Synergy (Syn):        {pid['syn']:.4f}")
    print(f"  Mutual Information:")
    print(f"    I(M1;T):              {mi['I(M1;T)']:.4f}")
    print(f"    I(M2;T):              {mi['I(M2;T)']:.4f}")
    print(f"    I(M1,M2;T):           {mi['I(M1,M2;T)']:.4f}")

print("\n" + "="*60)
print("COMPARISON: SONIC vs SHADOW".center(60))
print("="*60)

sonic_pid = idep_results["sonic"]["pid"]
shadow_pid = idep_results["shadow"]["pid"]
sonic_mi = idep_results["sonic"]["mi"]
shadow_mi = idep_results["shadow"]["mi"]

print("\nPID Decomposition Differences:")
print(f"  Redundancy (Red):     {sonic_pid['red']:.4f} - {shadow_pid['red']:.4f} = {sonic_pid['red'] - shadow_pid['red']:.4f}")
print(f"  Unique 1 (Unq1):      {sonic_pid['unq1']:.4f} - {shadow_pid['unq1']:.4f} = {sonic_pid['unq1'] - shadow_pid['unq1']:.4f}")
print(f"  Unique 2 (Unq2):      {sonic_pid['unq2']:.4f} - {shadow_pid['unq2']:.4f} = {sonic_pid['unq2'] - shadow_pid['unq2']:.4f}")
print(f"  Synergy (Syn):        {sonic_pid['syn']:.4f} - {shadow_pid['syn']:.4f} = {sonic_pid['syn'] - shadow_pid['syn']:.4f}")

print("\nMutual Information Differences:")
print(f"  I(M1;T):              {sonic_mi['I(M1;T)']:.4f} - {shadow_mi['I(M1;T)']:.4f} = {sonic_mi['I(M1;T)'] - shadow_mi['I(M1;T)']:.4f}")
print(f"  I(M2;T):              {sonic_mi['I(M2;T)']:.4f} - {shadow_mi['I(M2;T)']:.4f} = {sonic_mi['I(M2;T)'] - shadow_mi['I(M2;T)']:.4f}")
print(f"  I(M1,M2;T):           {sonic_mi['I(M1,M2;T)']:.4f} - {shadow_mi['I(M1,M2;T)']:.4f} = {sonic_mi['I(M1,M2;T)'] - shadow_mi['I(M1,M2;T)']:.4f}")

print("\n" + "="*60)
