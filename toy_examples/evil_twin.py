
import sys
import torch
from pathlib import Path

from functools import partial

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from gpid import tilde_pid
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss
from Partial_Information_Decomposition.PID_util import residual_rvs,create_cov_matrix
from utils import Tee
log = open("tilde_Evil_Twin_pidv.log", "w")

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

def evil_twin_idep(rng,data,on_rvs:callable=None):

    data_sonic = data["sonic"]
    data_shadow = data["shadow"]

    sources_sonic = list(data_sonic[0:2])  # (X1_so, X2_so)
    target_sonic = [data_sonic[2]]       # T_so
    sources_shadow = list(data_shadow[0:2])  # (X1_sh, X2_sh)
    target_shadow = [data_shadow[2]]     # T_sh

    if on_rvs is not None:
        print("\nApplying on_rvs transformation to sources...")
        sources_sonic = on_rvs(sources_sonic)

        sources_shadow = on_rvs(sources_shadow)

    idep_sonic = Idep_multivariate_gauss(rng,sources_sonic, target_sonic,bias_correction=False)
    pid_so,mi_so = idep_sonic.idep()
    idep_shadow = Idep_multivariate_gauss(rng,sources_shadow, target_shadow,bias_correction=False)
    pid_sh,mi_sh = idep_shadow.idep()

    return {
        "sonic": {"pid": pid_so, "mi": mi_so},
        "shadow": {"pid": pid_sh, "mi": mi_sh},
    }


def evil_twin_tilde_pid(data):
    """Compute pid using BROJA definiton and calculation from Venkateh et al. 2023.
    
    My code assumes [predictor1, predictor2, target] ordering in data["sonic"] and data["shadow"].
    gpid assumes [target, predictor1, predictor2] ordering, so we need to rearrange the data accordingly.
    """
    data_sonic = data["sonic"]
    data_shadow = data["shadow"]
    
    data_sonic_reordered = [data_sonic[2], data_sonic[0], data_sonic[1]]  # [T_so, X1_so, X2_so]
    data_shadow_reordered = [data_shadow[2], data_shadow[0], data_shadow[1]]  # [T_sh, X1_sh, X2_sh]

    dict_cov_sonic = create_cov_matrix(data_sonic_reordered)
    dict_cov_shadow = create_cov_matrix(data_shadow_reordered)

    cov_sonic = dict_cov_sonic["full_cov"]
    cov_shadow = dict_cov_shadow["full_cov"]
    print(f"\n Covariance matrix for Sonic (shape {cov_sonic.shape}):")
    
    cov_sonic = cov_sonic.cpu().numpy()
    cov_shadow = cov_shadow.cpu().numpy()

    dm_sonic,dx_sonic,dy_soinc=  data_sonic_reordered[0].shape[1], data_sonic_reordered[1].shape[1], data_sonic_reordered[2].shape[1]
    dm_shadow,dx_shadow,dy_shadow=  data_shadow_reordered[0].shape[1], data_shadow_reordered[1].shape[1], data_shadow_reordered[2].shape[1]

    # (imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si)
    output_so = tilde_pid.exact_gauss_tilde_pid(cov_sonic,dm_sonic,dx_sonic,dy_soinc,unbiased=True,sample_size=data_sonic_reordered[0].shape[0]) 
    output_sh = tilde_pid.exact_gauss_tilde_pid(cov_shadow,dm_shadow,dx_shadow,dy_shadow,unbiased=True,sample_size=data_shadow_reordered[0].shape[0])

    pid_so = {'red': output_so[7], 'unq1': output_so[7], 'unq2': output_so[6], 'syn': output_so[8]}
    pid_sh = {'red': output_sh[7], 'unq1': output_sh[7], 'unq2': output_sh[6], 'syn': output_sh[8]}

    mi_so = {'I(M1;T)': output_so[0], 'I(M2;T)': output_so[1], 'I(M1,M2;T)': output_so[2]}
    mi_sh = {'I(M1;T)': output_sh[0], 'I(M2;T)': output_sh[1], 'I(M1,M2;T)': output_sh[2]}

    return {
        "sonic": {"pid": pid_so, "mi": mi_so},
        "shadow": {"pid": pid_sh, "mi": mi_sh},
    }

torch.set_default_dtype(torch.float64)

g = torch.Generator().manual_seed(0)

n = 1000
p = 300 #dim for each rv

data = evil_twin_example_torch(g, n, p)

result = check_evil_twin_covariances_torch(
    data,
    atol=5e-2,
    rtol=5e-2,
)


res_rv = None #partial(residual_rvs, target_index=1) #rv2 is the predictor 
#idep_results = evil_twin_idep(g,data,on_rvs=residual_rvs)
gpid_results = evil_twin_tilde_pid(data)
pid_results = gpid_results

print("\n" + "="*60)
print("PARTIAL INFORMATION DECOMPOSITION (PID) RESULTS".center(60))
print("="*60)

for name in ["Sonic", "Shadow"]:
    key = name.lower()
    pid = pid_results[key]["pid"]
    mi = pid_results[key]["mi"]
    
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

sonic_pid = pid_results["sonic"]["pid"]
shadow_pid = pid_results["shadow"]["pid"]
sonic_mi = pid_results["sonic"]["mi"]
shadow_mi = pid_results["shadow"]["mi"]

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
