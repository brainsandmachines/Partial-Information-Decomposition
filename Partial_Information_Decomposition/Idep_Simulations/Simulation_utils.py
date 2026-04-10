import torch
import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.special import digamma
from pathlib import Path
import sys
import os

from shrinkaging import ledoit_wolf_cov, shrunk_cov
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *
import pandas as pd



def mean_std_csv_results(results_dict):
    """ Helper: Compute mean results across seeds """
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    mean_results = df.mean()
    std_results = df.std()
    return mean_results, std_results


def m7_m8_mean_std_csv_results(results_dict):
    """ Helper: Compute mean results across seeds """
    mean_results = {}
    std_results = {}
    for model in results_dict.keys():
        mean_results[model] = results_dict[model]['corrected_statistic']
        std_results[model] = results_dict[model]['std']
    return mean_results, std_results


def N_P_variation_simulation(config,mean_std_func=m7_m8_mean_std_csv_results):
    """ Helper: Run simulations across different N and p values 
    and then create a heatamp of the results. """
    N_values = config['N_values']
    p_values = config['p_values']
    simulation_func = config['simulation_func']
    all_results = []
    len_N = len(N_values)
    len_P = len(p_values)
    i=1
    for N in N_values:
        for p in p_values:
            print(f"\nRunning simulation for N={N}, p={p} ({i}/{len_N*len_P})")
            config['n_samples'] = N
            config['n0'] = p[0]
            config['n1'] = p[1]
            config['n2'] = p[2]
            results_dict = simulation_func(config)
            mean_results, std_results = mean_std_func(results_dict)
            row = {
            "N": N,
            "p": p,  
        }

            for key in mean_results.keys():
                row[f"{key}_mean"] = mean_results[key]
                row[f"{key}_std"] = std_results[key]
                row[f"{key}_ground_truth"] = results_dict[key]['ground_truth']
                row[f"{key}_emp_bias"] = results_dict[key]['emp_bias']

            all_results.append(row)
            print(f"Completed combination N={N}, p={p} ({i}/{len_N * len_P})")
            i += 1

    return all_results



def sample_data_from_cov(config,true_cov:torch.tensor,rng: np.random.Generator) -> np.ndarray:
    """
    Sample multivariate Gaussian data from the specified covariance.
    and return it's covariance matrix. This is a helper function for the m7_whiten bias simulation.
    """
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    d = true_cov.shape[0]
    n_samples = config['n_samples']

    true_cov = true_cov.to(device=config['device'], dtype=torch.float64)
    mean = torch.zeros(d,device=config['device']).to(torch.float64)
    dist = torch.distributions.MultivariateNormal(mean, true_cov)
    data = dist.sample((n_samples,))
    
    X0 = data[:, :n0]
    X1 = data[:, n0:n0+n1]
    X2 = data[:, n0+n1:n0+n1+n2]
    rv_list = [X0, X1, X2]
    sample_cov = torch.cov(data.T, correction=1) # Unbiased estimator with N-1 in the denominator    
    return sample_cov,rv_list # Unbiased estimator with N-1 in the denominator


def on_covriance(config,covariance_matrix):
    """This will call an intermidate function on the covraince"""

    on_cov = config['on_covariance']
    if on_cov == 'False':
        cov = covariance_matrix
    
    elif on_cov == 'ledoit_wolf':
        cov =  ledoit_wolf_cov(covariance_matrix.cpu().numpy())
    
    elif on_cov == 'shrunk_cov':
        alpha = config['alpha']
        cov = shrunk_cov(covariance_matrix.cpu().numpy(), alpha)

    if type(cov) != torch.Tensor:
        cov = torch.from_numpy(cov).to(covariance_matrix.device).to(covariance_matrix.dtype)
    return cov

def safe_logdet(A: torch.Tensor) -> float:
    """
    Compute log determinant and raise if matrix is not positive definite.
    """
    sign, ld = torch.linalg.slogdet(A)

    if torch.any(sign <= 0):
        eigmin = torch.min(torch.linalg.eigvalsh(0.5 * (A + A.mT)))
        raise RuntimeError(
            f"Matrix not positive definite in logdet. sign={sign}, min_eig={eigmin.item():.3e}"
        )

    return ld

def mi_calculation_from_cov(nume_matrix,denoq_matrix,denor_matrix,only_mi=False ) -> float:
    """
    Compute MI from covariance matrices using the formula:
    MI = 0.5 * (log|deno_matrix| - log|nume_matrix|)
    """
    logdet_deno = 0.5*safe_logdet(denoq_matrix) + 0.5*safe_logdet(denor_matrix)
    logdet_nume = 0.5*safe_logdet(nume_matrix)
    if only_mi:
        return logdet_nume - logdet_deno
    mi = (logdet_nume - logdet_deno)
    return mi.item(),logdet_deno.item(), logdet_nume.item()


def mi_calculation_not_whiten(config) -> float:
    """
    Compute MI from covariance matrices using the formula:
    MI = 0.5 * (log|deno_matrix| - log|nume_matrix|)
    """
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    device = config.get('device', 'cpu')
    S = config['Sigma'] #(B, d, d)
    S_dict = para_create_cov_matrix([config['n0'], config['n1'], config['n2']], S)

    if config['model'] == 'M8' or config['model'] == 'M8_M7':
        #M8 
        m8_sigma = S #Denominator of M8 is just the sample covariance
        deno8_raw = 0.5 * safe_logdet(m8_sigma)
        #Numerator
        joint_x0_x1 = S_dict['joint_x0_x1']
        cov_x2 = S_dict['cov_x2']
        nume_m8_joint_raw = 0.5 * safe_logdet(joint_x0_x1)
        nume_m8_target_raw = 0.5 * safe_logdet(cov_x2)
        nume8_raw = nume_m8_joint_raw + nume_m8_target_raw
        mi_m8_raw = nume8_raw - deno8_raw
        m8_Sigma = S
        final_dict_m8 = {'mi': mi_m8_raw,'nume': nume8_raw,'nume_joint': nume_m8_joint_raw,'nume_target': nume_m8_target_raw,'deno': deno8_raw}

    if config['model'] == 'M7' or config['model'] == 'M8_M7':
        #Calculate m7_whiten model logdets
        cross_x0_x1_m7 = S_dict['cross_x0_x2'] @ torch.linalg.inv(S_dict['cov_x2']) @ S_dict['cross_x1_x2'].mT
        cross_x1_x0_m7 = cross_x0_x1_m7.mT
        S_m7 = S.clone()
        S_m7[:, :n0, n0:n0+n1] = cross_x0_x1_m7
        S_m7[:, n0:n0+n1, :n0] = cross_x1_x0_m7

        S_m7_dict = para_create_cov_matrix([config['n0'], config['n1'], config['n2']], S_m7)
        assert torch.allclose(S_m7_dict['cross_x0_x2'], S_dict['cross_x0_x2'])
        assert torch.allclose(S_m7_dict['cross_x0_x1'], cross_x0_x1_m7)
        
        deno7_raw = 0.5 * safe_logdet(S_m7)
        nume_m7_joint_raw = 0.5 * safe_logdet(S_m7_dict['joint_x0_x1'])
        nume_m7_target_raw = 0.5 * safe_logdet(S_m7_dict['cov_x2'])
        nume7_raw = nume_m7_joint_raw + nume_m7_target_raw
        m7_Sigma = S_m7
        mi_m7_raw = nume7_raw - deno7_raw

        final_dict_m7 = {'mi': mi_m7_raw,'nume': nume7_raw,'nume_joint': nume_m7_joint_raw,'nume_target': nume_m7_target_raw,'deno': deno7_raw}
    if config['model'] == 'M8_M7':
        final_dict = {'M8': final_dict_m8, 'M7': final_dict_m7}

    else:
            final_dict = final_dict_m8 if config['model'] == 'M8' else final_dict_m7

    
    return (final_dict,{'M8': S, 'M7': S_m7}) if config['model'] == 'M8_M7' else final_dict

def extract_num_den_matrices(config:dict,matrix:torch.tensor):
    """Extract the numerator and denominator covariance matrices for M7/M8 from the full covariance matrix. 
    assumes whitening"""
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    if len(matrix.shape) == 2:
        nume_matrix = matrix[:n0+n1, :n0+n1]
        denoq_matrix = matrix[:n0, :n0]
        denor_matrix = matrix[n0:n0+n1, n0:n0+n1]
    return nume_matrix,denoq_matrix,denor_matrix


def mi_bias_calc(config:dict):
    model = config['model']
    bias_corr_func_model = config['bias_correction_func'] #Dict with keys to bias correction for each statistic (mi,nume,deno...)

    bias_corr_funcs_model = bias_corr_func_model[f'{model}']


    assert type(bias_corr_funcs_model) == dict, "Expected bias_corr_func to be a dict with keys 'M7' and 'M8'."

    bias_dict ={}
    for st,bc_func in zip(bias_corr_funcs_model.keys(), bias_corr_funcs_model.values()):
        config['st'] = st
        bias = bc_func(config)
        if type(bias) == dict:
            bias = bias[st]
        bias_dict[st] = bias
    return bias_dict


def para_nume_logdet(config,Sigmas: torch.Tensor) -> float:
    """Helper function to compute log determinant of the numerator covariance matrix."""
    n0 = config['n0']
    n1 = config['n1']
    P = Sigmas[:, :n0, n0:n0+n1]                 # already whitened/projected P block
    I1 = torch.eye(n1, dtype=Sigmas.dtype, device=Sigmas.device).repeat(P.shape[0], 1, 1)
    
    return 0.5*safe_logdet(I1 - P.mT @ P)   
    
def para_unique_bias_calc(config:dict):
    """Helper function to compute bias for the unique information estimator."""
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    df = config['n_samples'] - 1
    device = config.get('device', 'cpu')
    d = n0 + n1 + n2
    Sigma = config['Sigma'] #(B, d, d)

    Sigma_dict = para_create_cov_matrix([n0, n1, n2], Sigma)
    P = Sigma_dict['cross_x0_x1']
    Q = Sigma_dict['cross_x0_x2']
    R = Sigma_dict['cross_x1_x2']

    I1 = torch.eye(n1, dtype=Sigma.dtype, device=Sigma.device).repeat(P.shape[0], 1, 1)
    I2 = torch.eye(n2, dtype=Sigma.dtype, device=Sigma.device).repeat(P.shape[0], 1, 1)

    m1_t_mi = -0.5*safe_logdet(I2 - (Q.mT @ Q))
    m2_t_mi = -0.5*safe_logdet(I2 - (R.mT @ R))

    #M8
    nume_m8 = 0.5*safe_logdet(I1 - P.mT @ P)
    deno_m8 = 0.5*safe_logdet(Sigma)
    mi_m8 = nume_m8 - deno_m8

    #M7
    P_m7 = Q @ R.mT
    nume_m7 = 0.5*safe_logdet(I1 - P_m7.mT @ P_m7)
    deno7_q = torch.eye(n2, device=device)-(Q.mT @ Q)
    deno7_r = torch.eye(n2, device=device)-(R.mT @ R)
    deno7_raw = 0.5*safe_logdet(deno7_q) + 0.5*safe_logdet(deno7_r)
    mi_m7 = nume_m7 - deno7_raw
    

    i_value = mi_m7 - m2_t_mi - config['analytic_bias']['i']
    k_value = mi_m8 - m2_t_mi - config['analytic_bias']['k']
    h_value = mi_m7 - m1_t_mi - config['analytic_bias']['h']
    j_value = mi_m8 - m1_t_mi - config['analytic_bias']['j']

    return {
        'i': i_value,
        'k': k_value,
        'h': h_value,
        'j': j_value,}


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

def plot_heatmap_mean_std(
    results,
    x_col="N",
    y_col="p",
    mean_col="mean",
    std_col="std",
    emp_bias_col="emp_bias",
    ground_truth_col="ground_truth",
    title=None,
    cmap="viridis",
    figsize=(8, 6),
    save_path=None,
    mean_fmt=".3f",
    std_fmt=".3f",
):
    """
    Create a heatmap where:
        x-axis = N
        y-axis = p
        color  = mean
        text   = mean ± std

    Parameters
    ----------
    results : list[dict] or pd.DataFrame
        Each entry should look like:
        {'N': ..., 'p': ..., 'mean': ..., 'std': ...}

        Note:
        p can also be a list/tuple/array.
    """
    df = pd.DataFrame(results).copy()

    # Keep y values sortable numerically, not as strings
    def normalize_y(v):
        if isinstance(v, np.ndarray):
            return tuple(v.tolist())
        if isinstance(v, list):
            return tuple(v)
        return v

    # Pretty labels only for display
    def display_label(v):
        if isinstance(v, tuple):
            return str(list(v))
        return str(v)

    df[y_col] = df[y_col].apply(normalize_y)

    # Safer than pivot if duplicates ever appear
    mean_mat = df.pivot_table(index=y_col, columns=x_col, values=mean_col, aggfunc="mean")
    std_mat = df.pivot_table(index=y_col, columns=x_col, values=std_col, aggfunc="mean")
    ground_truth_mat = df.pivot_table(index=y_col, columns=x_col, values=ground_truth_col, aggfunc="mean")
    emp_bias = df.pivot_table(index=y_col, columns=x_col, values=emp_bias_col, aggfunc="mean")
    # Sort axes numerically
    mean_mat = mean_mat.sort_index()
    mean_mat = mean_mat.reindex(sorted(mean_mat.columns), axis=1)

    std_mat = std_mat.reindex(index=mean_mat.index, columns=mean_mat.columns)
    ground_truth_mat = ground_truth_mat.reindex(index=mean_mat.index, columns=mean_mat.columns)
    emp_bias = emp_bias.reindex(index=mean_mat.index, columns=mean_mat.columns)
    data = mean_mat.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    # origin="lower" puts the smallest row at the bottom
    im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(mean_mat.columns)))
    ax.set_xticklabels(mean_mat.columns)

    ax.set_yticks(np.arange(len(mean_mat.index)))
    ax.set_yticklabels([display_label(v) for v in mean_mat.index])

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if title is not None:
        ax.set_title(title)

    threshold = np.nanmean(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            m = mean_mat.iloc[i, j]
            s = std_mat.iloc[i, j]
            gt = ground_truth_mat.iloc[i, j]
            eb = emp_bias.iloc[i, j]
            if pd.isna(m):
                text = "NA"
            elif pd.isna(s):
                text = f"{m:{mean_fmt}}\n\nGT={gt:{mean_fmt}}"
            else:
                text = f"{m:{mean_fmt}}\n±{s:{std_fmt}}\n\nGT={gt:{mean_fmt}}\n\nEB={eb:{mean_fmt}}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if pd.notna(m) and m < threshold else "black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean")

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(f"{save_path}/{title}.png", dpi=300, bbox_inches="tight")



def corrected_statistic(statistics: np.ndarray, bias_correction: float) -> np.ndarray:
    """
    Apply bias correction to the raw statistics.
    """
    return statistics - bias_correction


