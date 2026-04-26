import torch
import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.special import digamma
from pathlib import Path
import sys
import os

from shrinkaging import ledoit_wolf_cov, oracle_shrinkage_cov, shrunk_cov
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from mi_functions import safe_logdet
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
                row[f"{key}_after_corr_bias"] = results_dict[key]['after_corr_bias']

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


def build_m8_terms(config, cov_dict,whiten:bool='whiten_ver',para=False):
    '''Build the covariance matrix for M7 using the specified covariance dictionary.
    input: 
    config - a dictionary with keys 'n0', 'n1', 'n2' and 'device
    cov_dict -  m7 dictionary of the covariance
    whiten:
            whiten_ver = String that tells to whiten
            True = Assume the covariance is already whitened and just use the cross-covariance blocks as P,Q,R
            False = Don't whiten and use the original covariance blocks as P,Q,R '''
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    device = config.get('device', 'cpu')

    if whiten == 'False':
        P = cov_dict['cross_x1_x2']
        Q = cov_dict['cross_x1_xt']
        R = cov_dict['cross_x2_xt']
        
    elif whiten == 'whiten_ver':
        P = whiten_block(cov_dict['cov_x1'], cov_dict['cross_x1_x2'], cov_dict['cov_x2'])
        Q = whiten_block(cov_dict['cov_x1'], cov_dict['cross_x1_xt'], cov_dict['cov_xt'])
        R = whiten_block(cov_dict['cov_x2'], cov_dict['cross_x2_xt'], cov_dict['cov_xt'])

    elif whiten == 'True': #If everything is already whitened 
        P = cov_dict['cross_x1_x2']
        Q = cov_dict['cross_x1_xt']
        R = cov_dict['cross_x2_xt']

    if not para:
        row1_m8 = torch.cat([torch.eye(n0, device=device), P, Q], dim=1)
        row2_m8 = torch.cat([P.T, torch.eye(n1, device=device), R], dim=1)
        row3_m8 = torch.cat([Q.T, R.T, torch.eye(n2, device=device)], dim=1)

        m8_Sigma = torch.cat([row1_m8, row2_m8, row3_m8], dim=0) if whiten != 'False' else cov_dict['full_cov']
    else:
        assert len(cov_dict['full_cov'].shape) == 3, "Expected full_cov to have shape (B, d, d) for parallel M8 construction."
        batch_size = cov_dict['cov_x1'].shape[0]
        row1_m8 = torch.cat([torch.eye(n0, device=device).repeat(batch_size, 1, 1), P, Q], dim=2)
        row2_m8 = torch.cat([P.mT, torch.eye(n1, device=device).repeat(batch_size, 1, 1), R], dim=2)
        row3_m8 = torch.cat([Q.mT, R.mT, torch.eye(n2, device=device).repeat(batch_size, 1, 1)], dim=2)

        m8_Sigma = torch.cat([row1_m8, row2_m8, row3_m8], dim=1) if whiten != 'False' else cov_dict['full_cov']
    return {'P': P,
        'Q': Q,
        'R': R,
        'Sigma': m8_Sigma}


def build_m7_terms(config, cov_dict,whiten:bool='whiten_ver',para=False):
    '''Build the covariance matrix for M7 using the specified covariance dictionary.
    input: 
    config - a dictionary with keys 'n0', 'n1', 'n2' and 'device
    cov_dict -  m7 dictionary of the covariance
    whiten:
            whiten_ver = String that tells to whiten
            True = Assume the covariance is already whitened and just use the cross-covariance blocks as P,Q,R
            False = Don't whiten and use the original covariance blocks as P,Q,R '''
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']

    device = config.get('device', 'cpu')
    if whiten == 'False':
        Q = cov_dict['cross_x1_xt']
        R = cov_dict['cross_x2_xt']
        cov_tt = cov_dict['cov_xt'] 

    elif whiten == 'whiten_ver':
        Q = para_whiten_block(cov_dict['cov_x1'], cov_dict['cross_x1_xt'], cov_dict['cov_xt'])
        R = para_whiten_block(cov_dict['cov_x2'], cov_dict['cross_x2_xt'], cov_dict['cov_xt'])
        cov_tt = torch.eye(n2,device=device)
    elif whiten== 'True': #If everything is already whitened
        Q = cov_dict['cross_x1_xt']
        R = cov_dict['cross_x2_xt']
        cov_tt = cov_dict['cov_xt'] #IDentity matrix
        assert torch.allclose(cov_tt, torch.eye(n2, device=device).to(dtype=cov_tt.dtype)), "Expected cov_x2 to be identity when whiten is True."
    
    if not para:
        cov_11 = torch.eye(n0, device=device) if whiten != 'False' else cov_dict['cov_x1']
        cov_22 = torch.eye(n1, device=device) if whiten != 'False' else cov_dict['cov_x2']
        cov_tt = torch.eye(n2, device=device) if whiten != 'False' else cov_dict['cov_xt']


        tt_inv = torch.linalg.inv(cov_tt).to(dtype=Q.dtype,device=device)
        P_m7 = Q @ tt_inv @ R.T
        
        row1_m7 = torch.cat([cov_11, P_m7, Q], dim=1)
        row2_m7 = torch.cat([P_m7.T, cov_22, R], dim=1)
        row3_m7 = torch.cat([Q.T, R.T, cov_tt], dim=1)   
        m7_Sigma = torch.cat([row1_m7, row2_m7, row3_m7], dim=0)

    else:
        assert len(cov_dict['full_cov'].shape) == 3, "Expected full_cov to have shape (B, d, d) for parallel M7 construction."
        batch_size = cov_dict['cov_x1'].shape[0]
        cov_11 = torch.eye(n0, device=device).repeat(batch_size, 1, 1) if whiten != 'False' else cov_dict['cov_x1']
        cov_22 = torch.eye(n1, device=device).repeat(batch_size, 1, 1) if whiten != 'False' else cov_dict['cov_x2']
        cov_tt = torch.eye(n2, device=device).repeat(batch_size, 1, 1) if whiten != 'False' else cov_dict['cov_xt']


        tt_inv = torch.linalg.inv(cov_tt).to(dtype=Q.dtype,device=device)
        P_m7 = Q @ tt_inv @ R.mT
        
        row1_m7 = torch.cat([cov_11, P_m7, Q], dim=2)
        row2_m7 = torch.cat([P_m7.mT, cov_22, R], dim=2)
        row3_m7 = torch.cat([Q.mT, R.mT, cov_tt], dim=2)   
        m7_Sigma = torch.cat([row1_m7, row2_m7, row3_m7], dim=1)

    return {'P': P_m7,
        'Q': Q,
        'R': R,
        'Sigma': m7_Sigma}


def on_covariance(config,data):
    """This will call an intermidate function on the covariance
    
    Input: data: a tuple containing the covariance matrix and a list of random variables
    
    config: a dictionary with data and parameters for the function to apply on the covariance matrix.
        config['on_cov'] = 'ledoit_wolf' or 'oas' or 'shrunk_cov' or 'False etc...'
        config['alpha'] = the alpha parameter for the shrunk_cov function if on_cov is 'shrunk_cov'
  
    
    Output: the covariance matrix after applying the function on it."""

    covariance_matrix = data
    cov_list = []
    on_cov = config['on_covariance'] #Check for srhinkage method 
    if on_cov == 'False':
        return {'cov': covariance_matrix}
    
    if covariance_matrix.ndim == 2:
        covariance_matrix = covariance_matrix.unsqueeze(0)

    for cov in covariance_matrix:
        if torch.any(torch.isnan(cov)):
            raise ValueError("Covariance matrix contains NaN values.")
        if torch.any(torch.isinf(cov)):
            raise ValueError("Covariance matrix contains Inf values.")
        
        elif on_cov == 'ledoit_wolf':
            cov =  ledoit_wolf_cov(cov.cpu().numpy())
        
        elif on_cov == 'shrunk_cov':
            alpha = config['alpha']
            cov = shrunk_cov(cov.cpu().numpy(), alpha)
        
        if on_cov == 'oas':
            cov =  oracle_shrinkage_cov(cov.cpu().numpy())

        if type(cov) != torch.Tensor:
            cov = torch.from_numpy(cov).to(covariance_matrix.device).to(covariance_matrix.dtype)
        cov_list.append(cov)
    cov = torch.stack(cov_list, dim=0)    
    assert cov.shape == covariance_matrix.shape, f"Expected output shape {covariance_matrix.shape}, got {cov.shape}." 
    return {'cov':cov}



# def mi_calculation_from_cov(nume_matrix,denoq_matrix,denor_matrix,only_mi=False ) -> float:
#     """
#     Compute MI from covariance matrices using the formula:
#     MI = 0.5 * (log|deno_matrix| - log|nume_matrix|)
#     """
#     logdet_deno = 0.5*safe_logdet(denoq_matrix) + 0.5*safe_logdet(denor_matrix)
#     logdet_nume = 0.5*safe_logdet(nume_matrix)
#     if only_mi:
#         return logdet_nume - logdet_deno
#     mi = (logdet_nume - logdet_deno)
#     return mi.item(),logdet_deno.item(), logdet_nume.item()




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
        bias = bc_func(config=config)
        if type(bias) == dict:
            try:
                bias = bias['bias']
            except KeyError:
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


def plot_nodes_as_alpha(node_dict, title=None, save_path=None):
    """Helper function to plot the bias-corrected statistics as a function of alpha."""
    alphas = list(node_dict.keys())
    i_values = [node_dict[alpha]['i'] for alpha in alphas]
    j_values = [node_dict[alpha]['j'] for alpha in alphas]
    k_values = [node_dict[alpha]['k'] for alpha in alphas]
    h_values = [node_dict[alpha]['h'] for alpha in alphas]

    plt.figure(figsize=(8, 6))
    plt.plot(alphas, i_values, label='I', marker=None)
    plt.plot(alphas, j_values, label='J', marker=None)
    plt.plot(alphas, k_values, label='K', marker=None)
    plt.plot(alphas, h_values, label='H', marker=None)
    plt.xlabel('Alpha')
    plt.ylabel('Node Value')
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(f"{save_path}/{title}_nodes.png", dpi=300, bbox_inches="tight")