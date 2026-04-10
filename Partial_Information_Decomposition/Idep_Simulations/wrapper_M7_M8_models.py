import torch 
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *




def _build_whitened_blocks_from_cov(S, n0, n1, n2):
    """
    Given a covariance matrix S in block order [X0, X1, Y],
    return the whitened blocks P, Q, R using the same helpers as the main code.
    """
    S_torch = torch.from_numpy(S).to(torch.float64)
    S_dict = create_cov_matrix(Sigma=S_torch, dims=[n0, n1, n2])

    P = whiten_block(S_dict["cov_x0"], S_dict["cross_x0_x1"], S_dict["cov_x1"]).numpy()
    Q = whiten_block(S_dict["cov_x0"], S_dict["cross_x0_x2"], S_dict["cov_x2"]).numpy()
    R = whiten_block(S_dict["cov_x1"], S_dict["cross_x1_x2"], S_dict["cov_x2"]).numpy()
    return P, Q, R, S_dict


def make_random_true_cov(
    config: dict,
    n0: int,
    n1: int,
    n2: int,
    q_scale: float = 0.25,
    r_scale: float = 0.25,
    p_scale: float = 0.25,
    rng:  torch.Generator | None = None,
    m7_whiten_structural: bool = True,
) -> np.ndarray:
    """
    Construct a generic positive-definite Gaussian M7_whiten and M8_Whiten covariance.

    Block order is [X0, X1, Y].

    m7_whiten population structure:
        Sigma_01 = Sigma_02 @ Sigma_22^{-1} @ Sigma_21
    and here Sigma_22 = I, so:
        P = Q @ R.T
    """

    A = torch.randn((n0, n2), generator=rng, dtype=torch.float64,device=config['device'])
    B = torch.randn((n1, n2), generator=rng, dtype=torch.float64,device=config['device'])
    C = torch.randn((n0, n1), generator=rng, dtype=torch.float64,device=config['device'])

    A_norm = torch.linalg.norm(A, ord=2)
    B_norm = torch.linalg.norm(B, ord=2)
    C_norm = torch.linalg.norm(C, ord=2)   
    if A_norm == 0 or B_norm == 0 or C_norm == 0:
        raise RuntimeError("Unexpected zero spectral norm in random construction.")

    Q = q_scale * A / A_norm
    R = r_scale * B / B_norm
    P = p_scale * C / C_norm

    # Construct M8 covariance - sample coavraiance after whitening each block
    row1_m8 = torch.cat([torch.eye(n0, device=config['device']), P, Q], dim=1)
    row2_m8 = torch.cat([P.T, torch.eye(n1, device=config['device']), R], dim=1)
    row3_m8 = torch.cat([Q.T, R.T, torch.eye(n2, device=config['device'])], dim=1)   
    true_cov_m8 = torch.cat([row1_m8, row2_m8, row3_m8], dim=0)

    eigvals = torch.linalg.eigvalsh(true_cov_m8)
    if torch.min(eigvals) <= 1e-10:
        raise ValueError(
            f"Constructed covariance not sufficiently PD. min eig={torch.min(eigvals):.3e}"
        )
 
    true_cov_m7 = create_m7_cov(config,true_cov_m8, whitening_normalize=True)


    
    eigvals_m7 = torch.linalg.eigvalsh(true_cov_m7)
    if torch.min(eigvals_m7) <= 1e-10:
        raise ValueError(
            f"Constructed M7 covariance not sufficiently PD. min eig={torch.min(eigvals_m7):.3e}"
        )

    # # Check precision-matrix m7_whiten condition: K_{X0,X1} = 0
    # K = np.linalg.inv(true_cov)
    # K01 = K[:n0, n0:n0+n1]
    # if not np.allclose(K01, 0, atol=1e-10):
    #     raise ValueError("Constructed covariance does not satisfy the m7_whiten precision condition.")

    return true_cov_m8, true_cov_m7


def create_m7_cov(config:dict,cov_m8,whitening_normalize:bool = True):
    "Takes covariance of m8 and creates m7"

    cov_m8_dict = create_cov_matrix(Sigma=cov_m8, dims=[config['n0'], config['n1'], config['n2']])
    diag0 = torch.eye(config['n0'],device=config['device']) if whitening_normalize else torch.diag(cov_m8_dict["cov_x0"]) 
    diag1 = torch.eye(config['n1'],device=config['device']) if whitening_normalize else torch.diag(cov_m8_dict["cov_x1"])
    diag2 = torch.eye(config['n2'],device=config['device']) if whitening_normalize else torch.diag(cov_m8_dict["cov_x2"])
    if whitening_normalize: 
        Q = whiten_block(cov_m8_dict["cov_x0"], cov_m8_dict["cross_x0_x2"], cov_m8_dict["cov_x2"])
        R = whiten_block(cov_m8_dict["cov_x1"], cov_m8_dict["cross_x1_x2"], cov_m8_dict["cov_x2"])
        P = Q @ R.T

    else:
        Q = cov_m8_dict["cross_x0_x2"]
        R = cov_m8_dict["cross_x1_x2"]
        P = Q @ torch.linalg.inv(cov_m8_dict["cov_x2"]) @ R.T

    #Construct M7 covariance 
    row1 = torch.cat([diag0, P, Q], dim=1)
    row2 = torch.cat([P.T, diag1, R], dim=1)
    row3 = torch.cat([Q.T, R.T, diag2], dim=1)
    return torch.cat([row1, row2, row3], dim=0)

def simulation(config,functions_dict:dict,seed=None):
    """
    Run a simulation over combinations of N and p values, computing the specified statistic and bias correction.
    """
    results_dict = {}
    device = config['device']
    rng = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    bias_method = config['bias_method']
    s_simulation_func = functions_dict["s_simulation"]
    bias_correction_func = functions_dict["bias_correction"]
    corrected_statistic_func = functions_dict["corrected_statistic"]

    m8_true_cov, m7_true_cov = make_random_true_cov(config,config["n0"], config["n1"], config["n2"], q_scale=config["q_scale"], r_scale=config["r_scale"], p_scale=config["p_scale"], rng=rng)

    data = [m8_true_cov, m7_true_cov]
    sim_config = config.copy()
    sim_config['bias_correction_func'] = bias_correction_func
    statistic = s_simulation_func(data, sim_config, rng)

    for statistic_key in statistic.keys():
        print(f"Finishing for {statistic_key}...")
        statistic_model = statistic[statistic_key]

        
        model_config = config.copy()
        model_config['statistics'] = statistic_model

        if  f"corrected_avg" in statistic_model:
            model_corr_values = statistic_model['corrected_avg']
        
        else:
            if bias_method[0] == 'analytic':
                model_bc_func = bias_correction_func[statistic_key]
                model_bias_correction = model_bc_func(model_config)
                
                model_corr_values = corrected_statistic_func(statistic_model['avg'], model_bias_correction['bias'])
            
            else: 
                    model_corr_values = statistic_model['avg_resample']

       
                
        results_dict[statistic_key] = {
            'sample': statistic_model['sample'],
            'avg': statistic_model['avg'],
            'std': statistic_model['std'],
            'emp_bias': statistic_model['emp_bias'],
            'corrected_statistic': model_corr_values,
            'ground_truth': statistic_model['ground_truth']
        }


    return results_dict






