import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from PID_util import *




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


def calcualte_mi(config,sigma_dict):
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    device = config['device']

    Q = sigma_dict['Q']
    R = sigma_dict['R']
    P = sigma_dict['P']
    sigma = sigma_dict['Sigma']
    nume_raw = 0.5*safe_logdet((torch.eye(n1, device=device) - (P.T @ P)))
    deno_q = torch.eye(n2, device=device)-(Q.T @ Q)
    deno_r = torch.eye(n2, device=device)-(R.T @ R)
    deno_raw = 0.5*safe_logdet(sigma)

    mi_raw = (nume_raw - deno_raw).item()
    return {"mi": mi_raw,'nume': nume_raw.item(),'deno': deno_raw.item()}
