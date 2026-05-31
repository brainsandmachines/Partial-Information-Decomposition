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





def np_safe_logdet(A, eps=1e-8):
    """Stable logdet for covariance matrices."""
    A = A + eps * np.eye(A.shape[0])
    sign, val = np.linalg.slogdet(A)
    if sign <= 0:
        raise ValueError("Matrix is not positive definite.")
    return val





def calcualte_mi(config,sigma_dict,term='full'):
    """This function calculates the tri-variate mutual information using the covariance 
    matrices and the formula MI = 0.5 * (log|deno_matrix| - log|nume_matrix|)"""
    dx1 = config['dx1']
    dx2 = config['dx2']
    dt = config['dt']
    device = config['device']

    Q = sigma_dict['Q']
    R = sigma_dict['R']
    P = sigma_dict['P']
    sigma = sigma_dict['Sigma']
    nume_raw = 0.5*safe_logdet((torch.eye(dx2, device=device) - (P.T @ P)))
    deno_q = torch.eye(dt, device=device)-(Q.T @ Q)
    deno_r = torch.eye(dt, device=device)-(R.T @ R)
    deno_raw = 0.5*safe_logdet(sigma)

    mi_tri = (nume_raw - deno_raw).item()
    mi_bi_1 = -0.5 * (safe_logdet(torch.eye(dt, device=device) - (Q.T @ Q)))
    mi_bi_2 = -0.5 * (safe_logdet(torch.eye(dt, device=device) - (R.T @ R)))
    all_terms_dict = {"mi_tri": mi_tri,'mi_bi_1': mi_bi_1,'mi_bi_2': mi_bi_2,'nume': nume_raw,'deno': deno_raw}
    if term == 'full':
        return {"mi_tri": mi_tri,'mi_bi_1': mi_bi_1,'mi_bi_2': mi_bi_2.item(),'nume': nume_raw.item(),'deno': deno_raw.item()}
    else:
        return {term: all_terms_dict[term]}
    
def calculate_mi_raw(device,sigma,dims):
    """This function calculates the tri-variate or bi-variate
    mutual information using the covariance matrices without any whitening - in raw mode (:
    
    Input: 
        device: torch.device - the device on which to perform the calculations
        sigma: torch.Tensor - the covariance matrix of the joint distribution of the sources and target, in the order [X1,X2,T]
        dims = list - the dimensions of the random variables\
        tri_variate: bool - whether to calculate tri-variate mutual information (True) or bi-variate mutual information (False)


    Output: Mutual Iformation value (float)"""


    sigma_dict = create_cov_matrix(Sigma=sigma, dims=dims, device=device)

    cov_x12 = sigma_dict['joint_x1_x2']
    cov_t = sigma_dict['cov_t']

    #log∣ΣX1​X2​​∣
    logdet_x12 = 0.5 * safe_logdet(cov_x12)
    #log∣ΣT∣
    logdet_t = 0.5 * safe_logdet(cov_t)
    #log∣ΣT​X1​X2​​∣
    logdet_joint = 0.5 * safe_logdet(sigma)
    mi_tri = logdet_x12 + logdet_t - logdet_joint

    #bi-X1T:
    cov_x1 = sigma_dict['cov_x1']
    cov_x1_t = sigma_dict['joint_x1_t']
    logdet_x1 = 0.5 * safe_logdet(cov_x1)
    logdet_x1_t = 0.5 * safe_logdet(cov_x1_t)
    mi_bi_1 = logdet_x1 + logdet_t - logdet_x1_t

    #bi-X2T:
    cov_x2 = sigma_dict['cov_x2']
    cov_x2_t = sigma_dict['joint_x2_t']
    logdet_x2 = 0.5 * safe_logdet(cov_x2)
    logdet_x2_t = 0.5 * safe_logdet(cov_x2_t)

    mi_bi_2 = logdet_x2 + logdet_t - logdet_x2_t

    return {'tri_mi':mi_tri.item(),'bi_mi_1': mi_bi_1.item(),'bi_mi_2': mi_bi_2.item()}




    

    
    

    
    

def para_calcualte_mi(config,sigma_dict,term='full',assumed_whitened = True):
    """This function calculates the tri-variate mutual information using the for 
    multiple covariances 
    matrices and the formula MI = 0.5 * (log|deno_matrix| - log|nume_matrix|)"""
    assert sigma_dict.keys() == {'P', 'Q', 'R', 'Sigma'}, f"Expected keys 'P', 'Q', 'R', 'Sigma' in sigma_dict, got {sigma_dict.keys()}"
    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    device = config['device']

    Q = sigma_dict['Q']
    R = sigma_dict['R']
    P = sigma_dict['P']
    sigma = sigma_dict['Sigma']
    nume_raw = 0.5*safe_logdet((torch.eye(n1, device=device) - (P.mT @ P)))
    deno_q = torch.eye(n2, device=device)-(Q.mT @ Q)
    deno_r = torch.eye(n2, device=device)-(R.mT @ R)
    deno_raw = 0.5*safe_logdet(sigma)

    mi_tri = (nume_raw - deno_raw)
    mi_bi_1 = -0.5 * (safe_logdet(torch.eye(n2, device=device) - (Q.mT @ Q)))
    mi_bi_2 = -0.5 * (safe_logdet(torch.eye(n2, device=device) - (R.mT @ R)))
    
    all_terms_dict = {"mi_tri": mi_tri,'mi_bi_1': mi_bi_1,'mi_bi_2': mi_bi_2,'nume': nume_raw,'deno': deno_raw}
    if term == 'full':
        return {"mi_tri": mi_tri,'mi_bi_1': mi_bi_1,'mi_bi_2': mi_bi_2.item(),'nume': nume_raw.item(),'deno': deno_raw.item()}
    else:
        return {term: all_terms_dict[term]}



def calculate_mi_lr(config,sigma_dict):
    """This function calculates the  trivarite (X1;X2,T) mutual information using the
    covaraince matrix especially for functions that use linear regression. 
    The function above uses matrices that ill-conditioned using linear regression. 
    Therefore we use the next equations:  [logdetΣX-logdet(Σ1|T)-logdet(Σ2|T)]
    where X = joint_cov_x1_x2
    
    inputs: 
    config: dict - contains the dimensions of the random variables and the device
    sigma_dict: dict - contains the covariance matrices needed for the calculation 
            using: create_con_matrix function in PID_util.py

            
    output: dict - contains the mutual information"""

    n0 = config['n0']
    n1 = config['n1']
    n2 = config['n2']
    device = config['device']

    joint_cov_x1_x2 = sigma_dict['joint_x1_x2']
    # Solve the linear system instead of explicitly inverting cov_t
    solve_x1 = torch.linalg.solve(sigma_dict['cov_t'], sigma_dict['cross_x1_t'].T)
    solve_x2 = torch.linalg.solve(sigma_dict['cov_t'], sigma_dict['cross_x2_t'].T)

    # Calculate conditional covariances
    cov_x1_given_t = (sigma_dict['cov_x1'] - sigma_dict['cross_x1_t'] @ solve_x1).to(device)
    cov_x2_given_t = (sigma_dict['cov_x2'] - sigma_dict['cross_x2_t'] @ solve_x2).to(device)

    mi_tri = 0.5 * (safe_logdet(joint_cov_x1_x2) - safe_logdet(cov_x1_given_t) - safe_logdet(cov_x2_given_t))
    mi_bi_1 = 0.5 * (safe_logdet(sigma_dict['cov_x1']) - safe_logdet(cov_x1_given_t))
    mi_bi_2 = 0.5 * (safe_logdet(sigma_dict['cov_x2']) - safe_logdet(cov_x2_given_t))

    return {'mi_tri':mi_tri.item(),'mi_bi_1': mi_bi_1.item(),'mi_bi_2': mi_bi_2.item(),'nume': safe_logdet(joint_cov_x1_x2).item(),'deno_1': safe_logdet(cov_x1_given_t).item(),'deno_2': safe_logdet(cov_x2_given_t).item()}



def mi_wrapper(config,sigma_dict,whiten_terms_dict,tri_variate=True):
    """This function is a wrapper for the mutual information calculation functions. 
    It takes in the config and sigma_dict and calls the appropriate function based on the mi_type argument.
    
    inputs: 
    config: dict - contains the dimensions of the random variables and the device
    sigma_dict: dict - contains the covariance matrices needed for the calculation 
            using: create_con_matrix function in PID_util.py
    mi_type: str - type of mutual information to calculate, either 'not_whiten' or 'lr'
    
    output: dict - contains the mutual information and the numerator and denominator of the calculation"""

    mi_type = config['mi_type'] if tri_variate else config['bi_mi_type']
    if mi_type == 'whiten':
        mi = calcualte_mi(config,whiten_terms_dict)
    elif mi_type == 'lr':
        mi = calculate_mi_lr(config,sigma_dict)
    else:
        raise ValueError(f"Invalid mi_type: {mi_type}. Must be either 'not_whiten' or 'lr'.")
    
    return mi




def pid_components(pid_config,print_results=False):
    """Calculate PID components with the known components. 
    Input: 
        pid_config: dict - contains all need mutual information (I(T,X1), I(T,X2), I(T,X1,X2)) and 
        at least one of the PID components (redundancy, synergy, unique1, unique2)
        
    Output:
        pid_dict: dict - contains all PID components calculated from the known components and the mutual information values"""
    
    mi_tri = pid_config['mi_tri']
    mi_bi_1 = pid_config['mi_bi_1']
    mi_bi_2 = pid_config['mi_bi_2']
    redundancy = pid_config.get('red', None)
    synergy = pid_config.get('sy', None)
    unique1 = pid_config.get('unq1', None)
    unique2 = pid_config.get('unq2', None)

    if redundancy is not None:
        unique1 = mi_bi_1 - redundancy
        unique2 = mi_bi_2 - redundancy
        synergy = mi_tri - unique1 - unique2 - redundancy
    elif synergy is not None:
        unique1 = mi_bi_1 - (mi_tri - synergy - mi_bi_2)
        unique2 = mi_bi_2 - (mi_tri - synergy - mi_bi_1)
        redundancy = mi_bi_1 - unique1
    elif unique1 is not None:
        redundancy = mi_bi_1 - unique1
        unique2 = mi_bi_2 - redundancy
        synergy = mi_tri - unique1 - unique2 - redundancy
    elif unique2 is not None:
        redundancy = mi_bi_2 - unique2
        unique1 = mi_bi_1 - redundancy
        synergy = mi_tri - unique1 - unique2 - redundancy
    else:
        raise ValueError("At least one of redundancy, synergy, unique1, or unique2 must be provided in pid_config.")
    
    pid_dict = {
        'mi_tri': mi_tri,
        'mi_bi_1': mi_bi_1,
        'mi_bi_2': mi_bi_2,
        'redundancy': redundancy,
        'unique1': unique1,
        'unique2': unique2,
        'synergy': synergy
    }
    if print_results:
        print("PID Components:")
        for key, value in pid_dict.items():
            print(f"  {key}: {value:.6f}")
    return pid_dict