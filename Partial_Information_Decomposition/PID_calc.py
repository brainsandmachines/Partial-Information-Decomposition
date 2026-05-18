import torch 
import numpy as np
from Idep_multivariate_gauss import Idep_multivariate_gauss
from pathlib import Path
import sys
from PID_util import create_cov_matrix
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from gpid import tilde_pid






def pid_calc(config=None,sources=None,target=None,rng=torch.Generator().manual_seed(56),method=None,on_rvs:callable=None,covariance:torch.Tensor = None):


    assert method is not None, "Please specify a method for PID calculation (e.g., 'idep', 'idep_tilde'...etc)"
    if on_rvs is not None:
        print("\nApplying on_rvs transformation to sources...")
        sources = on_rvs(sources)
    
    if method == "idep":
        pid,mi = pid_idep_wrapper(config,sources,target,covariance=covariance,rng=rng,on_rvs=on_rvs)

    elif method == "tilde":
        pid, mi = pid_tilde_wrapper(config=config,sources=sources,target=target,covariance=covariance,rng=rng,on_rvs=on_rvs)
    else:
        raise ValueError("Unsupported method specified")

    return pid, mi 


def pid_idep_wrapper(config,sources=None,target=None,covariance=None,rng=None,on_rvs=None):
    """This function is a wrapper to PID calculated by Idep_multivariate_gauss class, which implements the Idep PID calculation for multivariate Gaussian variables. This wrapper allows us to use the same input format for both Idep and BROJA implementations, and also allows us to apply transformations to the random variables before PID calculation if needed.
        if covariance is provided, it will used to calculate the PID directly from the covariance matrix without sampling. If covariance is not provided, the PID will be calculated from the sampled data covariance.
        
    Inputs:
        config: dict, configuration dictionary containing parameters for PID calculation
        sources: list of torch tensors, the source variables
        target: list of torch tensors, the target variable
        covariance: torch tensor, the covariance matrix
        rng: torch random generator, for reproducibility
        on_rvs: callable, a function to apply on the random variables (sources and targets) before PID calculation, if not None. 
        This can be used to apply transformations to the random
        variables.
    Outputs:
        pid: dict, containing the PID components (unique, redundant, synergistic)
        mi: dict, containing the mutual information values (I(X1;T), I(X2;T), I(X1,X2;T))
    """

    text = "Idep PID calculation with covariance provided" if covariance is not None else "Idep PID calculation without covariance, using sample covariance"
    print(f"\n{text}...")
    bias_corr = config['bias_correction'] if covariance is None else False
    idep = Idep_multivariate_gauss(config,rng,sources,target,bias_correction=bias_corr,cov_matrix=covariance)
    pid,mi = idep.idep()
    return pid, mi


def pid_tilde_wrapper(config:dict,sources:list,target:list,covariance:torch.Tensor,rng:torch.random.Generator,on_rvs:callable=None):
    """This function is a wrapper to PID calculated by BROJA and implemented by Venkatesh et al. 2023
    Because Idep and BROJA have different input format, this wrapper converts the input format to fit the BROJA implementation and then calls the PID calculation function.
    and calculates the PID using BROJA definition and calculation from Venkateh et al. 2023.
    
    Inputs: 
        config: dict, configuration dictionary containing parameters for PID calculation
        sources: list of torch tensors, the source variables
        target: list of torch tensors, the target variable
        covariance: torch tensor, the covariance matrix
        rng: torch random generator, for reproducibility
        on_rvs: callable, a function to apply on the random variables (sources and targets) before PID calculation, if not None. This can be used to apply transformations to the random"""

    dm , dx, dy = config['dt'] , config['dx1'] , config['dx2']
    
    
    if covariance is None:
        data = [target[0], sources[0], sources[1]]  # [T, X1, X2]
        dict_cov = create_cov_matrix(data)
        cov = dict_cov["full_cov"]
        N = data[0].shape[0]  # Sample size
        bias_corr = True if config['bias_correction'] else False
    else:
        cov = covariance
        N = config['n_samples']
        bias_corr = False
    print(f"\n Covariance matrix (shape {cov.shape}):")

    cov = cov.cpu().numpy() # Convert to numpy array for BROJA implementation
    
    output = tilde_pid.exact_gauss_tilde_pid(cov,dm,dx,dy,unbiased=bias_corr,sample_size=N) 
    pid = {'red': output[7], 'unq1': output[7], 'unq2': output[6], 'syn': output[8]}
    mi = {'I(X1;T)': output[0], 'I(X2;T)': output[1], 'I(X1,X2;T)': output[2]}

    return pid, mi