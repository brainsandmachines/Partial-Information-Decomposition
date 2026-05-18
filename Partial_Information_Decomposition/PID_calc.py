import torch 
import numpy as np
from PID_util import *
from Partial_Information_Decomposition.Idep_multivariate_gauss import Idep_multivariate_gauss
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from gpid import tilde_pid





def pid_calc(config,sources,target,rng,method=None,on_rvs:callable=None):


    assert method is not None, "Please specify a method for PID calculation (e.g., 'idep', 'idep_tilde'...etc)"
    if on_rvs is not None:
        print("\nApplying on_rvs transformation to sources...")
        sources = on_rvs(sources)
    
    if method == "idep":
        idep = Idep_multivariate_gauss(rng,sources, target,bias_correction=config['bias_correction'])
        pid,mi = idep.idep()
    
    elif method == "idep_tilde":
        pid, mi = pid_tilde_wrapper(config,sources,target,rng,on_rvs)
    else:
        raise ValueError("Unsupported method specified")

    return pid, mi 


def pid_tilde_wrapper(config:dict,sources:list,target:list,rng:torch.random.Generator,on_rvs:callable=None):
    """This function is a wrapper to PID calculated by BROJA and implemented by Venkatesh et al. 2023
    Because Idep and BROJA have different input format, this wrapper converts the input format to fit the BROJA implementation and then calls the PID calculation function.
    and calculates the PID using BROJA definition and calculation from Venkateh et al. 2023.
    
    Inputs: 
        config: dict, configuration dictionary containing parameters for PID calculation
        sources: list of torch tensors, the source variables
        target: list of torch tensors, the target variable
        rng: torch random generator, for reproducibility
        on_rvs: callable, a function to apply on the random variables (sources and targets) before PID calculation, if not None. This can be used to apply transformations to the random"""

    dm , dx, dy = target[0].shape[1], sources[0].shape[1], sources[1].shape[1] #(target,X1,X2)
    data = [data[0], data[1], data[2]]  # [T, X1, X2]

    dict_cov = create_cov_matrix(data)
    cov = dict_cov["full_cov"]
    print(f"\n Covariance matrix (shape {cov.shape}):")
    cov = cov.cpu().numpy() # Convert to numpy array for BROJA implementation
    N = data[0].shape[0]  # Sample size

    # (imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si)
    bias_corr = True if config['bias_correction'] else False
    
    
    output = tilde_pid.exact_gauss_tilde_pid(cov,dm,dx,dy,unbiased=bias_corr,sample_size=N) 
    pid = {'red': output[7], 'unq1': output[7], 'unq2': output[6], 'syn': output[8]}
    mi = {'I(M1;T)': output[0], 'I(M2;T)': output[1], 'I(M1,M2;T)': output[2]}

    return pid, mi