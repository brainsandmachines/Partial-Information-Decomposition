import torch 
import numpy as np
from pathlib import Path
import sys
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))
wrapper_root = root / "library_wrappers"
if str(wrapper_root) not in sys.path:
    sys.path.insert(0, str(wrapper_root))
from Partial_Information_Decomposition.PID_util import create_cov_matrix
from external.gpid.src.gpid import estimate
from external.gpid.src.gpid import tilde_pid
from Flow_PID import load_flow_pid
from Thin_PID import load_exact_gauss_thin_pid
from Partial_Information_Decomposition.bias_functions import  mi_wishart_bias
from Partial_Information_Decomposition.mi_functions import calculate_mi_raw






def pid_calc(config=None,sources=None,target=None,rng=torch.Generator().manual_seed(56),method=None,on_rvs:callable=None,covariance:torch.Tensor = None):


    assert method is not None, "Please specify a method for PID calculation (e.g., 'idep', 'idep_tilde'...etc)"
    if on_rvs is not None:
        print("\nApplying on_rvs transformation to sources...")
        sources = on_rvs(sources)
    
    if method == "idep":
        pid,mi = pid_idep_wrapper(config,sources,target,covariance=covariance,rng=rng,on_rvs=on_rvs)

    elif method == "tilde":
        pid, mi = pid_tilde_wrapper(config=config,sources=sources,target=target,covariance=covariance,rng=rng,on_rvs=on_rvs)
    
    elif method == "delta":
        pid, mi = delta_wrapper(config=config,sources=sources,target=target,covariance=covariance,rng=rng,on_rvs=on_rvs)
    
    elif method in ("flow", "flow_pid"):
        pid, mi = flow_pid_wrapper(config=config,sources=sources,target=target,covariance=covariance,rng=rng,on_rvs=on_rvs)
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

    #text = "Idep PID calculation with covariance provided" if covariance is not None else "Idep PID calculation without covariance, using sample covariance"
    #print(f"\n{text}...")
    from Partial_Information_Decomposition.Idep.Idep_multivariate_gauss import Idep_multivariate_gauss

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
    #print(f"\n Covariance matrix (shape {cov.shape}):")

    cov = cov.cpu().numpy() # Convert to numpy array for BROJA implementation
    
    output = tilde_pid.exact_gauss_tilde_pid(cov,dm,dx,dy,unbiased=bias_corr,sample_size=N) 
    imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si = output[:9]
    pid = {'red': ri, 'unq1': uix, 'unq2': uiy, 'syn': si}
    mi = {'tri_mi': imxy_debiased, 'bi_mi_1': imx, 'bi_mi_2': imy}

    return pid, mi


def delta_wrapper(config,sources,target,covariance,rng,on_rvs):
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
        data = [target[0], sources[0], sources[1]]  # Code Assumes - [T, X1, X2]
        dict_cov = create_cov_matrix(data)
        cov = dict_cov["full_cov"]
        N = data[0].shape[0]  # Sample size
        bias_corr = True if config['bias_correction'] else False
    else:
        cov = covariance
        N = config['n_samples']
        bias_corr = False
    #print(f"\n Covariance matrix (shape {cov.shape}):")

    cov = cov.cpu().numpy() # Convert to numpy array for Delta implementation
    
    output = estimate.approx_pid_from_cov(cov,dm,dx,dy,verbose=False) 
    imx, imy, imxy, def_y_minus_x, def_x_minus_y, uix, uiy, ri, si = output[:9]

    if sources and target is not None:
        cov_reg = create_cov_matrix(rvs=[sources[0], sources[1], target[0]],device=config.get('device', 'cpu'),dims=[dx, dy, dm])['full_cov']
        mi_dict = calculate_mi_raw(device=config.get('device', 'cpu'),sigma=cov_reg,dims=[dx, dy, dm])
        bias = mi_wishart_bias(dims=[dx, dy, dm], n_samples=N)

        mi_x1_t = mi_dict['bi_mi_1_t'] - bias['bias_mi_1_t']
        mi_x2_t = mi_dict['bi_mi_2_t'] - bias['bias_mi_2_t']
        mi_x1x2_t = mi_dict['tri_mi'] - bias['bias_tri_mi']

        imx = mi_x1_t / np.log(2)
        imy = mi_x2_t / np.log(2)
        imxy = mi_x1x2_t / np.log(2)

        ri = min(imx - def_x_minus_y, imy - def_y_minus_x)
        uix = imx - ri
        uiy = imy - ri
        si = imxy - uix - uiy - ri

    pid = {'red': ri, 'unq1': uix, 'unq2': uiy, 'syn': si}
    mi = {'tri_mi': imxy, 'bi_mi_1': imx, 'bi_mi_2': imy}
    return pid, mi


def _to_numpy_samples(data):
    """Convert torch/numpy samples to the numpy format expected by flow-pid."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def flow_pid_wrapper(config:dict,sources:list,target:list,covariance:torch.Tensor,rng:torch.random.Generator,on_rvs:callable=None):
    """Wrapper for flow-pid.

    Raw samples are sent to flow_pid. If covariance is provided, use Thin-PID
    because flow_pid is a trained sample-based estimator.
    """

    dm , dx, dy = config['dt'] , config['dx1'] , config['dx2']

    bias_correction = config['bias_correction'] if covariance is None else False

    if covariance is not None:
        exact_gauss_thin_pid = load_exact_gauss_thin_pid()
        cov = covariance.cpu().numpy() if isinstance(covariance, torch.Tensor) else np.asarray(covariance)
        expected_shape = (dm + dx + dy, dm + dx + dy)
        if cov.shape != expected_shape:
            raise ValueError(f"Flow/Thin covariance must have shape {expected_shape}, got {cov.shape}")
        # Thin-PID expects covariance blocks in [target, source1, source2] order.
        output = exact_gauss_thin_pid(cov,dm,dx,dy,unbiased=bias_correction,sample_size=None)
    else:
        flow_pid = load_flow_pid()
        m = _to_numpy_samples(target[0])
        x = _to_numpy_samples(sources[0])
        y = _to_numpy_samples(sources[1])
        if len({m.shape[0], x.shape[0], y.shape[0]}) != 1:
            raise ValueError(f"Flow-PID sample counts must match, got T={m.shape[0]}, X1={x.shape[0]}, X2={y.shape[0]}")
        if (m.shape[1], x.shape[1], y.shape[1]) != (dm, dx, dy):
            raise ValueError(
                f"Flow-PID dimensions must be (dt, dx1, dx2)=({dm}, {dx}, {dy}), "
                f"got ({m.shape[1]}, {x.shape[1]}, {y.shape[1]})"
            )

        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="flow_pid_training_") as temp_dir:
            try:
                os.chdir(temp_dir)
                output = flow_pid(
                    m,
                    x,
                    y,
                    n_flows=config.get('n_flows', 3),
                    n_epochs=config.get('n_epochs', 250),
                    batch_size=config.get('batch_size', 64),
                    lr=config.get('lr', 2e-4),
                    encoder=None,
                    verbose=config.get('verbose', False),
                    ret_t_sigt=False,
                    device=config.get('device', 'cpu'),
                )
            finally:
                os.chdir(original_cwd)

    imx, imy, imxy, _, _, uix, uiy, ri, si = output[:9]
    pid = {'red': ri, 'unq1': uix, 'unq2': uiy, 'syn': si}
    mi = {'tri_mi': imxy, 'bi_mi_1': imx, 'bi_mi_2': imy}
    return pid, mi
