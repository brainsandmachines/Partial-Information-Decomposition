import torch 
import numpy as np
from pathlib import Path
import sys
import os
import tempfile
from functools import partial

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))
wrapper_root = root / "library_wrappers"
if str(wrapper_root) not in sys.path:
    sys.path.insert(0, str(wrapper_root))
from Partial_Information_Decomposition.bias_functions import (
    broja_venkatesh_bias,
    equal_direct_wishart_control_obj_debias,
    lorenz_gaussian_obj_debias,
    permutation_null_debias,
)
from Partial_Information_Decomposition.PID_util import create_cov_matrix
from external.gpid.src.gpid import estimate
from external.gpid.src.gpid import tilde_pid
from Flow_PID import load_flow_pid
from Thin_PID import load_exact_gauss_thin_pid
from Partial_Information_Decomposition.mi_functions import calculate_mi_raw





def mi_wishart_bias(dims:list,n_samples:int):
    """Calculate Gaussian MI Wishart biases without loading simulation dependencies.

    Inputs:
        dims: list of three integers ordered as [source1, source2, target].
        n_samples: int, number of samples used to estimate the covariance.
    Outputs:
        bias: dict containing pairwise, joint, and source-source MI biases in nats.
    """
    if len(dims) != 3:
        raise ValueError(f"dims must have length 3. Got len(dims)={len(dims)}.")

    dx, dy, dm = dims
    df = n_samples - 1
    dimensions = (dx, dy, dm, dx + dy, dx + dm, dy + dm, dx + dy + dm)
    logdet_biases = []
    for dimension in dimensions:
        if df <= dimension - 1:
            raise ValueError(f"Need df > d-1. Got df={df}, d={dimension}.")
        indices = torch.arange(1, dimension + 1, dtype=torch.float64)  # scalar bounds -> (dimension,)
        scale = torch.tensor(2.0 / df, dtype=torch.float64)  # scalar -> ()
        bias = torch.sum(torch.special.digamma((df - indices + 1) / 2.0))
        logdet_biases.append((bias + dimension * torch.log(scale)).item())

    bias_x, bias_y, bias_m, bias_xy, bias_xm, bias_ym, bias_xym = logdet_biases
    return {
        'bias_mi_1_t': 0.5 * (bias_x + bias_m - bias_xm),
        'bias_mi_2_t': 0.5 * (bias_y + bias_m - bias_ym),
        'bias_tri_mi': 0.5 * (bias_xy + bias_m - bias_xym),
        'bias_mi_12': 0.5 * (bias_x + bias_y - bias_xy),
    }


def pid_calc(config=None,sources=None,target=None,rng=torch.Generator().manual_seed(56),
             method=None,on_rvs:callable=None,covariance:torch.Tensor = None,param_bias = False):


    assert method is not None, "Please specify a method for PID calculation (e.g., 'idep', 'idep_tilde'...etc)"
    if sources is not None and target is not None:
        dx1 = sources[0].shape[1]
        dx2 = sources[1].shape[1]
        dt = target[0].shape[1]
        config['dx1'] = dx1
        config['dx2'] = dx2
        config['dt'] = dt
    if on_rvs is not None:
        print("\nApplying on_rvs transformation to sources...")
        sources = on_rvs(sources)


    if not config['bias_correction']:
        if method == "tilde" and param_bias:
            if config.get('param_bias_method') == 'lorenz_gaussian_merged':
                print(
                    "\nVenkatesh's built-in correction is disabled; the "
                    "Lorenz Gaussian merged correction is enabled."
                )
            else:
                print(
                    "\nMarginal-MI bias correction is disabled; "
                    "the configured Venkatesh-objective correction is enabled."
                )
        else:
            print(f"\nWARNING: Bias correction is disabled:{config['bias_correction']} for PID calculation.")
    else: 
        print(f"\nBias correction is enabled for PID calculation:{config['bias_correction']}.")
    if method == "idep":
        print("\nCalculating PID using Idep...")
        pid,mi = pid_idep_wrapper(config,sources,target,covariance=covariance,rng=rng,on_rvs=on_rvs)

    elif method == "tilde":
        print("\nCalculating PID using Tilde...")
        pid, mi = pid_tilde_wrapper(config=config,sources=sources,target=target,covariance=covariance,rng=rng,on_rvs=on_rvs,param_bias=param_bias)
    
    elif method == "delta":
        print("\nCalculating PID using Delta...")
        pid, mi = delta_wrapper(config=config,sources=sources,target=target,covariance=covariance,rng=rng,on_rvs=on_rvs)
    
    elif method in ("thin", "thin_pid"):
        print("\nCalculating PID using Thin-PID...")
        pid, mi = thin_pid_wrapper(config=config,sources=sources,target=target,covariance=covariance,rng=rng,on_rvs=on_rvs)

    elif method in ("flow", "flow_pid"):
        print("\nCalculating PID using Flow...")
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


def pid_tilde_wrapper(config:dict,sources:list,target:list,covariance:torch.Tensor,rng:torch.random.Generator,on_rvs:callable=None,param_bias = False):
    """Calculate Gaussian BROJA/Tilde PID with an optional objective correction.

    Inputs:
        config: dict containing dimensions, sample size, bias settings, and
            optional ``param_bias_method`` selection.
        sources: list containing source X1 and X2 sample tensors, or None when
            a covariance is supplied for an uncorrected population calculation.
        target: list containing the target sample tensor, or None when a
            covariance is supplied for an uncorrected population calculation.
        covariance: optional torch.Tensor covariance ordered [T, X1, X2].
        rng: torch.Generator used by the legacy permutation correction.
        on_rvs: optional transformation callable accepted for wrapper
            compatibility.
        param_bias: bool, whether to use the selected objective-bias method.

    Outputs:
        tuple[dict, dict], where the first dict contains PID atoms, raw ``obj``,
        and the union information, and the second dict contains the three
        Gaussian mutual informations in bits.

    Notes:
        ``param_bias_method='lorenz_gaussian_merged'`` applies the Gaussian
        correction of Lorenz et al.: exact Goodman/Wishart MI correction plus
        merged resampling and target-shuffle synergy correction. The raw
        Venkatesh objective is retained as ``obj`` and corrected additively as
        ``union_info``. All corrected PID atoms are reconstructed without
        clipping. The older objective-only methods remain available.
    """

    dm , dx, dy = config['dt'] , config['dx1'] , config['dx2']
    
    
    covariance_was_supplied = covariance is not None
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

    covariance_for_bias = cov
    cov = (
        cov.detach().cpu().numpy()
        if torch.is_tensor(cov)
        else np.asarray(cov)
    )  # torch/array-like (D, D) -> NumPy (D, D)

    if not param_bias:
        print("\nCalculating PID using BROJA Tilde.")
        output = tilde_pid.exact_gauss_tilde_pid(
            cov,
            dm,
            dx,
            dy,
            unbiased=bias_corr,
            sample_size=N,
        )
        imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si = output[:9]
        pid = {'red': ri, 'unq1': uix, 'unq2': uiy, 'syn': si, 'union_info':union_info,'obj':obj}
        mi = {'tri_mi': imxy_debiased, 'bi_mi_1': imx, 'bi_mi_2': imy}
    else:
        param_bias_method = config.get('param_bias_method', 'permutation')
        print(
            "\nCalculating PID using BROJA Tilde with objective-bias method "
            f"'{param_bias_method}'."
        )
        output = tilde_pid.exact_gauss_tilde_pid(
            cov,
            dm,
            dx,
            dy,
            unbiased=False,
            sample_size=N,
        )
        imx, imy, imxy, _, obj, _, __, ___, ____ = output[:9]
        mi = {'tri_mi': imxy, 'bi_mi_1': imx, 'bi_mi_2': imy}

        if param_bias_method == 'lorenz_gaussian_merged':
            if sources is None or target is None:
                raise ValueError(
                    "The Lorenz Gaussian correction requires source and target "
                    "samples for target shuffling."
                )
            correction_config = config.copy()
            correction_config.update(
                {
                    'X1': sources[0],
                    'X2': sources[1],
                    'T': target[0],
                    'n_samples': N,
                }
            )
            correction = lorenz_gaussian_obj_debias(
                correction_config,
                covariance=covariance_for_bias,
                raw_obj=None,
            )
            obj = correction['raw_obj']
            obj_bias = correction['bias']
            corrected = correction['corrected']
            imx = corrected['mi_target_source_1']
            imy = corrected['mi_target_source_2']
            imxy = corrected['mi_target_joint_sources']
            mi = {'tri_mi': imxy, 'bi_mi_1': imx, 'bi_mi_2': imy}
            print(
                "\nLorenz Gaussian merged correction: "
                f"total raw-obj correction={obj_bias:.8f} bits, "
                f"resampling synergy bias="
                f"{correction['bias_components']['syn_resampling']:.8f}, "
                f"shuffle synergy bias="
                f"{correction['bias_components']['syn_shuffle']:.8f}."
            )
        elif param_bias_method == 'permutation':
            if sources is None or target is None:
                raise ValueError(
                    "The permutation objective correction requires source and "
                    "target samples."
                )
            X_1, X_2 = sources[0], sources[1]
            T = target[0]
            print("\nCalculating bias using permutation null distribution...")
            config['X1'] = X_1
            config['X2'] = X_2
            config['T'] = T
            bias_func = partial(broja_venkatesh_bias,config=config)
            bias = permutation_null_debias(config,func=bias_func)
            print(f"\nBias calculated using permutation null distribution: {bias}")
            obj_bias = bias['bias']
        else:
            raise ValueError(
                "param_bias_method must be 'permutation', "
                "'equal_direct_wishart_control', or "
                f"'lorenz_gaussian_merged'; got {param_bias_method!r}."
            )

        obj_debiased = obj - obj_bias

        uix = obj_debiased - imy
        uiy = obj_debiased - imx
        ri = imx + imy - obj_debiased
        si = imxy - obj_debiased
        pid = {
            'red': ri,
            'unq1': uix,
            'unq2': uiy,
            'syn': si,
            'union_info': obj_debiased,
            'obj': obj,
            'obj_bias': obj_bias,
        }
        if param_bias_method == 'lorenz_gaussian_merged':
            pid['bias_components'] = correction['bias_components']


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


def thin_pid_wrapper(config:dict,sources:list,target:list,covariance:torch.Tensor,rng:torch.random.Generator,on_rvs:callable=None):
    """Calculate Thin-PID from samples or covariance using the standard PID inputs.

    Inputs:
        config: dict containing PID dimensions and bias-correction settings.
        sources: list of two source sample tensors.
        target: list containing the target sample tensor.
        covariance: optional covariance tensor ordered as [target, source1, source2].
        rng: torch random generator (accepted for wrapper compatibility).
        on_rvs: optional transformation callable (accepted for wrapper compatibility).
    Outputs:
        pid: dict containing redundant, unique, and synergistic information.
        mi: dict containing joint and pairwise mutual information values.
    """
    dm, dx, dy = config['dt'], config['dx1'], config['dx2']
    if covariance is None:
        data = [target[0], sources[0], sources[1]]
        cov = create_cov_matrix(data)["full_cov"]
        sample_size = data[0].shape[0]
        bias_correction = config['bias_correction']
    else:
        cov, sample_size, bias_correction = covariance, None, False

    cov = cov.cpu().numpy() if isinstance(cov, torch.Tensor) else np.asarray(cov)  # (D, D) -> (D, D)
    output = load_exact_gauss_thin_pid()(cov, dm, dx, dy, unbiased=bias_correction, sample_size=sample_size)
    imx, imy, imxy, _, _, unq1, unq2, red, syn = output[:9]
    pid = {'red': red, 'unq1': unq1, 'unq2': unq2, 'syn': syn}
    mi = {'tri_mi': imxy, 'bi_mi_1': imx, 'bi_mi_2': imy}
    return pid, mi


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
                    n_epochs=config.get('n_epochs', 50),
                    batch_size=config.get('batch_size', 128),
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
