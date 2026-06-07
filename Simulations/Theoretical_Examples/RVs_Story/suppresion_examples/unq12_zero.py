import torch
import sys
from pathlib import Path
import yaml
import numpy as np
from core_model import main_func
root = Path(__file__).resolve().parents[3]
sys.path.append(str(root))  

from Partial_Information_Decomposition.PID_calc import pid_calc
from Partial_Information_Decomposition.PID_util import create_cov_matrix,pid_comparison_table,save_pid_comparison_table
from Partial_Information_Decomposition.mi_functions import calculate_mi_raw
from Partial_Information_Decomposition.bias_functions import mi_wishahrt_bias



def unq12_zero(rng, n, p, noise_std):
    """
    Intended theoretical structure:

        Y  = R + U + eps_y
        X1 = R + U + N
        X2 = R     + N

    X2 has:
        - redundancy through R
        - no private signal source about Y
        - synergistic/suppressor role through shared nuisance N

    So we expect:
        unq1 = 0
        unq2 = 0
        redundancy > 0
        synergy > 0
    under MMI-like Gaussian PID.
    """

    R = rng.standard_normal((n, p))   # redundant signal
    N = noise_std * rng.standard_normal((n, p))  # shared suppressor noise
    X1_noise = noise_std * rng.standard_normal((n, p))  # private noise for X1
    X2_noise = noise_std * rng.standard_normal((n, p))  # private noise for X2
    eps_y = noise_std * rng.standard_normal((n, p))


    t = R + eps_y
    X_1 = R + noise_std*N + X1_noise
    X_2 = R + noise_std*N + X2_noise

    return X_1, X_2, t



if __name__ == "__main__":
    
    config_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Simulations/Theoretical_Examples/rv_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config_dict = config['parameters']
    results = main_func(config_dict, unq12_zero)
    save_pid_comparison_table(results,save_path=f"{config_dict['results_dir']}/unq12_zero.png",config=config)

    

