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



def full_suppresion(rng, n, p, noise_std):
    """
    Intended theoretical structure:

        Y 
        X1 = Y + N
        X2 =  N

    X2 has:
        - no private signal source about Y
        - synergistic/suppressor role through shared nuisance N

    So we expect:
        unq2 = 0
        redundancy = 0
        unq1 > 0
        synergy > 0
    under MMI-like Gaussian PID.
    """


    
    N_t = noise_std * rng.standard_normal((n, p))  # shared suppressor noise
    N_x1 = noise_std * rng.standard_normal((n, p))  # noise for x1
    N_shared = noise_std * rng.standard_normal((n, p))  # shared noise for x1 and x2


    target = rng.standard_normal((n, p))

    t = target + N_t
    X_1 = t + N_x1 + N_shared
    X_2 = N_shared

    return X_1, X_2, t




if __name__ == "__main__":
    
    config_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Simulations/Theoretical_Examples/rv_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config_dict = config['parameters']
    results = main_func(config_dict, full_suppresion)
    save_pid_comparison_table(results,f"{config['results_dir']}/full_suppresion.png",config=config)
    
