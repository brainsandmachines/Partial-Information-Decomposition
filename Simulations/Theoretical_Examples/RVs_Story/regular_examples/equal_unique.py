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



def equal_unique(rng, n, p, noise_std):
    """
    Expectation:
        unq1 = unq2
        redundancy > 0
        synergy = 0
    """

    R = rng.standard_normal((n, p))   # redundant signal
    U1 = rng.standard_normal((n, p))   # unique-to-X1 signal
    U2 = rng.standard_normal((n, p))   # unique-to-X2 signal
    N1 = noise_std * rng.standard_normal((n, p))  # noise for x1
    N2 = noise_std * rng.standard_normal((n, p))  # noise for x2
    eps_y = noise_std * rng.standard_normal((n, p))

    y = R + U1 + U2 + eps_y
    X_1 = R + U1 + N1
    X_2 = R + U2 + N2

    return X_1, X_2, y






if __name__ == "__main__":
    
    config_path = "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/Simulations/Theoretical_Examples/rv_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config_dict = config['parameters']
    results = main_func(config_dict, equal_unique)
    save_pid_comparison_table(results,save_path=f"{config_dict['results_dir']}/equal_unique2.png",config=config)

    