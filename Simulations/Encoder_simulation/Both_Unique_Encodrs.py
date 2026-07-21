from Simulations.Encoder_simulation.both_unique import feature_creation, test_both_unique
import numpy as np
import torch
import sys
from pathlib import Path

from encoding_model.suppression_core import create_predictions
from my_utils import extract_all_components, run_configured_multiseed
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root)) 
from Commonality_Analysis.CA import commonality_analysis
from Partial_Information_Decomposition.Idep.Idep_multivariate_gauss import Idep_multivariate_gauss
from supression_effect.Suppresed_Encoder import (load_model_and_fmri,prepare_inputs,load_model_and_fmri)
from Partial_Information_Decomposition.PID_util import compare_results





def get_run_config() -> dict:
    return {
        "data_dir": "/mnt/data4tb/data_algonauts/",
        "parent_submission_dir": "/mnt/data4tb/data_algonauts/submissions",
        "subj": 1,
        "method": "ridge_cv",
        "n_s": 7000,
        "n_f": 500,
        "rng_seed": np.random.default_rng(seed=30),
        "n_seeds": 10000,
        "seed_start": 0,
        "snr": 5,
        "unique_ratio": 0.5,
        "redundant_dim": 0.2,
        "verbose": True,
        "path_to_load": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/trained_models/roi_models/FBA-1_alexnet_features.8_subj01_1.pth/FBA-1_alexnet_features.8_subj01.pth_encoding_model.joblib",
        "fmri_dict_path": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/encoding_model/fmri_dicts/subj1_fmri_dicts.joblib",
        "roi_name": "FBA-1",
        "results_dir": "/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/simulation_results/Both_Unique_Encoder",
        "results_prefix": "seed_summary",
        "all_runs_results_prefix": "seed_runs",
        "progress_print_every": 1,
        "test_name": 'TMI_BC_Both_Unique_Encoder_No_Orthogonal',  # Optional: specify a custom name for the summary file; if None, uses timestampW
    }


def creature_featurs(rng,snr,unique_ratio,features,signal,redundant_dim=None):
    n, p = features.shape
    r_head = int(p * redundant_dim)
    d_unique = int((p-r_head)*unique_ratio)

    R = features[:,:r_head]
    z1 = features[:, r_head:r_head+ d_unique]
    z2 = features[:, r_head+ d_unique:]
    
    assert R.shape[1] + z1.shape[1] + z2.shape[1] == p , f"Total features {p} does not match sum of redundant and unique features {R.shape[1] + z1.shape[1] + z2.shape[1]}"
    q = rng.standard_normal((p, p))
    R = R @ q[:r_head,:]
    X_M1 = R + z1 @ q[r_head:r_head+ d_unique,:]
    X_M2 = R + z2 @ q[r_head+ d_unique:,:]

    noise_1 = rng.standard_normal(X_M1.shape)
    noise_2 = rng.standard_normal(X_M2.shape)

    X_M1 += noise_1 * np.std(X_M1) / snr
    X_M2 += noise_2 * np.std(X_M2) / snr
    
    std = np.std(signal)
    noise_std = std.item() / snr
    signal_dim1 , signal_dim2 = signal.shape[0], signal.shape[1]
    target = signal +  noise_std * rng.standard_normal((signal_dim1 , signal_dim2))
    return X_M1, X_M2, target




def run_single_seed(seed:int, config:dict,features: torch.Tensor,fmri_dict:dict) -> dict:

    #Create encoder and selected features based on the config and the loaded fmri_dict
    encoder , selected_features = prepare_inputs(config, features,fmri_dict=fmri_dict)

    y_hat_lh, y_hat_rh = create_predictions(encoder, reg_rh=None, features=selected_features)

    if config['verbose']:
        print("Predictions created.\nPredicted fMRI shape (LH): ", y_hat_lh.shape) if y_hat_lh is not None else None
        print("\nPredicted fMRI shape (RH): ", y_hat_rh.shape) if y_hat_rh is not None else None

    X_M1, X_M2, target = creature_featurs(rng=config["rng_seed"],snr=config["snr"],unique_ratio=config["unique_ratio"],features=selected_features,signal=y_hat_lh,redundant_dim=config["redundant_dim"])

    ca = commonality_analysis(X_M1, X_M2, target, method=config["method"])
    ca_results = ca.ca(X_M1, X_M2, target, method=config["method"], alphas=None)
    print('\nDone calculating commonality analysis.')
    m1 = torch.tensor(X_M1, dtype=torch.float64)
    m2 = torch.tensor(X_M2, dtype=torch.float64)
    t = torch.tensor(target, dtype=torch.float64)

    sources = [m1, m2]
    targets = [t]
    idep_class = Idep_multivariate_gauss(sources, targets, bias_correction=False)
    pid_results, mi_results = idep_class.idep()
    return extract_all_components(ca_results, pid_results, mi_results)




def main():
    config = get_run_config()
    features, fmri_dict = load_model_and_fmri(config)
    run_configured_multiseed(
        config,
        per_seed_runner=lambda seed, config: run_single_seed(seed, config, features, fmri_dict),
    )


if __name__ == "__main__":

    #Run Single Seed
    # real_features, fmri_dict = load_model_and_fmri(get_run_config())
    # ca_results, pid,mi = run_single_seed(seed=0, config=get_run_config(), features=real_features, fmri_dict=fmri_dict)
    # compare_results(ca_results, pid, mi)

    #Run Multi Seed
    main()
