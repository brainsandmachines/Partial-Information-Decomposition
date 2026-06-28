from typing import Any
import torch
import numpy as np
from sklearn.decomposition import PCA
import sys
from pathlib import Path

import yaml


root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from full_OTC.otc_experiment import run_otc_experiment



"""This is a wrapper that will output a graph that 
represents the PID computation as a function of the number 
of PCA components used to reduce the dimensionality of the source features.
"""






def pca_as_function(pipeline_config: str | Path, pca_config: str | Path) -> dict[str, Any]:
    """Run the full-OTC PID experiment from the YAML config beside this file.
    
    target_components is constant"""



    n_components_list = pca_config["n_components_list"]
    pid_results_dict = {}
    mi_results_dict = {}

    for n_components in n_components_list:
        print(f"\nRunning PID computation with {n_components} PCA components...")
        # Update the pipeline config with the current number of PCA components
        pipeline_config["feature_manipulation_kwargs"]["n_components_source_1"] = n_components
        #pipeline_config["feature_manipulation_kwargs"]["n_components_source_2"] = n_components
    
        # Run the OTC experiment with the updated config
        results = run_otc_experiment(pipeline_config) #This is the context
        broja_results = results['pid_results'] #{'mi':{bi_mi_1_t,},'pid:{unq1,unq2,red,syn}'}
        mi_results = broja_results['mi']
        pid_results = broja_results['pid']


        pid_results_dict[n_components] = pid_results
        mi_results_dict[n_components] = mi_results


    return pid_results_dict, mi_results_dict




def plot_(results_dict: dict[str, Any], pca_config: str,pipeline_config: str ) -> None:
    """Plot the results of the PID computation as a function of the number of PCA components."""
    import matplotlib.pyplot as plt

    save_fig_path = Path(pca_config["save_path"])
    target_components = pipeline_config["feature_manipulation_kwargs"]["n_components_target"]
    model_name_1 = pipeline_config["sources_kwargs"]["model_name_1"]
    model_name_2 = pipeline_config["sources_kwargs"]["model_name_2"]
    n_components_list = pca_config["n_components_list"]
    unique_info_source_1 = [results_dict[n]["unq1"] for n in n_components_list]
    unique_info_source_2 = [results_dict[n]["unq2"] for n in n_components_list]
    shared_info = [results_dict[n]["red"] for n in n_components_list]
    synergistic_info = [results_dict[n]["syn"] for n in n_components_list]
    title = (
    "PID Computation as a Function of PCA Components\n"
    f"Target and Source 2 {model_name_2} Fixed Components: {target_components}\n VS"
    f"Source 1: {model_name_1}\n")

    plt.figure(figsize=(10, 6))
    plt.plot(n_components_list, unique_info_source_1, label="Unique Info Source 1", marker='o')
    plt.plot(n_components_list, unique_info_source_2, label="Unique Info Source 2", marker='o')
    plt.plot(n_components_list, shared_info, label="Shared Info", marker='o')
    plt.plot(n_components_list, synergistic_info, label="Synergistic Info", marker='o')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Information (bits)")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(save_fig_path,bbox_inches="tight")
    return



if __name__ == "__main__":
    pipeline_config_path = Path("/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/full_OTC/otc_config.yaml")
    pca_config_path = Path("/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/analysis/pca_analysis/pca_config.yaml")

    with open(pipeline_config_path, "r") as f:
        pipeline_config = yaml.safe_load(f)

    with open(pca_config_path, "r") as f:
        pca_config = yaml.safe_load(f)

    pid_results_dict, mi_results_dict = pca_as_function(pipeline_config, pca_config)
    plot_(pid_results_dict, pca_config, pipeline_config)