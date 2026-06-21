from voxel_experiment import run_voxel_experiment
import torch
import numpy as np 
import yaml
from pathlib import Path
import sys
from functools import partial
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root)) 

from pipeline_phases import report_results







if __name__ == "__main__":
    
    #Load config
    config_path = Path("/home/ohadshee/Desktop/Partial-Information-Decomposition/pipeline/voxel_experiments/voxel_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    #Run experiment
    results = run_voxel_experiment(config)