import yaml
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from pipeline.voxel_experiments.voxel_experiment import run_voxel_experiment

if __name__ == "__main__":
    
    #Load config
    config_path = Path(__file__).with_name("voxel_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    #Run experiment
    results = run_voxel_experiment(config)
