"""Run the full-OTC PID experiment from the YAML config beside this file."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from pipeline.full_OTC.otc_experiment import run_otc_experiment


def make_str_as_path(config: dict[str, any]) -> dict[str, any]:
    """Convert string paths in the config to Path objects."""
    for key, value in config.items():
        if isinstance(value, str) and (value.startswith("/") or value.startswith(".")):
            config[key] = Path(value)
        elif isinstance(value, dict):
            config[key] = make_str_as_path(value)
    return config

def check_path_exists(config: dict[str, any]) -> None:
    """Check if the paths in the config exist."""
    for key, value in config.items():
        if isinstance(value, Path) and not value.exists():
            raise FileNotFoundError(f"Path {value} does not exist.")
        elif isinstance(value, dict):
            check_path_exists(value)

if __name__ == "__main__":
    config_name = 'ridge_otc_config'
    config_path = Path(__file__).with_name(f"{config_name}.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        if config_name == 'ridge_otc_config':
            model_name_1 = config['sources_kwargs']['model_name_1']
            model_name_2 = config['sources_kwargs']['model_name_2']

            config['feature_manipulation_kwargs']['model_name_1'] = model_name_1
            config['feature_manipulation_kwargs']['model_name_2'] = model_name_2
            config['feature_manipulation_kwargs']['seed'] = config['pid_kwargs']['rng_seed'] 

        config = make_str_as_path(config)
        check_path_exists(config)
    print(f"\nRunning full-OTC PID experiment with config: {config_name}!!!")
    results = run_otc_experiment(config)