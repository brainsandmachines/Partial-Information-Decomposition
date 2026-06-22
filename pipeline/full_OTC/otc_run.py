"""Run the full-OTC PID experiment from the YAML config beside this file."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))

from otc_experiment import run_otc_experiment


if __name__ == "__main__":
    config_path = Path(__file__).with_name("otc_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = run_otc_experiment(config)
