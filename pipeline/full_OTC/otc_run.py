"""Run the full-OTC PID experiment from the YAML config beside this file."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from pipeline.full_OTC.otc_experiment import run_otc_experiment


if __name__ == "__main__":
    config_path = Path(__file__).with_name("otc_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = run_otc_experiment(config)
