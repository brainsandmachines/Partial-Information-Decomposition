"""Run the full-OTC PID experiment from the YAML config beside this file."""


import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from pipeline.full_OTC.otc_experiment import run_otc_experiment


from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="otc_config",
)
def main(cfg: DictConfig) -> None:
    """Run the OTC experiment using the Hydra configuration."""

    config: dict[str, Any] = OmegaConf.to_container(
        cfg,
        resolve=True,
    )

    results = run_otc_experiment(config)


if __name__ == "__main__":
    main()