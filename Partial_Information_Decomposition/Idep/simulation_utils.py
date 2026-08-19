"""Small reusable data and result helpers retained from legacy Idep simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from Partial_Information_Decomposition.output_utils import safe_filename


def to_python_scalar(value: Any) -> Any:
    """Convert torch/NumPy scalar-like values to serializable Python values.

    Inputs:
        value: Any scalar, array, tensor, or already-serializable object.

    Outputs:
        A Python scalar/list when conversion is needed, otherwise ``value``.
    """

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.numpy().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()
    return value


def flatten_pid_results(pid_results: dict) -> dict[str, Any]:
    """Flatten one level of nested PID result dictionaries.

    Inputs:
        pid_results: dict mapping PID names to scalar values or metric dictionaries.

    Outputs:
        dict[str, Any] with keys such as ``red_mean`` and Python-native values.
    """

    flattened: dict[str, Any] = {}
    for key, value in pid_results.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened[f"{key}_{sub_key}"] = to_python_scalar(sub_value)
        else:
            flattened[key] = to_python_scalar(value)
    return flattened


def get_pid_ver_csv_path(
    output_folder: str | Path,
    pid_ver: str,
    csv_title: str = "pid_results",
) -> Path:
    """Build the result CSV path for one PID definition.

    Inputs:
        output_folder: str or Path destination directory.
        pid_ver: str PID definition used as a filename suffix.
        csv_title: str experiment filename prefix.

    Outputs:
        Path to ``<safe-title>_<safe-pid-version>.csv``.
    """

    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe_filename(csv_title)}_{safe_filename(pid_ver)}.csv"


def append_row_to_csv(
    row: dict,
    output_folder: str | Path,
    csv_title: str = "pid_results",
) -> Path:
    """Append one simulation row to its PID-specific CSV.

    Inputs:
        row: dict containing a required ``pid_ver`` field.
        output_folder: str or Path destination directory.
        csv_title: str experiment filename prefix.

    Outputs:
        Path to the CSV that received the row.
    """

    if "pid_ver" not in row:
        raise ValueError("row must contain a 'pid_ver' key.")
    output_csv = get_pid_ver_csv_path(output_folder, row["pid_ver"], csv_title)
    pd.DataFrame([row]).to_csv(
        output_csv,
        mode="a",
        header=not output_csv.exists(),
        index=False,
    )
    return output_csv


def already_exists_in_csv(
    output_folder: str | Path,
    n_samples: int,
    dimensions: tuple[int, int, int] | list[int],
    pid_ver: str,
    seed: int,
    csv_title: str = "pid_results",
) -> bool:
    """Check whether one exact simulation setting is already recorded.

    Inputs:
        output_folder: str or Path containing per-method result CSVs.
        n_samples: int sample count stored in column ``N``.
        dimensions: source-1, source-2, and target dimensions.
        pid_ver: str PID method identifier.
        seed: int simulation seed.
        csv_title: str experiment filename prefix.

    Outputs:
        bool indicating whether a matching row exists.
    """

    output_csv = get_pid_ver_csv_path(output_folder, pid_ver, csv_title)
    if not output_csv.exists():
        return False
    frame = pd.read_csv(output_csv)
    required = {"N", "dx1", "dx2", "dt", "pid_ver", "seed"}
    if not required.issubset(frame.columns):
        return False
    mask = (
        (frame["N"] == n_samples)
        & (frame["dx1"] == dimensions[0])
        & (frame["dx2"] == dimensions[1])
        & (frame["dt"] == dimensions[2])
        & (frame["pid_ver"] == pid_ver)
        & (frame["seed"] == seed)
    )
    return bool(mask.any())


def sample_data_from_cov(
    config: dict,
    true_cov: torch.Tensor,
    rng: np.random.Generator | torch.Generator | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sample Gaussian RVs and their unbiased empirical covariance.

    Inputs:
        config: dict containing dimensions, sample count, and torch device.
        true_cov: torch.Tensor population covariance shaped ``(D, D)``.
        rng: optional generator retained for API compatibility; torch sampling
            follows the active torch random state, as in the legacy helper.

    Outputs:
        tuple containing empirical covariance ``(D, D)`` and tensors
        ``[X1, X2, T]`` shaped ``(N, dx1)``, ``(N, dx2)``, and ``(N, dt)``.
    """

    del rng
    dx1, dx2, dt = config["dx1"], config["dx2"], config["dt"]
    dimension = true_cov.shape[0]
    covariance = true_cov.to(device=config["device"], dtype=torch.float64)
    mean = torch.zeros(dimension, device=config["device"], dtype=torch.float64)  # dimension D -> (D,)
    data = torch.distributions.MultivariateNormal(mean, covariance).sample(
        (config["n_samples"],)
    )  # distribution D, sample count N -> (N, D)
    source_1 = data[:, :dx1]  # (N, D) -> (N, dx1)
    source_2 = data[:, dx1 : dx1 + dx2]  # (N, D) -> (N, dx2)
    target = data[:, dx1 + dx2 : dx1 + dx2 + dt]  # (N, D) -> (N, dt)
    sample_covariance = torch.cov(data.T, correction=1)  # (N, D) -> (D, D)
    return sample_covariance, [source_1, source_2, target]


def make_pre_config(
    exp: str,
    mi_config: dict,
    mi0_config: dict,
    above0_m7_mi_config: dict,
    above0_m8_mi_config: dict,
    n_p_config: dict,
    unknown_config: dict,
    de_config: dict | None = None,
) -> dict:
    """Merge the configuration fragments for one Idep simulation regime.

    Inputs:
        exp: str regime name such as ``MI=0``, ``M7_MI>0``, or ``unknown``.
        mi_config: dict common mutual-information settings.
        mi0_config: dict settings for zero mutual information.
        above0_m7_mi_config: dict positive-MI M7 settings.
        above0_m8_mi_config: dict positive-MI M8 settings.
        n_p_config: dict sample-size and dimension sweep settings.
        unknown_config: dict settings for the unknown regime.
        de_config: optional dict for the only-unq1-zero experiment.

    Outputs:
        dict containing the merged configuration for ``exp``.
    """

    variants = {
        "MI=0": mi0_config,
        "M7_MI>0": above0_m7_mi_config,
        "M8_MI>0": above0_m8_mi_config,
        "unknown": unknown_config,
    }
    config = {**mi_config, **variants.get(exp, {}), **n_p_config}
    if config.get("ver") == "only_unq1_zero" and de_config:
        config.update(de_config)
    return config
