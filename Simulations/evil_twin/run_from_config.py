"""Run the evil-twin PID experiment from an editable Python configuration."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from external.gpid.src.gpid import estimate
from Partial_Information_Decomposition.PID_calc import pid_calc
from Partial_Information_Decomposition.PID_util import create_cov_matrix
from Simulations.evil_twin.covariance_example import evil_twin_example_torch
from Simulations.evil_twin.evil_twin_pid_batch_utils import (
    CSV_FIELDS,
    error_row,
    format_summary_table,
    make_pid_config,
    mean_summary_rows,
    method_csv_path,
    result_row,
    save_summary_csv,
    summary_csv_path,
    write_rows_to_csv,
)


# Edit only this dictionary to configure the experiment.
CONFIG = {
    "seeds": [0],
    "n_samples": 10000,
    "dimension": 100,
    "methods": ("eigen",'tilde','idep'),
    "bias_correction": False,
    "device": "cpu",
    "output_dir": "simulation_results/evil_twin_config",
    "csv_prefix": "evil_twin",
    "overwrite_outputs": True,
}

SUPPORTED_METHODS = ("idep", "tilde", "delta",'eigen')


def validate_config(config: dict) -> None:
    """Validate the editable evil-twin experiment configuration.

    Inputs:
        config: dict, experiment settings containing seeds, sample count,
            dimension, methods, bias-correction mode, device, and output names.

    Outputs:
        None. Raises TypeError or ValueError when a setting is invalid.
    """
    required_keys = {
        "seeds",
        "n_samples",
        "dimension",
        "methods",
        "bias_correction",
        "device",
        "output_dir",
        "csv_prefix",
        "overwrite_outputs",
    }
    missing_keys = required_keys.difference(config)
    if missing_keys:
        raise ValueError(f"Missing CONFIG keys: {sorted(missing_keys)}")

    seeds = config["seeds"]
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise TypeError("CONFIG['seeds'] must be a non-empty list or tuple of integers")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise TypeError("Every seed must be an integer")

    n_samples = config["n_samples"]
    dimension = config["dimension"]
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError("CONFIG['n_samples'] must be a positive integer")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError("CONFIG['dimension'] must be a positive integer")
    if n_samples <= 3 * dimension:
        raise ValueError("n_samples must exceed 3 * dimension for a full-rank empirical covariance")

    methods = config["methods"]
    if not isinstance(methods, (list, tuple)) or not methods:
        raise TypeError("CONFIG['methods'] must be a non-empty list or tuple")
    unsupported_methods = sorted(set(methods).difference(SUPPORTED_METHODS))
    if unsupported_methods:
        raise ValueError(
            f"Unsupported methods: {unsupported_methods}. Supported methods: {SUPPORTED_METHODS}"
        )
    if len(set(methods)) != len(methods):
        raise ValueError("CONFIG['methods'] must not contain duplicates")

    if not isinstance(config["bias_correction"], bool):
        raise TypeError("CONFIG['bias_correction'] must be True or False")
    if not isinstance(config["device"], str) or not config["device"]:
        raise TypeError("CONFIG['device'] must be a non-empty string")
    if not isinstance(config["output_dir"], (str, Path)):
        raise TypeError("CONFIG['output_dir'] must be a string or pathlib.Path")
    if not isinstance(config["csv_prefix"], str) or not config["csv_prefix"]:
        raise TypeError("CONFIG['csv_prefix'] must be a non-empty string")
    if not isinstance(config["overwrite_outputs"], bool):
        raise TypeError("CONFIG['overwrite_outputs'] must be True or False")


def uncorrected_delta_pid(
    config: dict,
    sources: list[torch.Tensor],
    target: list[torch.Tensor],
) -> tuple[dict, dict]:
    """Calculate raw Delta PID without applying Wishart bias correction.

    Inputs:
        config: dict, PID configuration containing dt, dx1, and dx2 dimensions.
        sources: list[torch.Tensor], two sample matrices shaped
            (n_samples, source_dimension).
        target: list[torch.Tensor], one sample matrix shaped
            (n_samples, target_dimension).

    Outputs:
        tuple[dict, dict], PID components and mutual-information values in bits.

    Notes:
        The existing Delta wrapper always subtracts Wishart bias for sample
        inputs. This local path exposes the solver's raw values without changing
        that shared wrapper.
    """
    dm = config["dt"]
    dx = config["dx1"]
    dy = config["dx2"]
    covariance = create_cov_matrix([target[0], sources[0], sources[1]])["full_cov"]
    output = estimate.approx_pid_from_cov(
        covariance.detach().cpu().numpy(),
        dm,
        dx,
        dy,
        verbose=False,
    )
    imx, imy, imxy, _, _, uix, uiy, redundancy, synergy = output[:9]
    pid = {
        "red": redundancy,
        "unq1": uix,
        "unq2": uiy,
        "syn": synergy,
    }
    mi = {
        "tri_mi": imxy,
        "bi_mi_1": imx,
        "bi_mi_2": imy,
    }
    return pid, mi


def calculate_pid(
    config: dict,
    sources: list[torch.Tensor],
    target: list[torch.Tensor],
    generator: torch.Generator,
    method: str,
) -> tuple[dict, dict]:
    """Run one configured PID method on one evil-twin sample set.

    Inputs:
        config: dict, PID configuration including dimensions and bias mode.
        sources: list[torch.Tensor], two source sample matrices.
        target: list[torch.Tensor], one target sample matrix.
        generator: torch.Generator, seeded generator for reproducible PID calls.
        method: str, one of "idep", "tilde", or "delta".

    Outputs:
        tuple[dict, dict], PID components and mutual-information values in bits.
    """
    if method == "delta" and not config["bias_correction"]:
        return uncorrected_delta_pid(config, sources, target)
    return pid_calc(
        config=config,
        sources=sources,
        target=target,
        rng=generator,
        method=method,
    )


def run_from_config(config: dict) -> dict:
    """Run the configured evil-twin experiment and save method and summary CSVs.

    Inputs:
        config: dict, validated experiment configuration. The dimension applies
            equally to source 1, source 2, and the target.

    Outputs:
        dict, containing nested results by seed/method/twin and written CSV paths.
    """
    validate_config(config)

    seeds = list(config["seeds"])
    methods = tuple(config["methods"])
    n_samples = config["n_samples"]
    dimension = config["dimension"]
    device = config["device"]
    output_dir = Path(config["output_dir"])
    bias_label = "bias_corrected" if config["bias_correction"] else "uncorrected"
    csv_prefix = f"{config['csv_prefix']}_{bias_label}"

    method_paths = {
        method: method_csv_path(output_dir, method, csv_prefix)
        for method in methods
    }
    expected_paths = [*method_paths.values(), summary_csv_path(output_dir, csv_prefix)]
    if not config["overwrite_outputs"]:
        existing_paths = [str(path) for path in expected_paths if path.exists()]
        if existing_paths:
            raise FileExistsError(
                "Output files already exist and overwrite_outputs is False: "
                + ", ".join(existing_paths)
            )

    pid_config = make_pid_config(
        n=n_samples,
        p=dimension,
        device=device,
        flow_epochs=0,
        flow_verbose=False,
    )
    pid_config["bias_correction"] = config["bias_correction"]

    all_results = {}
    rows_by_method = {method: [] for method in methods}
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        twin_data = evil_twin_example_torch(
            generator=generator,
            n=n_samples,
            p=dimension,
            device=device,
            dtype=torch.float64,
        )
        all_results[seed] = {}
        for method in methods:
            all_results[seed][method] = {}
            for twin, (source1, source2, target) in twin_data.items():
                try:
                    pid, mi = calculate_pid(
                        config=pid_config,
                        sources=[source1, source2],
                        target=[target],
                        generator=generator,
                        method=method,
                    )
                    all_results[seed][method][twin] = {"pid": pid, "mi": mi}
                    row = result_row(seed, twin, method, n_samples, dimension, pid, mi)
                except Exception as error:
                    all_results[seed][method][twin] = {"error": error}
                    row = error_row(seed, twin, method, n_samples, dimension, error)
                rows_by_method[method].append(row)

    for method, rows in rows_by_method.items():
        write_rows_to_csv(method_paths[method], rows, list(CSV_FIELDS))

    summary_rows = mean_summary_rows(
        all_results,
        methods,
        n_samples=n_samples,
        dimension=dimension,
        bias_correction=config["bias_correction"],
    )
    summary_path = save_summary_csv(output_dir, csv_prefix, summary_rows)
    print("\nEvil-twin configuration")
    print(
        f"seeds={seeds}, n_samples={n_samples}, dimension={dimension}, "
        f"bias_correction={config['bias_correction']}"
    )
    print("\nMean PID/MI summary across seeds")
    print(format_summary_table(summary_rows))
    print(f"\nSaved summary CSV to: {summary_path}")
    for method, path in method_paths.items():
        print(f"Saved {method} rows to: {path}")

    return {
        "results": all_results,
        "method_csvs": method_paths,
        "summary_csv": summary_path,
    }


def main() -> dict:
    """Run the module-level evil-twin configuration.

    Inputs:
        No inputs. Uses the module-level CONFIG dictionary.

    Outputs:
        dict, containing nested results by seed/method/twin and written CSV paths.
    """
    return run_from_config(CONFIG)


if __name__ == "__main__":
    main()
