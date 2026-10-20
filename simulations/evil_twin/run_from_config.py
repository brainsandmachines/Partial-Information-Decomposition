"""Run the Sonic/Shadow Eigen-PID experiment from an editable configuration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from simulations.evil_twin.plot_results import plot_pid_atoms
from simulations.evil_twin.covariance_example import evil_twin_example_torch
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from pid.calculation import pid_calc


CONFIG = {
    "seeds": [0],
    "n_samples": 1000000,
    "dimension": 100,
    "bias_correction": False,
    "device": "cuda",
    "output_dir": "simulation_results/evil_twin_config",
    "csv_prefix": "evil_twin",
    "overwrite_outputs": True,
}

CSV_FIELDS = (
    "seed",
    "twin",
    "n",
    "p",
    "bias_correction",
    "status",
    "red",
    "unq1",
    "unq2",
    "syn",
    "tri_mi",
    "bi_mi_1",
    "bi_mi_2",
    "error",
)


def validate_config(config: dict) -> None:
    """Validate the editable evil-twin experiment configuration.

    Inputs:
        config: dict containing seeds, sample count, dimension,
            bias-correction mode, device, and output settings.

    Outputs:
        None. Raises TypeError or ValueError when a setting is invalid.
    """
    required_keys = {
        "seeds",
        "n_samples",
        "dimension",
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


def result_row(
    seed: int,
    twin: str,
    n: int,
    p: int,
    bias_correction: bool,
    pid: dict,
    mi: dict,
) -> dict:
    """Create one successful evil-twin result row.

    Inputs:
        seed: int, sample-generation seed.
        twin: str, sonic or shadow.
        n: int, number of samples.
        p: int, dimension of each random variable.
        bias_correction: bool, whether Eigen-PID correction was enabled.
        pid: dict, PID components returned by pid_calc.
        mi: dict, mutual-information values returned by pid_calc.

    Outputs:
        dict, one detailed CSV row.
    """
    return {
        "seed": seed,
        "twin": twin,
        "n": n,
        "p": p,
        "bias_correction": bias_correction,
        "status": "ok",
        "red": pid["red"],
        "unq1": pid["unq1"],
        "unq2": pid["unq2"],
        "syn": pid["syn"],
        "tri_mi": mi["tri_mi"],
        "bi_mi_1": mi["bi_mi_1"],
        "bi_mi_2": mi["bi_mi_2"],
        "error": "",
    }


def error_row(
    seed: int,
    twin: str,
    n: int,
    p: int,
    bias_correction: bool,
    error: Exception,
) -> dict:
    """Create one failed evil-twin result row.

    Inputs:
        seed: int, sample-generation seed.
        twin: str, sonic or shadow.
        n: int, number of samples.
        p: int, dimension of each random variable.
        bias_correction: bool, whether Eigen-PID correction was enabled.
        error: Exception raised during PID calculation.

    Outputs:
        dict, one detailed CSV row with error status and message.
    """
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "seed": seed,
            "twin": twin,
            "n": n,
            "p": p,
            "bias_correction": bias_correction,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def run_from_config(config: dict) -> dict:
    """Run Eigen-PID for Sonic and Shadow, save one CSV, and plot all atoms.

    Inputs:
        config: dict, validated experiment configuration. The dimension applies
            equally to source 1, source 2, and the target.

    Outputs:
        dict containing results by seed and twin, the CSV path, and plot path.
    """
    validate_config(config)

    seeds = list(config["seeds"])
    n_samples = config["n_samples"]
    dimension = config["dimension"]
    device = config["device"]
    output_dir = Path(config["output_dir"])
    bias_correction = config["bias_correction"]
    bias_label = "bias_corrected" if bias_correction else "uncorrected"
    results_path = output_dir / f"{config['csv_prefix']}_{bias_label}.csv"
    plot_path = results_path.with_name(f"{results_path.stem}_pid_atoms.pdf")

    if not config["overwrite_outputs"]:
        existing_paths = [str(path) for path in (results_path, plot_path) if path.exists()]
        if existing_paths:
            raise FileExistsError(
                "Output files already exist and overwrite_outputs is False: "
                + ", ".join(existing_paths)
            )

    pid_config = {
        "n_samples": n_samples,
        "dx1": dimension,
        "dx2": dimension,
        "dt": dimension,
        "device": device,
        "bias_correction": bias_correction,
    }
    all_results = {}
    rows = []
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
        for twin, (source1, source2, target) in twin_data.items():
            try:
                pid, mi = pid_calc(
                    config=pid_config,
                    sources=[source1, source2],
                    target=[target],
                    rng=generator,
                )
                all_results[seed][twin] = {"pid": pid, "mi": mi}
                row = result_row(
                    seed,
                    twin,
                    n_samples,
                    dimension,
                    bias_correction,
                    pid,
                    mi,
                )
            except Exception as error:
                all_results[seed][twin] = {"error": error}
                row = error_row(
                    seed,
                    twin,
                    n_samples,
                    dimension,
                    bias_correction,
                    error,
                )
            rows.append(row)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    saved_plot_path = plot_pid_atoms(results_path, plot_path)
    print(
        f"Evil-twin Eigen-PID complete: seeds={seeds}, n_samples={n_samples}, "
        f"dimension={dimension}, bias_correction={bias_correction}"
    )
    print(f"Saved detailed results to: {results_path}")
    print(f"Saved PID atom plot to: {saved_plot_path}")
    return {
        "results": all_results,
        "results_csv": results_path,
        "plot": saved_plot_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse optional command-line overrides.

    Inputs:
        No inputs; values are read from the process command line.

    Outputs:
        argparse.Namespace containing optional configuration overrides.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--n-samples", "--n", dest="n_samples", type=int)
    parser.add_argument("--dimension", "--p", dest="dimension", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--csv-prefix")
    parser.add_argument(
        "--bias-correction",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--overwrite-outputs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> dict:
    """Apply explicitly supplied command-line values to CONFIG.

    Inputs:
        args: argparse.Namespace returned by parse_args.

    Outputs:
        dict, independent experiment configuration.
    """
    config = dict(CONFIG)
    for key in (
        "seeds",
        "n_samples",
        "dimension",
        "bias_correction",
        "device",
        "output_dir",
        "csv_prefix",
        "overwrite_outputs",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def main() -> dict:
    """Run the configured evil-twin experiment.

    Inputs:
        No inputs. Uses the module-level CONFIG and command-line overrides.

    Outputs:
        dict containing results by seed and twin, the CSV path, and plot path.
    """
    return run_from_config(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
