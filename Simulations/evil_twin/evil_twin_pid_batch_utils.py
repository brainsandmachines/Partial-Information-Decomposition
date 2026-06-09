"""Seed-loop and CSV helpers for evil-twin PID_calc sweeps."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Partial_Information_Decomposition.PID_calc import pid_calc
from Simulations.evil_twin.covariance_example import evil_twin_example_torch

DEFAULT_METHODS = ("idep", "tilde", "delta", "flow")
CSV_FIELDS = (
    "seed",
    "twin",
    "method",
    "n",
    "p",
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
SUMMARY_VALUE_FIELDS = ("red", "unq1", "unq2", "syn", "tri_mi", "bi_mi_1", "bi_mi_2")
SUMMARY_TWINS = ("sonic", "shadow")


def summary_csv_path(output_dir: Path, prefix: str = "evil_twin_pid") -> Path:
    """Build the output CSV path for the mean summary table.

    Inputs:
        output_dir: Path, directory where the summary CSV should be stored.
        prefix: str, filename prefix used before "summary".

    Outputs:
        Path, CSV path for the summary table.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{prefix}_summary.csv"


def method_csv_path(output_dir: Path, method: str, prefix: str = "evil_twin_pid") -> Path:
    """Build the output CSV path for one PID method.

    Inputs:
        output_dir: Path, directory where method CSV files should be stored.
        method: str, PID method name accepted by pid_calc.
        prefix: str, filename prefix used before the method name.

    Outputs:
        Path, CSV path for the requested method.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{prefix}_{method}.csv"


def append_rows_to_csv(path: Path, rows: list[dict]) -> Path:
    """Append rows to a CSV file, creating the header when needed.

    Inputs:
        path: Path, CSV file path.
        rows: list[dict], rows keyed by CSV_FIELDS.

    Outputs:
        Path, the CSV path that was written.
    """
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return path


def write_rows_to_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    """Write a complete CSV table, replacing any existing file.

    Inputs:
        path: Path, CSV file path.
        rows: list[dict], rows to write.
        fieldnames: list[str], ordered CSV columns.

    Outputs:
        Path, the CSV path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_pid_config(n: int, p: int, device: str, flow_epochs: int, flow_verbose: bool) -> dict:
    """Create the config dictionary expected by PID_calc wrappers.

    Inputs:
        n: int, number of samples.
        p: int, dimension of each source and target random variable.
        device: str, torch device name.
        flow_epochs: int, number of epochs for the flow PID method.
        flow_verbose: bool, whether flow PID should print training logs.

    Outputs:
        dict, PID_calc-compatible configuration.
    """
    return {
        "n_samples": n,
        "dx1": p,
        "dx2": p,
        "dt": p,
        "device": device,
        "bias_correction": True,
        "n_epochs": flow_epochs,
        "verbose": flow_verbose,
    }


def result_row(seed: int, twin: str, method: str, n: int, p: int, pid: dict, mi: dict) -> dict:
    """Create a successful result row for one twin and PID method.

    Inputs:
        seed: int, seed used to generate the evil-twin samples.
        twin: str, twin label such as "sonic" or "shadow".
        method: str, PID method name.
        n: int, number of samples.
        p: int, random-variable dimension.
        pid: dict, PID components returned by pid_calc.
        mi: dict, mutual information values returned by pid_calc.

    Outputs:
        dict, CSV row with PID and MI values.
    """
    return {
        "seed": seed,
        "twin": twin,
        "method": method,
        "n": n,
        "p": p,
        "status": "ok",
        "red": pid.get("red"),
        "unq1": pid.get("unq1"),
        "unq2": pid.get("unq2"),
        "syn": pid.get("syn"),
        "tri_mi": mi.get("tri_mi"),
        "bi_mi_1": mi.get("bi_mi_1"),
        "bi_mi_2": mi.get("bi_mi_2"),
        "error": "",
    }


def error_row(seed: int, twin: str, method: str, n: int, p: int, error: Exception) -> dict:
    """Create an error row for one failed twin and PID method.

    Inputs:
        seed: int, seed used to generate the evil-twin samples.
        twin: str, twin label such as "sonic" or "shadow".
        method: str, PID method name.
        n: int, number of samples.
        p: int, random-variable dimension.
        error: Exception, exception raised by pid_calc.

    Outputs:
        dict, CSV row with status "error" and the exception text.
    """
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "seed": seed,
            "twin": twin,
            "method": method,
            "n": n,
            "p": p,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def run_seed(seed: int, config: dict, methods: tuple[str, ...], output_dir: Path, csv_prefix: str) -> dict:
    """Run all requested PID methods for one evil-twin seed and save CSV rows.

    Inputs:
        seed: int, seed used for sample generation.
        config: dict, PID_calc-compatible configuration containing n_samples, dimensions, and device.
        methods: tuple[str, ...], PID method names accepted by pid_calc.
        output_dir: Path, directory where method CSV files are stored.
        csv_prefix: str, filename prefix for output CSVs.

    Outputs:
        dict, nested results keyed by method and twin with PID_calc outputs or errors.
    """
    device = config["device"]
    generator = torch.Generator(device=device).manual_seed(seed)
    data = evil_twin_example_torch(
        generator=generator,
        n=config["n_samples"],
        p=config["dx1"],
        device=device,
        dtype=torch.float64,
    )

    seed_results = {}
    for method in methods:
        method_rows = []
        seed_results[method] = {}
        for twin, (x1, x2, target) in data.items():
            try:
                pid, mi = pid_calc(
                    config=config,
                    sources=[x1, x2],
                    target=[target],
                    rng=generator,
                    method=method,
                )
                seed_results[method][twin] = {"pid": pid, "mi": mi}
                method_rows.append(result_row(seed, twin, method, config["n_samples"], config["dx1"], pid, mi))
            except Exception as error:
                seed_results[method][twin] = {"error": error}
                method_rows.append(error_row(seed, twin, method, config["n_samples"], config["dx1"], error))
        append_rows_to_csv(method_csv_path(output_dir, method, csv_prefix), method_rows)
    return seed_results


def summary_fieldnames(twins: tuple[str, ...] = SUMMARY_TWINS) -> list[str]:
    """Build ordered column names for the mean summary table.

    Inputs:
        twins: tuple[str, ...], twin labels included in the summary.

    Outputs:
        list[str], ordered summary table field names.
    """
    fields = ["method"]
    for twin in twins:
        fields.extend(f"{twin}_{name}_mean" for name in SUMMARY_VALUE_FIELDS)
        fields.extend((f"{twin}_n_ok", f"{twin}_n_error"))
    return fields


def mean_summary_rows(seed_results: dict, methods: tuple[str, ...], twins: tuple[str, ...] = SUMMARY_TWINS) -> list[dict]:
    """Calculate mean PID and MI values across seeds for each method.

    Inputs:
        seed_results: dict, nested results keyed by seed, method, and twin.
        methods: tuple[str, ...], PID method names to summarize.
        twins: tuple[str, ...], twin labels to summarize separately.

    Outputs:
        list[dict], one summary row per PID method with mean PID and MI columns.
    """
    rows = []
    for method in methods:
        row = {"method": method}
        for twin in twins:
            twin_results = [
                result[method][twin]
                for result in seed_results.values()
                if method in result and twin in result[method]
            ]
            ok_results = [result for result in twin_results if "pid" in result and "mi" in result]
            error_results = [result for result in twin_results if "error" in result]
            for field in SUMMARY_VALUE_FIELDS:
                values = []
                for result in ok_results:
                    source = result["pid"] if field in result["pid"] else result["mi"]
                    values.append(float(source[field]))
                row[f"{twin}_{field}_mean"] = sum(values) / len(values) if values else ""
            row[f"{twin}_n_ok"] = len(ok_results)
            row[f"{twin}_n_error"] = len(error_results)
        rows.append(row)
    return rows


def save_summary_csv(output_dir: Path, prefix: str, rows: list[dict]) -> Path:
    """Save the mean summary table to a CSV file.

    Inputs:
        output_dir: Path, directory where the summary CSV should be stored.
        prefix: str, filename prefix for the summary CSV.
        rows: list[dict], summary rows from mean_summary_rows.

    Outputs:
        Path, written summary CSV path.
    """
    return write_rows_to_csv(summary_csv_path(output_dir, prefix), rows, summary_fieldnames())


def summary_image_path(output_dir: Path, prefix: str, twin: str) -> Path:
    """Build the output image path for one twin's mean summary table.

    Inputs:
        output_dir: Path, directory where the summary image should be stored.
        prefix: str, filename prefix for the image.
        twin: str, twin label such as "sonic" or "shadow".

    Outputs:
        Path, image path for the requested twin summary table.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{prefix}_{twin}_summary.png"


def summary_rows_to_pid_results(rows: list[dict], twin: str) -> dict:
    """Convert mean summary rows to the PID result shape used by RVs_Story tables.

    Inputs:
        rows: list[dict], summary rows from mean_summary_rows.
        twin: str, twin label to extract from the summary rows.

    Outputs:
        dict, results keyed by PID method and valued as (pid, mi) tuples.
    """
    results = {}
    for row in rows:
        if row.get(f"{twin}_n_ok", 0) == 0:
            continue
        pid = {
            "red": row[f"{twin}_red_mean"],
            "unq1": row[f"{twin}_unq1_mean"],
            "unq2": row[f"{twin}_unq2_mean"],
            "syn": row[f"{twin}_syn_mean"],
        }
        mi = {
            "tri_mi": row[f"{twin}_tri_mi_mean"],
            "bi_mi_1": row[f"{twin}_bi_mi_1_mean"],
            "bi_mi_2": row[f"{twin}_bi_mi_2_mean"],
        }
        results[row["method"].title() if row["method"] != "flow" else "Flow"] = (pid, mi)
    return results


def save_summary_table_images(
    rows: list[dict],
    output_dir: Path,
    prefix: str,
    config: dict,
    twins: tuple[str, ...] = SUMMARY_TWINS,
) -> list[Path]:
    """Save RVs_Story-style PID comparison images for the mean summary.

    Inputs:
        rows: list[dict], summary rows from mean_summary_rows.
        output_dir: Path, directory where summary images should be stored.
        prefix: str, filename prefix for each image.
        config: dict, PID_calc-compatible configuration used for table metadata.
        twins: tuple[str, ...], twin labels to save as separate images.

    Outputs:
        list[Path], image paths written to disk.
    """
    from Partial_Information_Decomposition.PID_util import save_pid_comparison_table

    image_paths = []
    table_config = {
        "n": config["n_samples"],
        "p": config["dx1"],
        "dx1": config["dx1"],
        "dx2": config["dx2"],
        "dt": config["dt"],
        "bias_correction": config["bias_correction"],
    }
    for twin in twins:
        results = summary_rows_to_pid_results(rows, twin)
        if not results:
            continue
        image_path = summary_image_path(output_dir, prefix, twin)
        save_pid_comparison_table(
            results,
            save_path=str(image_path),
            title=f"Evil Twin {twin.title()} Mean PID Summary",
            config=table_config,
        )
        image_paths.append(image_path)
    return image_paths


def format_summary_value(value, decimals: int) -> str:
    """Format one summary table cell for terminal output.

    Inputs:
        value: object, numeric or string cell value.
        decimals: int, number of decimal places for floats.

    Outputs:
        str, formatted table cell.
    """
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def format_summary_table(rows: list[dict], decimals: int = 6) -> str:
    """Format summary rows as an aligned plain-text table.

    Inputs:
        rows: list[dict], summary rows from mean_summary_rows.
        decimals: int, number of decimal places for floats.

    Outputs:
        str, formatted summary table.
    """
    if not rows:
        return "No summary rows."

    headers = summary_fieldnames()
    formatted_rows = [
        [format_summary_value(row.get(header, ""), decimals) for header in headers]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in formatted_rows))
        for index, header in enumerate(headers)
    ]
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in formatted_rows
    ]
    return "\n".join([header_line, divider, *body])


def run_evil_twin_pid_sweep(
    seeds: list[int],
    n: int = 1000,
    p: int = 1,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    output_dir: Path | str = Path("simulation_results/evil_twin_pid"),
    device: str = "cpu",
    flow_epochs: int = 250,
    flow_verbose: bool = False,
    csv_prefix: str = "evil_twin_pid",
) -> dict:
    """Run PID_calc methods on Sonic and Shadow across multiple seeds.

    Inputs:
        seeds: list[int], seeds to run.
        n: int, number of samples per seed.
        p: int, dimension of each source and target random variable.
        methods: tuple[str, ...], PID method names accepted by pid_calc.
        output_dir: Path | str, directory where method CSV files are stored.
        device: str, torch device name.
        flow_epochs: int, number of training epochs for flow PID.
        flow_verbose: bool, whether flow PID should print training logs.
        csv_prefix: str, filename prefix for method CSV files.

    Outputs:
        dict, nested in-memory results keyed by seed, method, and twin.
    """
    output_path = Path(output_dir)
    config = make_pid_config(n, p, device, flow_epochs, flow_verbose)
    all_results = {}
    for seed in seeds:
        all_results[seed] = run_seed(seed, config, methods, output_path, csv_prefix)
    summary_rows = mean_summary_rows(all_results, methods)
    summary_path = save_summary_csv(output_path, csv_prefix, summary_rows)
    image_paths = save_summary_table_images(summary_rows, output_path, csv_prefix, config)
    print("\nMean PID/MI summary across seeds")
    print(format_summary_table(summary_rows))
    print(f"\nSaved summary CSV to: {summary_path}")
    for image_path in image_paths:
        print(f"Saved summary image to: {image_path}")
    return all_results
