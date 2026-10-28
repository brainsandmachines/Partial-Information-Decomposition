"""Run rank-selection methods on toy or OTC datasets from one YAML config.

    python -m pipeline.rank_selection.run --config pipeline/rank_selection/configs/smoke.yaml
    python -m pipeline.rank_selection.run --config ... --list
    python -m pipeline.rank_selection.run --config ... --task-index 3   # one SLURM array task
    python -m pipeline.rank_selection.run --config ... --summarize
    python -m pipeline.rank_selection.run --config ... --only cross_decomposition --list

Results go to ``<output_root>/<experiment_name>/``: an exact copy of the config
(``config.yaml``), ``run_info.yaml``, ``summary.csv``, and one folder per dataset
holding ``<method>_seed<k>.csv`` (curve), ``.json`` (scalars), and ``.pdf``.
A folder whose ``config.yaml`` differs from the requested config is refused;
with an identical config, finished results are skipped and only missing ones run.
With ``--task-index`` (SLURM arrays) the summary is not rebuilt; run
``--summarize`` once all tasks have finished.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.rank_selection import plotting
from pipeline.rank_selection.data import RankData, load_otc, load_otc_pair, make_toy
from pipeline.rank_selection.registry import (
    METHODS,
    get_run_function,
    option_names,
    unmet_requirement,
    uses_seed,
)
from pipeline.rank_selection.result import RankResult

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class Task:
    """One method variant on one dataset with one seed.

    Fields: dataset (dict, dataset spec with ``name`` and ``kind``), method
        (str), options (dict), label (str, file-safe variant name), seed (int or
        None).
    """

    dataset: dict
    method: str
    options: dict
    label: str
    seed: int | None

    def stem(self) -> str:
        """Return the result file stem ``<label>_seed<k>``.

        Input: none.
        Output: str.
        """
        return f"{self.label}_seed{'NA' if self.seed is None else self.seed}"


def _expand_options(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand list-valued options into every combination.

    Input: options, dict whose values may be lists.
    Output: list[dict], one dict per combination (a single dict if no lists).
    """
    keys = list(options)
    values = [value if isinstance(value, list) else [value] for value in options.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _format_value(value: Any) -> str:
    """Format an option or parameter value for labels.

    Input: value, any YAML scalar.
    Output: str.
    """
    return str(value).replace("/", "-").replace(" ", "")


def expand_methods(config: dict[str, Any]) -> list[tuple[str, dict, str]]:
    """List the method variants requested by the config.

    Input: config, dict with ``methods: {name: options-dict or list of dicts}``.
    Output: list of (method name, options dict, file-safe label). Labels name
        the options whose values differ between a method's variants.
    """
    variants = []
    for name, entries in config["methods"].items():
        if name not in METHODS:
            raise KeyError(f"Unknown method {name!r}; choose from {sorted(METHODS)}.")
        entries = [{}] if entries is None else entries
        entries = entries if isinstance(entries, list) else [entries]
        expanded = [combo for entry in entries for combo in _expand_options(entry or {})]
        try:
            allowed = set(option_names(name))
        except ImportError:
            allowed = None  # unavailable here; validated where the method runs
        for options in expanded:
            unknown = set(options) - allowed if allowed is not None else set()
            if unknown:
                raise KeyError(f"Method {name!r} has no options {sorted(unknown)}; allowed: {sorted(allowed)}.")
        varying = [
            key for key in dict.fromkeys(k for options in expanded for k in options)
            if len({_format_value(options.get(key)) for options in expanded}) > 1
        ]
        for options in expanded:
            suffix = "-".join(f"{key}={_format_value(options.get(key))}" for key in varying)
            variants.append((name, options, f"{name}-{suffix}" if suffix else name))
    return variants


def expand_datasets(config: dict[str, Any]) -> list[dict[str, Any]]:
    """List the datasets requested by the config.

    Input: config with ``data``. Toy: ``{source: toy, fixed: {...}, grid:
        {param: [values]}}`` (``make_toy`` parameters). OTC: ``{source: otc,
        subjects: [...], image_set, store, pairs: all_ordered | [[s, t], ...],
        pair_image_set}``.
    Output: list of dataset specs (dict with ``name``, ``kind`` in toy / otc /
        otc_pair, and the loader parameters).
    """
    data = config["data"]
    if data["source"] == "toy":
        fixed, grid = data.get("fixed", {}) or {}, data.get("grid", {}) or {}
        specs = []
        for combo in _expand_options(grid):
            name = "toy" + "".join(f"_{key}={_format_value(value)}" for key, value in combo.items())
            specs.append({"name": name, "kind": "toy", "params": {**fixed, **combo}})
        return specs
    if data["source"] == "otc":
        subjects = list(data["subjects"])
        common = {"store": data["store"]}
        specs = [
            {"name": subject, "kind": "otc", "subject": subject,
             "image_set": data.get("image_set", "unique"), **common}
            for subject in subjects
        ]
        pairs = data.get("pairs", "all_ordered")
        if pairs == "all_ordered":
            pairs = list(itertools.permutations(subjects, 2))
        for source, target in pairs or []:
            specs.append({
                "name": f"{source}_to_{target}", "kind": "otc_pair", "source": source,
                "target": target, "image_set": data.get("pair_image_set", "shared"), **common,
            })
        return specs
    raise ValueError("data.source must be 'toy' or 'otc'.")


def build_tasks(config: dict[str, Any]) -> list[Task]:
    """Combine datasets, method variants, and seeds into an ordered task list.

    Input: config dict.
    Output: list[Task]. Pair methods run only on pair datasets (or toys with
        two subjects); single-subject methods never run on OTC pair datasets.
        Seed-independent variants get one task with seed None.
    """
    seeds = config.get("seeds", [0])
    tasks = []
    for dataset in expand_datasets(config):
        for name, options, label in expand_methods(config):
            if METHODS[name].needs_pair != (dataset["kind"] == "otc_pair") and dataset["kind"] != "toy":
                continue
            for seed in (seeds if uses_seed(name, options) else [None]):
                tasks.append(Task(dataset, name, options, label, seed))
    return tasks


def load_dataset(spec: dict[str, Any]) -> RankData:
    """Load or generate one dataset from its spec.

    Input: spec, dict from ``expand_datasets``.
    Output: RankData named ``spec["name"]``.
    """
    if spec["kind"] == "toy":
        data = make_toy(**spec["params"])
    elif spec["kind"] == "otc":
        data = load_otc(spec["subject"], spec["image_set"], spec["store"])
    else:
        data = load_otc_pair(spec["source"], spec["target"], spec["image_set"], spec["store"])
    data.name = spec["name"]
    return data


def method_true_rank(method: str, shared_rank: int | None, signal_rank: int | None) -> int | None:
    """Return the rank a method should recover on toy data.

    Inputs: method, str registry name; shared_rank, int or None, the rank both
        subjects share; signal_rank, int or None, one subject's stimulus-driven
        rank (shared + private).
    Output: int or None. Pair methods (PLS) see only what both subjects share;
        every other method sees one subject's full signal.
    """
    return shared_rank if METHODS[method].needs_pair else signal_rank


def toy_truths(experiment_dir: Path) -> dict[str, tuple[int, int]]:
    """Read each toy dataset's shared and single-subject signal ranks from the saved config.

    Input: experiment_dir, Path containing ``config.yaml``.
    Output: dict mapping dataset name to (shared_rank, signal_rank); empty
        for OTC experiments.
    """
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        return {}
    truths = {}
    for spec in expand_datasets(yaml.safe_load(config_path.read_text())):
        if spec["kind"] == "toy":
            shared = int(spec["params"].get("true_rank", 5))
            truths[spec["name"]] = (shared, shared + int(spec["params"].get("private_rank", 0)))
    return truths


def _git_state() -> dict[str, Any]:
    """Return the repository commit and whether the working tree is dirty.

    Input: none.
    Output: dict with ``git_commit`` (str or None) and ``git_dirty`` (bool or None).
    """
    try:
        commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _environment_info() -> dict[str, Any]:
    """Collect where and with what this process runs.

    Input: none.
    Output: dict with host, date, Python and package versions, SLURM job IDs,
        and the git state.
    """
    versions = {}
    for package in ("numpy", "scipy", "sklearn", "torch", "pandas"):
        module = sys.modules.get(package)
        version = getattr(module, "__version__", None) if module else None
        versions[package] = None if version is None else str(version)
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "packages": versions,
        **_git_state(),
    }


def prepare_experiment_dir(config: dict[str, Any], config_path: Path) -> Path:
    """Create the experiment folder with an exact copy of the config.

    Inputs: config, parsed dict; config_path, Path of the YAML file.
    Output: Path of ``<output_root>/<experiment_name>``. Raises ValueError when
        the folder already holds a different ``config.yaml``.
    """
    output_root = Path(config.get("output_root") or DEFAULT_OUTPUT_ROOT)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    experiment_dir = output_root / config["experiment_name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    saved_config = experiment_dir / "config.yaml"
    if saved_config.exists():
        if saved_config.read_bytes() != config_bytes:
            raise ValueError(
                f"{experiment_dir} was produced by a different config. Use a new "
                f"experiment_name, or restore the config saved in {saved_config}."
            )
    else:
        temporary = saved_config.with_name(f".config.{os.getpid()}.tmp")
        temporary.write_bytes(config_bytes)
        temporary.replace(saved_config)
    run_info = experiment_dir / "run_info.yaml"
    if not run_info.exists():
        run_info.write_text(yaml.safe_dump({"config_source": str(config_path.resolve()),
                                            **_environment_info()}, sort_keys=False))
    return experiment_dir


def _json_ready(value: Any) -> Any:
    """Convert NumPy scalars for JSON.

    Input: value, any.
    Output: JSON-serializable value.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_task(task: Task, experiment_dir: Path, cache: dict[str, RankData]) -> dict[str, Any]:
    """Run one task unless its result exists; write its CSV, JSON, and PDF.

    Inputs: task, Task; experiment_dir, Path; cache, dict of loaded datasets
        (reused across tasks in one process).
    Output: dict, the task's summary row (status ``done``, ``skipped``, or
        ``existing``).
    """
    folder = experiment_dir / task.dataset["name"]
    json_path = folder / f"{task.stem()}.json"
    if json_path.exists():
        print(f"Exists, skipping: {json_path}")
        return {**json.loads(json_path.read_text()), "status": "existing"}
    if task.dataset["name"] not in cache:
        cache[task.dataset["name"]] = load_dataset(task.dataset)
    data = cache[task.dataset["name"]]
    base_row = {"dataset": data.name, "method": task.label, "seed": task.seed}
    reason = unmet_requirement(task.method, task.options, data)
    if reason is not None:
        print(f"Skipping {task.label} on {data.name}: {reason}")
        row = {**base_row, "status": "skipped", "reason": reason}
        folder.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(row, indent=1))
        return row

    options = dict(task.options)
    if task.method == "eigenvector" and options.get("folds") == "loo":
        options.setdefault("checkpoint_path", str(folder / f"{task.stem()}_checkpoint.csv"))
    if task.seed is not None:
        options["seed"] = task.seed
    print(f"Running {task.label} (seed {task.seed}) on {data.name}...")
    start = time.perf_counter()
    result: RankResult = get_run_function(task.method)(data, **options)
    runtime = time.perf_counter() - start
    if data.info.get("source") == "toy":
        result.true_rank = method_true_rank(
            task.method, data.info["shared_rank"], data.info["signal_rank"]
        )

    folder.mkdir(parents=True, exist_ok=True)
    result.to_table().to_csv(folder / f"{task.stem()}.csv", index=False)
    plotting.plot_result(result, folder / f"{task.stem()}.pdf")
    environment = _environment_info()
    row = {
        **result.summary(),
        "method": task.label,
        "method_detail": result.method,
        "seed": task.seed,
        "status": "done",
        "runtime_seconds": runtime,
        "options": json.dumps(task.options),
        **{key: environment[key] for key in ("host", "slurm_job_id", "git_commit", "git_dirty")},
    }
    row = {key: _json_ready(value) for key, value in row.items()}
    temporary = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(row, indent=1))
    temporary.replace(json_path)
    print(f"{task.label} on {data.name}: selected rank {result.selected_rank}.")
    return row


def summarize(experiment_dir: Path) -> pd.DataFrame:
    """Collect every finished result into summary.csv and overview plots.

    Input: experiment_dir, Path.
    Output: pd.DataFrame, one row per finished task (also written to
        ``summary.csv``). Writes ``<dataset>/overlay.pdf`` per dataset,
        one ``method_grids/<condition>.pdf`` per condition (every method, mean
        over seeds), and
        ``rank_recovery.pdf`` when toy ground truth exists. Toy true ranks are
        taken from the saved config for each method (shared rank for pair
        methods, shared + private for the others).
    """
    truths = toy_truths(experiment_dir)
    rows = [json.loads(path.read_text()) for path in sorted(experiment_dir.glob("*/*.json"))]
    for row in rows:
        if row.get("dataset") in truths:
            row["true_rank"] = method_true_rank(row["method"].split("-")[0], *truths[row["dataset"]])
    summary = pd.DataFrame(rows)
    temporary = experiment_dir / f".summary.{os.getpid()}.tmp"
    summary.to_csv(temporary, index=False)
    temporary.replace(experiment_dir / "summary.csv")
    curves = []
    for folder in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
        results = []
        for json_path in sorted(folder.glob("*.json")):
            row = json.loads(json_path.read_text())
            if row.get("status") != "done":
                continue
            if row["dataset"] in truths:
                row["true_rank"] = method_true_rank(row["method"].split("-")[0], *truths[row["dataset"]])
            table = pd.read_csv(json_path.with_suffix(".csv"))
            results.append(RankResult(
                method=row["method_detail"], dataset=row["dataset"], seed=row["seed"],
                selected_rank=row["selected_rank"], selection_rule=row["selection_rule"],
                score_name=row["score_name"], rank=table["rank"].to_numpy(),
                score=table["score"].to_numpy(), true_rank=row.get("true_rank"),
            ))
            curves.append({
                "method": row["method"], "method_detail": row["method_detail"],
                "condition": re.sub(r"_seed=[^_]+", "", row["dataset"]),
                "score_name": row["score_name"], "rank": table["rank"].to_numpy(),
                "score": table["score"].to_numpy(), "selected_rank": row["selected_rank"],
                "true_rank": row.get("true_rank"),
            })
        if results:
            plotting.plot_dataset_overlay(results, folder / "overlay.pdf")
    if curves:
        plotting.plot_condition_grids(curves, experiment_dir / "method_grids")
    if not summary.empty and "true_rank" in summary:
        plotting.plot_rank_recovery(summary, experiment_dir / "rank_recovery.pdf")
    print(f"Summary of {len(summary)} results: {experiment_dir / 'summary.csv'}")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Input: argv, list[str] or None (defaults to sys.argv).
    Output: None. Lists tasks, runs one or all tasks, or summarizes.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--list", action="store_true", help="print the task list and exit")
    parser.add_argument("--task-index", type=int, help="run only this task (SLURM array index)")
    parser.add_argument("--summarize", action="store_true", help="only rebuild summary.csv and plots")
    parser.add_argument("--only", nargs="+", metavar="METHOD",
                        help="keep only these methods (task indices refer to the filtered list)")
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    config = yaml.safe_load(config_path.read_text())
    tasks = build_tasks(config)
    if args.only:
        unknown = set(args.only) - set(METHODS)
        if unknown:
            raise KeyError(f"Unknown methods in --only: {sorted(unknown)}.")
        tasks = [task for task in tasks if task.method in args.only]
    if args.list:
        for index, task in enumerate(tasks):
            print(f"{index:4d}  {task.dataset['name']:40s}  {task.stem()}")
        print(f"{len(tasks)} tasks (SLURM: --array=0-{len(tasks) - 1})")
        return
    experiment_dir = prepare_experiment_dir(config, config_path)
    if not args.summarize:
        cache: dict[str, RankData] = {}
        selected = tasks if args.task_index is None else [tasks[args.task_index]]
        unavailable = []
        for task in selected:
            try:
                run_task(task, experiment_dir, cache)
            except ImportError as error:
                # Not saved, so the task runs once started in the right environment.
                print(f"Unavailable here, not run: {task.stem()} on {task.dataset['name']}: {error}")
                unavailable.append(task)
        if unavailable and args.task_index is not None:
            raise SystemExit(1)
        if args.task_index is not None:
            print("Single task done; run --summarize after all tasks finish.")
            return
    summarize(experiment_dir)


if __name__ == "__main__":
    main()
