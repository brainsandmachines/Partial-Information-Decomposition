"""End-to-end tests of the config-driven runner on a tiny toy dataset."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline.rank_selection import run as runner

CONFIG = {
    "experiment_name": "unit",
    "data": {
        "source": "toy",
        "fixed": {"n_images": 120, "n_voxels": 15, "true_rank": 3, "n_subjects": 2, "seed": 0},
        "grid": {"noise_type": ["isotropic", "correlated"]},
    },
    "methods": {
        "eigenvector": {"basis": "pca", "folds": 4, "max_components": 8},
        "repeat_reconstruction": {"basis": "pca", "folds": 4, "max_components": 8},
        "mle": {},
        "pls": {"max_rank": 8, "folds": 4},
    },
    "seeds": [0, 1],
}


def _write_config(tmp_path: Path, config: dict) -> Path:
    """
    Write a config whose results go to tmp_path.

    Input:
        tmp_path: Path, pytest temporary directory.
        config: dict, runner config without ``output_root``.

    Output:
        path: Path of the written YAML file.
    """

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**config, "output_root": str(tmp_path / "results")}))
    return path


def test_runner_writes_results_config_copy_and_summary(tmp_path: Path) -> None:
    """
    Run every task, keep an exact config copy, and summarize every result.

    Input:
        tmp_path: Path, pytest temporary directory.

    Output:
        None. Assertions on the result tree and summary.
    """

    config_path = _write_config(tmp_path, CONFIG)
    runner.main(["--config", str(config_path)])

    experiment = tmp_path / "results" / "unit"
    assert (experiment / "config.yaml").read_bytes() == config_path.read_bytes()
    assert (experiment / "run_info.yaml").exists()
    summary = pd.read_csv(experiment / "summary.csv")
    # 2 datasets x (3 seeded methods x 2 seeds + mle once) = 14 results
    assert len(summary) == 14
    assert set(summary["dataset"]) == {"toy_noise_type=isotropic", "toy_noise_type=correlated"}
    assert (summary["status"] == "done").all()
    assert (summary["true_rank"] == 3).all()
    result_csv = experiment / "toy_noise_type=isotropic" / "pls_seed1.csv"
    table = pd.read_csv(result_csv)
    assert {"rank", "score", "standard_error", "is_selected"} <= set(table.columns)
    assert (experiment / "toy_noise_type=isotropic" / "overlay.pdf").exists()
    assert (experiment / "rank_recovery.pdf").exists()


def test_runner_resumes_and_refuses_a_different_config(tmp_path: Path) -> None:
    """
    Skip finished results on rerun; refuse to mix results from another config.

    Input:
        tmp_path: Path, pytest temporary directory.

    Output:
        None.
    """

    config_path = _write_config(tmp_path, CONFIG)
    runner.main(["--config", str(config_path), "--task-index", "0"])
    first = tmp_path / "results" / "unit" / "toy_noise_type=isotropic" / "eigenvector_seed0.json"
    written = first.stat().st_mtime_ns
    runner.main(["--config", str(config_path), "--task-index", "0"])
    assert first.stat().st_mtime_ns == written

    changed = {**CONFIG, "seeds": [5]}
    with pytest.raises(ValueError, match="different config"):
        runner.main(["--config", str(_write_config(tmp_path, changed))])


def test_runner_records_skipped_tasks_and_filters_methods(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """
    Record why a method cannot run on a dataset, and restrict tasks with --only.

    Input:
        tmp_path: Path, pytest temporary directory.
        capsys: pytest fixture capturing printed output.

    Output:
        None.
    """

    config = {**CONFIG, "data": {"source": "toy", "fixed": {"n_images": 12, "n_voxels": 20,
                                                            "true_rank": 2, "seed": 0}}}
    config_path = _write_config(tmp_path, config)
    runner.main(["--config", str(config_path), "--only", "mle", "pls", "--list"])
    listing = capsys.readouterr().out
    assert "mle_seedNA" in listing and "pls_seed0" in listing and "eigenvector" not in listing

    runner.main(["--config", str(config_path), "--only", "mle", "pls"])
    row = json.loads((tmp_path / "results" / "unit" / "toy" / "mle_seedNA.json").read_text())
    assert row["status"] == "skipped" and "n_images >= n_voxels" in row["reason"]
    pls_row = json.loads((tmp_path / "results" / "unit" / "toy" / "pls_seed0.json").read_text())
    assert pls_row["status"] == "skipped" and "second subject" in pls_row["reason"]
