"""Tests for the config-driven full-OTC experiment assembler."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import pipeline.full_OTC.otc_experiment as otc_experiment


def dummy_otc_target(target_offset: float = 0.0, voxel_index: int | None = None) -> dict[str, Any]:
    """Create a tiny full-OTC target context for runner tests.

    Inputs:
        target_offset: float, value added to every target sample.
        voxel_index: int or None, ignored value accepted for voxel-config symmetry.

    Output:
        target_context: dict, fake context containing a 2D full-OTC "target".
    """

    del voxel_index
    return {
        "target": [[1.0 + target_offset, 10.0], [2.0 + target_offset, 20.0]],
        "image_ids_for_subj": [0, 1],
    }


def dummy_sources(model_name_1: str, model_name_2: str) -> dict[str, dict[str, Any]]:
    """Create tiny source contexts for OTC runner tests.

    Inputs:
        model_name_1: str, first fake model name.
        model_name_2: str, second fake model name.

    Output:
        sources: dict, fake source contexts under "X1" and "X2".
    """

    return {
        "X1": {"model_name": model_name_1, "base": 100.0, "layers_ordered": ["a", "b", "c", "d"]},
        "X2": {"model_name": model_name_2, "base": 200.0, "layers_ordered": ["e", "f", "g", "h", "i"]},
    }


def dummy_feature_extraction(
    source_context: dict[str, Any],
    layer_index: int,
    target_context: dict[str, Any],
    feature_scale: float = 1.0,
) -> list[list[float]]:
    """Create fake source features from source and selected layer values.

    Inputs:
        source_context: dict, fake source context.
        layer_index: int, selected fake layer index.
        target_context: dict, fake target context with image IDs.
        feature_scale: float, scalar multiplier applied to fake features.

    Output:
        features: list[list[float]], fake feature matrix.
    """

    del target_context
    start = (source_context["base"] + layer_index) * feature_scale
    return [[start], [start + 1.0]]


def dummy_pid(
    target: list[list[float]],
    source_1: list[list[float]],
    source_2: list[list[float]],
    method: str,
) -> dict[str, Any]:
    """Return simple sums from final full-OTC target and source arrays.

    Inputs:
        target: list[list[float]], final target samples.
        source_1: list[list[float]], final X1 samples.
        source_2: list[list[float]], final X2 samples.
        method: str, fake PID method name.

    Output:
        pid_results: dict, fake PID result.
    """

    return {
        "method": method,
        "target_sum": sum(sum(row) for row in target),
        "x1_sum": sum(row[0] for row in source_1),
        "x2_sum": sum(row[0] for row in source_2),
    }


def register_dummy_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register dummy wrapper names in the OTC experiment registry.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for test isolation.

    Output:
        None. The OTC experiment function registry is patched in place.
    """

    monkeypatch.setitem(otc_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_otc_target", dummy_otc_target)
    monkeypatch.setitem(otc_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_sources", dummy_sources)
    monkeypatch.setitem(otc_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_feature", dummy_feature_extraction)
    monkeypatch.setitem(otc_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_pid", dummy_pid)


def base_config() -> dict[str, Any]:
    """Create a valid fake full-OTC experiment config.

    Inputs:
        No inputs.

    Output:
        config: dict, config using dummy and shared pipeline step names.
    """

    return {
        "functions": {
            "target_extraction": "dummy_otc_target",
            "sources_extraction": "dummy_sources",
            "choose_layer": "specific_layer_index",
            "feature_extraction": "dummy_feature",
            "preprocess": None,
            "feature_manipulation": None,
            "pid_calculation": "dummy_pid",
            "pid_report": None,
        },
        "target_kwargs": {"target_offset": 10.0},
        "sources_kwargs": {"model_name_1": "M1", "model_name_2": "M2"},
        "choose_layer_kwargs": {"X1_index": 1, "X2_index": 2},
        "feature_extraction_kwargs": {"feature_scale": 2.0},
        "feature_manipulation_kwargs": {},
        "pid_kwargs": {"method": "dummy_method"},
        "report_kwargs": {},
    }


def test_run_otc_experiment_uses_full_target_without_voxel_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the OTC runner uses a 2D full target and explicit layer indexes.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for dummy registry names.

    Output:
        None. Assertions validate the returned pipeline context.
    """

    register_dummy_functions(monkeypatch)

    result = otc_experiment.run_otc_experiment(base_config())

    assert result["target"] == [[11.0, 10.0], [12.0, 20.0]]
    assert result["selected_layers"] == {"X1": 1, "X2": 2}
    assert result["pid_results"] == {
        "method": "dummy_method",
        "target_sum": 53.0,
        "x1_sum": 405.0,
        "x2_sum": 809.0,
    }


def test_run_otc_experiment_accepts_ignored_voxel_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check OTC config can contain voxel_index without selecting one voxel.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for dummy registry names.

    Output:
        None. Assertions validate full-target behavior.
    """

    register_dummy_functions(monkeypatch)
    config = base_config()
    config["target_kwargs"]["voxel_index"] = 7

    result = otc_experiment.run_otc_experiment(config)

    assert result["target"] == [[11.0, 10.0], [12.0, 20.0]]


def test_run_otc_experiment_can_choose_overall_best_layers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Check OTC overall-best layer lookup uses model names from source contexts.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for dummy registry names.
        tmp_path: Path, temporary directory for a tiny overall-best CSV file.

    Output:
        None. Assertions validate config-driven overall best-layer selection.
    """

    register_dummy_functions(monkeypatch)
    csv_path = tmp_path / "overall_best.csv"
    csv_path.write_text(
        "model_name,best_layer_name,best_layer_index,mean_cv_corr,test_corr\n"
        "M1,b,3,0.1,0.2\n"
        "M2,h,4,0.3,0.4\n",
        encoding="utf-8",
    )
    config = base_config()
    config["functions"]["choose_layer"] = "overall_best_layer"
    config["choose_layer_kwargs"] = {"path_to_results": str(csv_path)}

    result = otc_experiment.run_otc_experiment(config)

    assert result["selected_layers"] == {"X1": 3, "X2": 4}


def test_run_otc_experiment_can_choose_random_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check OTC random layer selection picks valid indexes for both sources.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for dummy registry names.

    Output:
        None. Assertions validate random layer selection bounds.
    """

    register_dummy_functions(monkeypatch)
    config = base_config()
    config["functions"]["choose_layer"] = "random_layer_selection"
    config["choose_layer_kwargs"] = {"random_seed": 10}

    result = otc_experiment.run_otc_experiment(config)

    assert 0 <= result["selected_layers"]["X1"] < 4
    assert 0 <= result["selected_layers"]["X2"] < 5


def test_run_otc_experiment_rejects_voxel_best_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check full-OTC runner rejects voxel-specific best-layer selection.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for dummy registry names.

    Output:
        None. Assertions validate the ValueError.
    """

    register_dummy_functions(monkeypatch)
    config = base_config()
    config["functions"]["choose_layer"] = "voxel_best_layer"

    with pytest.raises(ValueError, match="voxel_best_layer"):
        otc_experiment.run_otc_experiment(config)
