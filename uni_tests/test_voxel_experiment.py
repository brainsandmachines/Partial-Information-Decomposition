"""Tests for the config-driven voxel experiment assembler."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import pipeline.voxel_experiments.voxel_experiment as voxel_experiment


def dummy_target(voxel_index: int, target_offset: float = 0.0) -> dict[str, Any]:
    """Create a tiny target context for voxel runner tests.

    Inputs:
        voxel_index: int, selected fake voxel index.
        target_offset: float, value added to every target sample.

    Output:
        target_context: dict, fake context containing "target".
    """

    return {
        "target": [[1.0 + target_offset], [2.0 + target_offset]],
        "voxel_index": voxel_index,
    }


def dummy_sources(model_name_1: str, model_name_2: str) -> dict[str, dict[str, Any]]:
    """Create tiny source contexts for voxel runner tests.

    Inputs:
        model_name_1: str, first fake model name.
        model_name_2: str, second fake model name.

    Output:
        sources: dict, fake source contexts under "X1" and "X2".
    """

    return {
        "X1": {"model_name": model_name_1, "base": 100.0},
        "X2": {"model_name": model_name_2, "base": 200.0},
    }


def dummy_choose_layer(
    sources: dict[str, dict[str, Any]],
    X1_index: int,
    X2_index: int,
) -> dict[str, int]:
    """Choose fake layer indexes for both sources.

    Inputs:
        sources: dict, fake source contexts under "X1" and "X2".
        X1_index: int, selected layer index for X1.
        X2_index: int, selected layer index for X2.

    Output:
        selected_layers: dict, selected indexes under "X1" and "X2".
    """

    del sources
    return {"X1": X1_index, "X2": X2_index}


def dummy_feature_extraction(
    source_context: dict[str, Any],
    layer_index: int,
    target_context: dict[str, Any],
    feature_scale: float = 1.0,
) -> list[list[float]]:
    """Create fake features from source, layer, and voxel values.

    Inputs:
        source_context: dict, fake source context.
        layer_index: int, selected fake layer index.
        target_context: dict, fake target context containing voxel_index.
        feature_scale: float, scalar multiplier applied to fake features.

    Output:
        features: list[list[float]], fake feature matrix.
    """

    start = (source_context["base"] + layer_index + target_context["voxel_index"]) * feature_scale
    return [[start], [start + 1.0]]


def dummy_feature_manipulation(
    source_1: list[list[float]],
    source_2: list[list[float]],
    add_value: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Add a value to both fake source matrices.

    Inputs:
        source_1: list[list[float]], first source features.
        source_2: list[list[float]], second source features.
        add_value: float, value added to every source entry.

    Output:
        sources: tuple, manipulated source_1 and source_2.
    """

    return (
        [[value + add_value for value in row] for row in source_1],
        [[value + add_value for value in row] for row in source_2],
    )


def dummy_pid(
    target: list[list[float]],
    source_1: list[list[float]],
    source_2: list[list[float]],
    method: str,
) -> dict[str, Any]:
    """Return simple sums from final target and source arrays.

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
        "target_sum": sum(row[0] for row in target),
        "x1_sum": sum(row[0] for row in source_1),
        "x2_sum": sum(row[0] for row in source_2),
    }


def dummy_report(pid_results: dict[str, Any], context: dict[str, Any], label: str) -> str:
    """Create a compact fake report string.

    Inputs:
        pid_results: dict, fake PID result.
        context: dict, full pipeline context.
        label: str, report label.

    Output:
        report: str, fake report output.
    """

    return f"{label}:{pid_results['method']}:{context['selected_layers']['X1']}"


def register_dummy_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register dummy wrapper names in the voxel experiment registry.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for test isolation.

    Output:
        None. The voxel experiment function registry is patched in place.
    """

    monkeypatch.setitem(voxel_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_target", dummy_target)
    monkeypatch.setitem(voxel_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_sources", dummy_sources)
    monkeypatch.setitem(voxel_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_layer", dummy_choose_layer)
    monkeypatch.setitem(voxel_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_feature", dummy_feature_extraction)
    monkeypatch.setitem(voxel_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_manip", dummy_feature_manipulation)
    monkeypatch.setitem(voxel_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_pid", dummy_pid)
    monkeypatch.setitem(voxel_experiment.PIPELINE_STEP_FUNCTIONS, "dummy_report", dummy_report)


def base_config() -> dict[str, Any]:
    """Create a valid fake voxel experiment config.

    Inputs:
        No inputs.

    Output:
        config: dict, config using dummy pipeline step names.
    """

    return {
        "functions": {
            "target_extraction": "dummy_target",
            "sources_extraction": "dummy_sources",
            "choose_layer": "dummy_layer",
            "feature_extraction": "dummy_feature",
            "preprocess": None,
            "feature_manipulation": "dummy_manip",
            "pid_calculation": "dummy_pid",
            "pid_report": "dummy_report",
        },
        "target_kwargs": {"voxel_index": 5, "target_offset": 10.0},
        "sources_kwargs": {"model_name_1": "M1", "model_name_2": "M2"},
        "choose_layer_kwargs": {"X1_index": 1, "X2_index": 2},
        "feature_extraction_kwargs": {"feature_scale": 2.0},
        "feature_manipulation_kwargs": {"add_value": 3.0},
        "pid_kwargs": {"method": "dummy_method"},
        "report_kwargs": {"label": "ok"},
    }


def test_run_voxel_experiment_resolves_functions_and_runs_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that config names resolve and kwargs reach PIDPipeline.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for dummy registry names.

    Output:
        None. Assertions validate the returned pipeline context.
    """

    register_dummy_functions(monkeypatch)

    result = voxel_experiment.run_voxel_experiment(base_config())

    assert result["target"] == [[11.0], [12.0]]
    assert result["selected_layers"] == {"X1": 1, "X2": 2}
    assert result["raw_features"]["X1"] == [[212.0], [213.0]]
    assert result["raw_features"]["X2"] == [[414.0], [415.0]]
    assert result["source_1"] == [[215.0], [216.0]]
    assert result["source_2"] == [[417.0], [418.0]]
    assert result["pid_results"] == {
        "method": "dummy_method",
        "target_sum": 23.0,
        "x1_sum": 431.0,
        "x2_sum": 835.0,
    }
    assert result["report_output"] == "ok:dummy_method:1"


def test_optional_steps_can_be_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that optional manipulation and report functions can be skipped.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for dummy registry names.

    Output:
        None. Assertions validate skipped optional-step behavior.
    """

    register_dummy_functions(monkeypatch)
    config = base_config()
    config["functions"]["feature_manipulation"] = None
    config["functions"]["pid_report"] = None

    result = voxel_experiment.run_voxel_experiment(config)

    assert result["source_1"] == [[212.0], [213.0]]
    assert result["source_2"] == [[414.0], [415.0]]
    assert result["report_output"] is None


def test_run_voxel_experiment_rejects_missing_voxel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that voxel_index=None is invalid for the voxel runner.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for dummy registry names.

    Output:
        None. Assertions validate the ValueError.
    """

    register_dummy_functions(monkeypatch)
    config = base_config()
    config["target_kwargs"]["voxel_index"] = None

    with pytest.raises(ValueError, match="voxel_index"):
        voxel_experiment.run_voxel_experiment(config)


def test_run_voxel_experiment_rejects_unknown_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that unknown configured function names fail clearly.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for dummy registry names.

    Output:
        None. Assertions validate the ValueError.
    """

    register_dummy_functions(monkeypatch)
    config = base_config()
    config["functions"]["pid_calculation"] = "not_registered"

    with pytest.raises(ValueError, match="Unknown function name"):
        voxel_experiment.run_voxel_experiment(config)


def test_pid_calc_adapter_uses_lazy_import_and_returns_pid_mi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check pid_calc_adapter output using a fake PID module.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, patch helper for sys.modules.

    Output:
        None. Assertions validate lazy PID adapter behavior.
    """

    fake_module = types.ModuleType("Partial_Information_Decomposition.PID_calc")
    seen: dict[str, Any] = {}

    def fake_pid_calc(**kwargs: Any) -> tuple[dict[str, float], dict[str, float]]:
        """Fake pid_calc implementation for adapter tests.

        Inputs:
            kwargs: any, keyword arguments received by pid_calc_adapter.

        Output:
            result: tuple, fake PID and MI dictionaries.
        """

        seen.update(kwargs)
        return {"red": 1.0}, {"tri_mi": 2.0}

    fake_module.pid_calc = fake_pid_calc
    monkeypatch.setitem(sys.modules, "Partial_Information_Decomposition.PID_calc", fake_module)

    result = voxel_experiment.pid_calc_adapter(
        target=[1.0, 2.0],
        source_1=[[3.0], [4.0]],
        source_2=[[5.0], [6.0]],
        method="fake_method",
        config={"bias_correction": True},
        rng_seed=7,
    )

    assert result == {"pid": {"red": 1.0}, "mi": {"tri_mi": 2.0}, "method": "fake_method"}
    assert seen["method"] == "fake_method"
    assert seen["config"]["dt"] == 1
    assert seen["config"]["dx1"] == 1
    assert seen["config"]["dx2"] == 1
    assert seen["config"]["n_samples"] == 2
    assert seen["config"]["bias_correction"] is True


def test_import_does_not_load_pid_calc() -> None:
    """Check voxel_experiment does not import pid_calc at module import time.

    Inputs:
        No inputs.

    Output:
        None. Assertions validate lazy PID import behavior.
    """

    sys.modules.pop("Partial_Information_Decomposition.PID_calc", None)
    importlib.reload(voxel_experiment)
    assert "Partial_Information_Decomposition.PID_calc" not in sys.modules
