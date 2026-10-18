"""Focused tests for the strict PIDPipeline orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pipeline.pid_pipeline import PIDPipeline, PIDPipelineFunctions


def test_pipeline_requires_core_functions() -> None:
    """Check that the pipeline refuses to run without required functions.

    Inputs:
        No inputs.

    Output:
        None. Assertions validate constructor errors.
    """

    with pytest.raises(ValueError, match="target_extraction"):
        PIDPipeline(PIDPipelineFunctions())


def test_pipeline_runs_strict_flow_and_returns_context() -> None:
    """Check strict flow order, per-step kwargs, and returned context fields.

    Inputs:
        No inputs.

    Output:
        None. Assertions validate the orchestrated pipeline run.
    """

    calls: list[str] = []

    def target_extraction(subject: str) -> dict[str, Any]:
        """Create target context for the test run.

        Inputs:
            subject: str, subject label supplied through target_kwargs.

        Output:
            target_context: dict, target context containing "target".
        """

        calls.append(f"target:{subject}")
        return {"target": [[1.0], [2.0]], "subject": subject}

    def sources_extraction(model_1: str, model_2: str) -> dict[str, dict[str, Any]]:
        """Create source contexts for the test run.

        Inputs:
            model_1: str, first source model name.
            model_2: str, second source model name.

        Output:
            sources: dict, source contexts under "X1" and "X2".
        """

        calls.append(f"sources:{model_1}:{model_2}")
        return {
            "X1": {
                "name": model_1,
                "layers": ["early", "late"],
                "features": {"late": [[10.0, 100.0], [20.0, 200.0]]},
            },
            "X2": {
                "name": model_2,
                "layers": ["early", "late"],
                "features": {"late": [[1.0, 10.0], [2.0, 20.0]]},
            },
        }

    def choose_layer(sources: dict[str, dict[str, Any]], layer_index: int) -> dict[str, str]:
        """Choose one layer for each source.

        Inputs:
            sources: dict, source contexts under "X1" and "X2".
            layer_index: int, selected layer index.

        Output:
            selected_layers: dict, selected layers under "X1" and "X2".
        """

        calls.append(f"choose:{layer_index}")
        return {
            "X1": sources["X1"]["layers"][layer_index],
            "X2": sources["X2"]["layers"][layer_index],
        }

    def feature_extraction(
        source_context: dict[str, Any],
        layer_name: str,
        target_context: dict[str, Any],
        add_value: float,
    ) -> list[list[float]]:
        """Extract features for one source and one layer.

        Inputs:
            source_context: dict, one source context.
            layer_name: str, selected layer name.
            target_context: dict, target context containing "subject".
            add_value: float, value added to every extracted entry.

        Output:
            features: list[list[float]], extracted feature matrix.
        """

        calls.append(f"feature:{source_context['name']}:{layer_name}:{target_context['subject']}")
        return [[value + add_value for value in row] for row in source_context["features"][layer_name]]

    def preprocess(
        target: list[list[float]],
        source_1: list[list[float]],
        source_2: list[list[float]],
        scale: float,
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        """Preprocess target and sources together.

        Inputs:
            target: list[list[float]], target samples.
            source_1: list[list[float]], first source samples.
            source_2: list[list[float]], second source samples.
            scale: float, scalar multiplier for all arrays.

        Output:
            processed: tuple, scaled target, source_1, and source_2.
        """

        calls.append(f"preprocess:{scale}")
        return (
            [[value * scale for value in row] for row in target],
            [[value * scale for value in row] for row in source_1],
            [[value * scale for value in row] for row in source_2],
        )

    def feature_manipulation(
        source_1: list[list[float]],
        source_2: list[list[float]],
        keep_columns: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Reduce both source matrices to the requested number of columns.

        Inputs:
            source_1: list[list[float]], first source samples.
            source_2: list[list[float]], second source samples.
            keep_columns: int, number of columns to retain.

        Output:
            reduced_sources: tuple, reduced source_1 and source_2.
        """

        calls.append(f"manipulate:{keep_columns}")
        return (
            [row[:keep_columns] for row in source_1],
            [row[:keep_columns] for row in source_2],
        )

    def pid_calculation(
        target: list[list[float]],
        source_1: list[list[float]],
        source_2: list[list[float]],
        method_name: str,
    ) -> dict[str, Any]:
        """Compute a tiny deterministic PID-like result.

        Inputs:
            target: list[list[float]], final target samples.
            source_1: list[list[float]], final first source samples.
            source_2: list[list[float]], final second source samples.
            method_name: str, method label supplied through pid_kwargs.

        Output:
            pid_results: dict, dummy PID-like result.
        """

        calls.append(f"pid:{method_name}")
        return {
            "method": method_name,
            "target_total": sum(row[0] for row in target),
            "x1_total": sum(row[0] for row in source_1),
            "x2_total": sum(row[0] for row in source_2),
        }

    def pid_report(pid_results: dict[str, Any], context: dict[str, Any], prefix: str) -> str:
        """Report the PID result for the test run.

        Inputs:
            pid_results: dict, PID calculation output.
            context: dict, full pipeline context.
            prefix: str, report prefix supplied through report_kwargs.

        Output:
            report: str, compact report string.
        """

        calls.append(f"report:{prefix}:{context['selected_layers']['X1']}")
        return f"{prefix}:{pid_results['method']}"

    functions = PIDPipelineFunctions(
        target_extraction=target_extraction,
        sources_extraction=sources_extraction,
        choose_layer=choose_layer,
        feature_extraction=feature_extraction,
        preprocess=preprocess,
        feature_manipulation=feature_manipulation,
        pid_calculation=pid_calculation,
        pid_report=pid_report,
    )
    pipeline = PIDPipeline(functions)

    result = pipeline.run(
        target_kwargs={"subject": "S1"},
        sources_kwargs={"model_1": "M1", "model_2": "M2"},
        choose_layer_kwargs={"layer_index": 1},
        feature_extraction_kwargs={"add_value": 1.0},
        preprocess_kwargs={"scale": 2.0},
        feature_manipulation_kwargs={"keep_columns": 1},
        pid_kwargs={"method_name": "dummy_pid"},
        report_kwargs={"prefix": "ok"},
    )

    assert calls == [
        "target:S1",
        "sources:M1:M2",
        "choose:1",
        "feature:M1:late:S1",
        "feature:M2:late:S1",
        "preprocess:2.0",
        "manipulate:1",
        "pid:dummy_pid",
        "report:ok:late",
    ]
    assert result["selected_layers"] == {"X1": "late", "X2": "late"}
    assert result["raw_features"]["X1"] == [[11.0, 101.0], [21.0, 201.0]]
    assert result["raw_features"]["X2"] == [[2.0, 11.0], [3.0, 21.0]]
    assert result["target"] == [[2.0], [4.0]]
    assert result["source_1"] == [[22.0], [42.0]]
    assert result["source_2"] == [[4.0], [6.0]]
    assert result["pid_results"] == {
        "method": "dummy_pid",
        "target_total": 6.0,
        "x1_total": 64.0,
        "x2_total": 10.0,
    }
    assert result["report_output"] == "ok:dummy_pid"


def test_pipeline_skips_optional_steps_when_not_supplied() -> None:
    """Check that optional preprocessing, manipulation, and report steps pass through.

    Inputs:
        No inputs.

    Output:
        None. Assertions validate optional-step behavior.
    """

    def target_extraction() -> dict[str, Any]:
        """Create a minimal target context.

        Inputs:
            No inputs.

        Output:
            target_context: dict, target context containing "target".
        """

        return {"target": [[1.0]]}

    def sources_extraction() -> dict[str, dict[str, Any]]:
        """Create minimal source contexts.

        Inputs:
            No inputs.

        Output:
            sources: dict, source contexts under "X1" and "X2".
        """

        return {"X1": {"features": [[2.0]]}, "X2": {"features": [[3.0]]}}

    def choose_layer(sources: dict[str, dict[str, Any]]) -> dict[str, str]:
        """Return placeholder layer names for both sources.

        Inputs:
            sources: dict, source contexts under "X1" and "X2".

        Output:
            selected_layers: dict, placeholder layers under "X1" and "X2".
        """

        del sources
        return {"X1": "only", "X2": "only"}

    def feature_extraction(
        source_context: dict[str, Any],
        layer_name: str,
        target_context: dict[str, Any],
    ) -> list[list[float]]:
        """Return the source features without using the layer placeholder.

        Inputs:
            source_context: dict, one source context.
            layer_name: str, placeholder selected layer.
            target_context: dict, target context.

        Output:
            features: list[list[float]], source features.
        """

        del layer_name, target_context
        return source_context["features"]

    def pid_calculation(
        target: list[list[float]],
        source_1: list[list[float]],
        source_2: list[list[float]],
    ) -> dict[str, float]:
        """Return sums from the final arrays.

        Inputs:
            target: list[list[float]], final target samples.
            source_1: list[list[float]], final first source samples.
            source_2: list[list[float]], final second source samples.

        Output:
            pid_results: dict, simple sums by array.
        """

        return {"target": target[0][0], "x1": source_1[0][0], "x2": source_2[0][0]}

    pipeline = PIDPipeline(
        PIDPipelineFunctions(
            target_extraction=target_extraction,
            sources_extraction=sources_extraction,
            choose_layer=choose_layer,
            feature_extraction=feature_extraction,
            pid_calculation=pid_calculation,
        )
    )

    result = pipeline.run()

    assert result["target"] == [[1.0]]
    assert result["source_1"] == [[2.0]]
    assert result["source_2"] == [[3.0]]
    assert result["pid_results"] == {"target": 1.0, "x1": 2.0, "x2": 3.0}
    assert result["report_output"] is None


def test_pipeline_validates_step_contracts() -> None:
    """Check that strict function contracts fail with readable errors.

    Inputs:
        No inputs.

    Output:
        None. Assertions validate contract errors.
    """

    def bad_target_extraction() -> dict[str, Any]:
        """Return an invalid target context.

        Inputs:
            No inputs.

        Output:
            target_context: dict, intentionally missing "target".
        """

        return {"neural_data": [[1.0]]}

    def sources_extraction() -> dict[str, dict[str, Any]]:
        """Create minimal source contexts.

        Inputs:
            No inputs.

        Output:
            sources: dict, source contexts under "X1" and "X2".
        """

        return {"X1": {}, "X2": {}}

    def choose_layer(sources: dict[str, dict[str, Any]]) -> dict[str, str]:
        """Choose placeholder layers.

        Inputs:
            sources: dict, source contexts under "X1" and "X2".

        Output:
            selected_layers: dict, placeholder layers under "X1" and "X2".
        """

        del sources
        return {"X1": "only", "X2": "only"}

    def feature_extraction(
        source_context: dict[str, Any],
        layer_name: str,
        target_context: dict[str, Any],
    ) -> list[list[float]]:
        """Return a placeholder source feature matrix.

        Inputs:
            source_context: dict, one source context.
            layer_name: str, selected layer name.
            target_context: dict, target context.

        Output:
            features: list[list[float]], placeholder source features.
        """

        del source_context, layer_name, target_context
        return [[1.0]]

    def pid_calculation(
        target: list[list[float]],
        source_1: list[list[float]],
        source_2: list[list[float]],
    ) -> dict[str, float]:
        """Return an empty placeholder PID result.

        Inputs:
            target: list[list[float]], final target samples.
            source_1: list[list[float]], final first source samples.
            source_2: list[list[float]], final second source samples.

        Output:
            pid_results: dict, empty placeholder result.
        """

        del target, source_1, source_2
        return {}

    pipeline = PIDPipeline(
        PIDPipelineFunctions(
            target_extraction=bad_target_extraction,
            sources_extraction=sources_extraction,
            choose_layer=choose_layer,
            feature_extraction=feature_extraction,
            pid_calculation=pid_calculation,
        )
    )

    with pytest.raises(ValueError, match="target"):
        pipeline.run()
