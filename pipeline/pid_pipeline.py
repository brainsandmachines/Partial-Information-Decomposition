"""Strict, readable orchestration class for one PID pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PIDPipelineFunctions:
    """Store the user-selected functions for one PID pipeline run.

    Inputs:
        target_extraction: callable, creates the target context and must return a
            dict containing "target".
        sources_extraction: callable, creates source contexts and must return a
            dict containing "X1" and "X2".
        choose_layer: callable, chooses source layers and must return a dict
            containing "X1" and "X2".
        feature_extraction: callable, extracts features for one source and one layer.
        pid_calculation: callable, computes PID from target, source_1, and source_2.
        preprocess: callable or None, optional grouped preprocessing for target,
            source_1, and source_2.
        feature_manipulation: callable or None, optional grouped manipulation for
            source_1 and source_2.
        pid_report: callable or None, optional reporter for PID results and context.

    Output:
        PIDPipelineFunctions instance containing the selected pipeline functions.
    """

    target_extraction: Callable[..., dict[str, Any]] | None = None
    sources_extraction: Callable[..., dict[str, Any]] | None = None
    choose_layer: Callable[..., dict[str, Any]] | None = None
    feature_extraction: Callable[..., Any] | None = None
    pid_calculation: Callable[..., Any] | None = None
    preprocess: Callable[..., tuple[Any, Any, Any]] | None = None
    feature_manipulation: Callable[..., tuple[Any, Any]] | None = None
    pid_report: Callable[..., Any] | None = None


class PIDPipeline:
    """Run the PID pipeline by connecting the provided functions in order."""

    def __init__(self, functions: PIDPipelineFunctions) -> None:
        """Create a strict PID pipeline orchestrator.

        Inputs:
            functions: PIDPipelineFunctions, user-selected functions for target
                extraction, sources extraction, layer choice, feature extraction,
                optional preprocessing, optional feature manipulation, PID
                calculation, and optional reporting.

        Output:
            None. The PIDPipeline instance stores the selected functions and
            validates that required functions were provided.
        """

        required_function_names = (
            "target_extraction",
            "sources_extraction",
            "choose_layer",
            "feature_extraction",
            "pid_calculation",
        )
        missing_function_names = [
            name
            for name in required_function_names
            if not callable(getattr(functions, name))
        ]
        if missing_function_names:
            raise ValueError(f"Missing required pipeline functions: {missing_function_names}")

        optional_function_names = ("preprocess", "feature_manipulation", "pid_report")
        invalid_optional_names = [
            name
            for name in optional_function_names
            if getattr(functions, name) is not None and not callable(getattr(functions, name))
        ]
        if invalid_optional_names:
            raise ValueError(f"Optional pipeline functions must be callable or None: {invalid_optional_names}")

        self.functions = functions

    def run(
        self,
        *,
        target_kwargs: dict[str, Any] | None = None,
        sources_kwargs: dict[str, Any] | None = None,
        choose_layer_kwargs: dict[str, Any] | None = None,
        feature_extraction_kwargs: dict[str, Any] | None = None,
        preprocess_kwargs: dict[str, Any] | None = None,
        feature_manipulation_kwargs: dict[str, Any] | None = None,
        pid_kwargs: dict[str, Any] | None = None,
        report_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run target, sources, layers, features, optional transforms, PID, and report.

        Inputs:
            target_kwargs: dict or None, keyword arguments for target_extraction.
            sources_kwargs: dict or None, keyword arguments for sources_extraction.
            choose_layer_kwargs: dict or None, keyword arguments for choose_layer.
            feature_extraction_kwargs: dict or None, keyword arguments for feature_extraction.
            preprocess_kwargs: dict or None, keyword arguments for preprocess.
            feature_manipulation_kwargs: dict or None, keyword arguments for feature_manipulation.
            pid_kwargs: dict or None, keyword arguments for pid_calculation.
            report_kwargs: dict or None, keyword arguments for pid_report.

        Output:
            context: dict, full run context containing target_context, sources,
            selected_layers, raw_features, target, source_1, source_2, pid_results,
            and report_output.
        """

        target_context = self.functions.target_extraction(**(target_kwargs or {}))
        if not isinstance(target_context, dict) or "target" not in target_context:
            raise ValueError("target_extraction must return a dict containing 'target'.")

        sources = self.functions.sources_extraction(**(sources_kwargs or {}))
        if not isinstance(sources, dict) or "X1" not in sources or "X2" not in sources:
            raise ValueError("sources_extraction must return a dict containing 'X1' and 'X2'.")

        selected_layers = self.functions.choose_layer(sources, **(choose_layer_kwargs or {}))
        if not isinstance(selected_layers, dict) or "X1" not in selected_layers or "X2" not in selected_layers:
            raise ValueError("choose_layer must return a dict containing 'X1' and 'X2'.")

        source_1_raw = self.functions.feature_extraction(
            sources["X1"],
            selected_layers["X1"],
            target_context,
            **(feature_extraction_kwargs or {}),
        )
        source_2_raw = self.functions.feature_extraction(
            sources["X2"],
            selected_layers["X2"],
            target_context,
            **(feature_extraction_kwargs or {}),
        )

        target = target_context["target"]
        source_1 = source_1_raw
        source_2 = source_2_raw

        if self.functions.preprocess is not None:
            target, source_1, source_2 = self.functions.preprocess(
                target,
                source_1,
                source_2,
                **(preprocess_kwargs or {}),
            )

        if self.functions.feature_manipulation is not None:
            source_1, source_2 = self.functions.feature_manipulation(
                source_1,
                source_2,
                **(feature_manipulation_kwargs or {}),
            )

        pid_results = self.functions.pid_calculation(
            target,
            source_1,
            source_2,
            **(pid_kwargs or {}),
        )

        context = {
            "target_context": target_context,
            "sources": sources,
            "selected_layers": selected_layers,
            "raw_features": {"X1": source_1_raw, "X2": source_2_raw},
            "target": target,
            "source_1": source_1,
            "source_2": source_2,
            "pid_results": pid_results,
            "report_output": None,
        }

        if self.functions.pid_report is not None:
            context["report_output"] = self.functions.pid_report(
                pid_results,
                context,
                **(report_kwargs or {}),
            )

        return context
