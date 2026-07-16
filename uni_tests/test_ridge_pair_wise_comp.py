"""Focused tests for reusable ridge helpers and the efficient pairwise runner."""

from __future__ import annotations

import sys
import threading
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from pipeline.analysis.pca_analysis.all_models_pairwise import ridge_pair_wise_comp
from pipeline.analysis.pca_analysis.all_models_pairwise import ridge_pairwise_utils
from pipeline.ridge_find_alpha import find_alpha
from pipeline.pipeline_phases import feature_manipulations

_missmda_stub = ModuleType("library_wrappers.missmda_ncp")
_missmda_stub.estimate_ncp_pca = Mock(  # type: ignore[attr-defined]
    side_effect=RuntimeError("missMDA must be mocked in unit tests.")
)
sys.modules.setdefault("library_wrappers.missmda_ncp", _missmda_stub)

from pipeline.subj_PCs import subj_pc_analysis


def _write_alpha_archive(
    path: Path,
    *,
    alphas: np.ndarray,
    pc_indices: np.ndarray,
    model_name: str | np.ndarray,
    layer_index: int | float | np.ndarray,
    omit_fields: frozenset[str] = frozenset(),
) -> Path:
    """Write a deterministic ridge-alpha archive for a helper test.

    Inputs:
        path: Path where the ``.npz`` archive will be written.
        alphas: np.ndarray containing per-target-component ridge penalties.
        pc_indices: np.ndarray containing the archived target-PC ordering.
        model_name: str or np.ndarray containing archived model metadata.
        layer_index: int, float, or np.ndarray containing layer metadata.
        omit_fields: frozenset[str] naming fields to leave out deliberately.

    Outputs:
        Path pointing to the written archive.
    """

    fields: dict[str, object] = {
        "alphas": alphas,
        "pc_indices": pc_indices,
        "model_name": model_name,
        "layer_index": layer_index,
    }
    np.savez(path, **{name: value for name, value in fields.items() if name not in omit_fields})
    return path


def test_target_pca_centers_raw_data_without_variance_standardization() -> None:
    """Check target PCA uses raw feature scales while retaining mean-centering.

    Inputs:
        None.

    Outputs:
        None. Assertions compare the helper with direct scikit-learn PCA on the
        unstandardized input and verify that no scaler artifact is returned.
    """

    target = np.array(
        [
            [1.0, 100.0, 0.01],
            [2.0, 250.0, 0.04],
            [4.0, 450.0, 0.02],
            [8.0, 900.0, 0.08],
        ]
    )
    expected_pca = PCA(n_components=None, svd_solver="full")
    expected_scores = expected_pca.fit_transform(target)

    result = subj_pc_analysis.pca_by_variance(target, variance_threshold=1.0)

    np.testing.assert_allclose(result["pca"].mean_, target.mean(axis=0))
    np.testing.assert_allclose(result["transformed_data"], expected_scores)
    assert "scaler" not in result


def test_missmda_selection_explicitly_disables_variance_standardization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check missMDA selects components with ``scale=False`` before raw PCA.

    Inputs:
        monkeypatch: pytest.MonkeyPatch replacing the external selector with a
            deterministic call recorder.

    Outputs:
        None. Assertions cover the selector flag and the fitted PCA mean.
    """

    target = np.array(
        [
            [1.0, 100.0, 0.01],
            [2.0, 250.0, 0.04],
            [4.0, 450.0, 0.02],
            [8.0, 900.0, 0.08],
        ]
    )
    selector = Mock(return_value={"ncp": 2, "criterion": {}})
    monkeypatch.setattr(subj_pc_analysis, "estimate_ncp_pca", selector)

    result = subj_pc_analysis.pca_func(
        target,
        mode="missmda_CV",
        max_features=2,
    )

    assert selector.call_args.kwargs["scale"] is False
    np.testing.assert_allclose(result["pca"].mean_, target.mean(axis=0))
    assert "scaler" not in result


def test_alpha_search_and_saved_pca_use_raw_unstandardized_arrays(
    tmp_path: Path,
) -> None:
    """Check alpha search has no scaler step and saved PCA transforms raw data.

    Inputs:
        tmp_path: Path provided by pytest for the centered PCA artifact.

    Outputs:
        None. Assertions cover the raw-input ridge pipeline and direct PCA
        transformation without an intermediate variance scaler.
    """

    predictor = np.array(
        [
            [1.0, 10.0, 0.1],
            [2.0, 30.0, 0.2],
            [3.0, 20.0, 0.4],
            [5.0, 80.0, 0.8],
            [8.0, 50.0, 1.6],
            [13.0, 130.0, 3.2],
        ]
    )
    target = np.column_stack(
        (predictor[:, 0] + predictor[:, 2], predictor[:, 1] - predictor[:, 0])
    )

    alphas, ridge_pipeline = find_alpha.find_alpha_per_pc(predictor, target)

    assert alphas.shape == (target.shape[1],)
    assert list(ridge_pipeline.named_steps) == ["ridgecv"]
    assert ridge_pipeline.predict(predictor).shape == target.shape

    target_pca = PCA(n_components=2, svd_solver="full").fit(target)
    target_pca_path = tmp_path / "target_pca.pkl"
    joblib.dump(target_pca, target_pca_path)
    transformed = find_alpha.load_and_apply_pca(target, target_pca_path)
    np.testing.assert_allclose(transformed, target_pca.transform(target))


def test_prepare_ridge_target_projects_once_and_preserves_held_out_order(
    tmp_path: Path,
) -> None:
    """Check saved target PCA projection and exact Boolean-mask splitting.

    Inputs:
        tmp_path: Path provided by pytest for the target PCA artifact.

    Outputs:
        None. Assertions validate train/test values, ordering, and mask copying.
    """

    target = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 5.0],
            [3.0, 4.0, 2.0],
            [5.0, 2.0, 1.0],
            [8.0, 3.0, 4.0],
            [13.0, 5.0, 7.0],
        ]
    )
    mask = np.array([False, True, False, True, False, False])
    pca = PCA(n_components=2, svd_solver="full").fit(target)
    pca_path = tmp_path / "target_pca.pkl"
    joblib.dump(pca, pca_path)
    target_context = {"shared1000_subj": mask.copy()}

    train_target, test_target, returned_mask = feature_manipulations.prepare_ridge_target(
        target,
        target_context,
        pca_path,
    )

    projected = pca.transform(target)
    np.testing.assert_allclose(train_target, projected[~mask])
    np.testing.assert_allclose(test_target, projected[mask])
    np.testing.assert_array_equal(returned_mask, mask)
    target_context["shared1000_subj"][:] = False
    np.testing.assert_array_equal(returned_mask, mask)


def test_prepare_ridge_target_rejects_invalid_target_mask_and_artifact(
    tmp_path: Path,
) -> None:
    """Check target, held-out-mask, and PCA artifact validation.

    Inputs:
        tmp_path: Path provided by pytest for target transform artifacts.

    Outputs:
        None. Assertions validate failures before ridge fitting can begin.
    """

    target = np.arange(18.0).reshape(6, 3)
    mask = np.array([False, False, False, False, True, True])
    pca_path = tmp_path / "target_pca.pkl"
    joblib.dump(PCA(n_components=2).fit(target), pca_path)

    with pytest.raises(ValueError, match="two-dimensional"):
        feature_manipulations.prepare_ridge_target(
            target[:, 0],
            {"shared1000_subj": mask},
            pca_path,
        )
    with pytest.raises(ValueError, match="NaN or infinite"):
        invalid_target = target.copy()
        invalid_target[0, 0] = np.nan
        feature_manipulations.prepare_ridge_target(
            invalid_target,
            {"shared1000_subj": mask},
            pca_path,
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        feature_manipulations.prepare_ridge_target(target, None, pca_path)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="shared1000_subj"):
        feature_manipulations.prepare_ridge_target(target, {}, pca_path)
    with pytest.raises(ValueError, match="one-dimensional Boolean"):
        feature_manipulations.prepare_ridge_target(
            target,
            {"shared1000_subj": mask.astype(np.int64)},
            pca_path,
        )
    with pytest.raises(ValueError, match="one entry for each target sample"):
        feature_manipulations.prepare_ridge_target(
            target,
            {"shared1000_subj": mask[:-1]},
            pca_path,
        )
    with pytest.raises(ValueError, match="at least one held-out"):
        feature_manipulations.prepare_ridge_target(
            target,
            {"shared1000_subj": np.zeros(target.shape[0], dtype=bool)},
            pca_path,
        )
    with pytest.raises(ValueError, match="at least one training"):
        feature_manipulations.prepare_ridge_target(
            target,
            {"shared1000_subj": np.ones(target.shape[0], dtype=bool)},
            pca_path,
        )

    invalid_pca_path = tmp_path / "invalid_pca.pkl"
    joblib.dump({"transform": None}, invalid_pca_path)
    with pytest.raises(TypeError, match="provide a transform method"):
        feature_manipulations.prepare_ridge_target(
            target,
            {"shared1000_subj": mask},
            invalid_pca_path,
        )


def test_load_ridge_alphas_accepts_exact_model_layer_and_pc_metadata(
    tmp_path: Path,
) -> None:
    """Check successful strict loading of a model's per-PC penalties.

    Inputs:
        tmp_path: Path provided by pytest for the alpha archive.

    Outputs:
        None. Assertions validate alpha values, dtype, and owned memory.
    """

    original_alphas = np.array([0.125, 1.5, 20.0], dtype=np.float32)
    archive_path = _write_alpha_archive(
        tmp_path / "alphas.npz",
        alphas=original_alphas,
        pc_indices=np.array([1, 2, 3]),
        model_name="org/model-name",
        layer_index=7,
    )

    loaded = feature_manipulations.load_ridge_alphas(
        archive_path,
        model_name="org/model-name",
        expected_target_dim=3,
        expected_layer_index=7,
    )

    np.testing.assert_allclose(loaded, original_alphas)
    assert loaded.dtype == np.float64
    assert loaded.flags.owndata


def test_load_ridge_alphas_rejects_invalid_values_and_metadata(
    tmp_path: Path,
) -> None:
    """Check strict alpha dimension, ordering, model, and layer validation.

    Inputs:
        tmp_path: Path provided by pytest for malformed alpha archives.

    Outputs:
        None. Assertions validate informative failures for artifact mismatch.
    """

    archive_path = tmp_path / "alphas.npz"

    _write_alpha_archive(
        archive_path,
        alphas=np.array([0.1, 0.2]),
        pc_indices=np.array([1, 2]),
        model_name="model-a",
        layer_index=3,
        omit_fields=frozenset({"layer_index"}),
    )
    with pytest.raises(ValueError, match="missing required fields: layer_index"):
        feature_manipulations.load_ridge_alphas(
            archive_path,
            model_name="model-a",
            expected_target_dim=2,
        )

    _write_alpha_archive(
        archive_path,
        alphas=np.array([0.1, 0.2]),
        pc_indices=np.array([1, 2]),
        model_name="model-a",
        layer_index=3,
    )
    with pytest.raises(ValueError, match="exactly one alpha"):
        feature_manipulations.load_ridge_alphas(
            archive_path,
            model_name="model-a",
            expected_target_dim=3,
        )
    with pytest.raises(ValueError, match="model name mismatch"):
        feature_manipulations.load_ridge_alphas(
            archive_path,
            model_name="model-b",
            expected_target_dim=2,
        )
    with pytest.raises(ValueError, match="layer index mismatch"):
        feature_manipulations.load_ridge_alphas(
            archive_path,
            model_name="model-a",
            expected_target_dim=2,
            expected_layer_index=4,
        )
    with pytest.raises(TypeError, match="expected_target_dim must be an integer"):
        feature_manipulations.load_ridge_alphas(
            archive_path,
            model_name="model-a",
            expected_target_dim=True,
        )
    with pytest.raises(TypeError, match="expected_layer_index must be an integer"):
        feature_manipulations.load_ridge_alphas(
            archive_path,
            model_name="model-a",
            expected_target_dim=2,
            expected_layer_index=True,
        )

    malformed_cases = (
        (
            np.array([0.1, -0.2]),
            np.array([1, 2]),
            "model-a",
            3,
            "non-negative",
        ),
        (
            np.array([0.1, np.inf]),
            np.array([1, 2]),
            "model-a",
            3,
            "NaN or infinite",
        ),
        (
            np.array([0.1, 0.2]),
            np.array([2, 1]),
            "model-a",
            3,
            "ordered and one-based",
        ),
        (
            np.array([0.1, 0.2]),
            np.array([1, 2]),
            np.array(["model-a", "model-b"]),
            3,
            "model_name metadata must be scalar",
        ),
        (
            np.array([0.1, 0.2]),
            np.array([1, 2]),
            "model-a",
            3.5,
            "layer_index metadata must be an integer",
        ),
    )
    for alphas, pc_indices, model_name, layer_index, message in malformed_cases:
        _write_alpha_archive(
            archive_path,
            alphas=alphas,
            pc_indices=pc_indices,
            model_name=model_name,
            layer_index=layer_index,
        )
        with pytest.raises((TypeError, ValueError), match=message):
            feature_manipulations.load_ridge_alphas(
                archive_path,
                model_name="model-a",
                expected_target_dim=2,
            )


def test_ridge_predict_shared_matches_direct_held_out_ridge() -> None:
    """Check that ridge fits non-shared rows and predicts shared rows only.

    Inputs:
        No inputs.

    Outputs:
        None. Assertions compare the helper with a direct scikit-learn fit.
    """

    rng = np.random.default_rng(29)
    source = rng.normal(size=(12, 4))
    shared_mask = np.array(
        [False, True, False, False, True, False, False, True, False, False, True, False]
    )
    weights = np.array(
        [
            [1.5, -0.5, 0.3],
            [0.2, 1.1, -0.8],
            [-1.0, 0.4, 0.6],
            [0.7, -0.2, 1.3],
        ]
    )
    full_target = source @ weights + np.array([0.5, -1.0, 2.0])
    train_target = full_target[~shared_mask]
    alphas = np.array([0.05, 0.5, 5.0])

    expected_model = Ridge(alpha=alphas, fit_intercept=True, random_state=17)
    expected_model.fit(source[~shared_mask], train_target)
    expected = expected_model.predict(source[shared_mask])
    actual = feature_manipulations.ridge_predict_shared(
        source,
        train_target,
        shared_mask,
        alphas,
        seed=17,
    )

    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (int(shared_mask.sum()), train_target.shape[1])


def test_ridge_predict_shared_preserves_single_target_component_axis() -> None:
    """Check one-component ridge predictions remain two-dimensional.

    Inputs:
        No inputs.

    Outputs:
        None. Assertions require held-out predictions with shape
        ``(n_test, 1)`` and validate their numerical values.
    """

    source = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, -1.0],
            [4.0, 2.0],
            [5.0, -2.0],
        ]
    )
    shared_mask = np.array([False, False, True, False, True, False])
    full_target = (2.0 * source[:, 0] - 0.75 * source[:, 1] + 1.25).reshape(-1, 1)
    train_target = full_target[~shared_mask]
    alphas = np.array([0.4])

    direct_model = Ridge(alpha=alphas, fit_intercept=True, random_state=31)
    direct_model.fit(source[~shared_mask], train_target)
    expected = np.asarray(direct_model.predict(source[shared_mask])).reshape(-1, 1)
    actual = feature_manipulations.ridge_predict_shared(
        source,
        train_target,
        shared_mask,
        alphas,
        seed=31,
    )

    assert actual.shape == (int(shared_mask.sum()), 1)
    np.testing.assert_allclose(actual, expected)


def test_ridge_predict_shared_rejects_invalid_arrays_and_seed() -> None:
    """Check source, target, mask, alpha, and random-seed validation.

    Inputs:
        No inputs.

    Outputs:
        None. Assertions validate failures before or after ridge prediction.
    """

    source = np.arange(18.0).reshape(6, 3)
    shared_mask = np.array([False, False, False, False, True, True])
    train_target = np.arange(8.0).reshape(4, 2)
    alphas = np.array([0.1, 1.0])

    with pytest.raises(ValueError, match="source must be a two-dimensional"):
        feature_manipulations.ridge_predict_shared(
            source[:, 0],
            train_target,
            shared_mask,
            alphas,
            seed=1,
        )
    with pytest.raises(ValueError, match="source contains NaN or infinite"):
        invalid_source = source.copy()
        invalid_source[0, 0] = np.inf
        feature_manipulations.ridge_predict_shared(
            invalid_source,
            train_target,
            shared_mask,
            alphas,
            seed=1,
        )
    with pytest.raises(ValueError, match="train_target must be a two-dimensional"):
        feature_manipulations.ridge_predict_shared(
            source,
            train_target[:, 0],
            shared_mask,
            alphas,
            seed=1,
        )
    with pytest.raises(ValueError, match="train_target contains NaN or infinite"):
        invalid_target = train_target.copy()
        invalid_target[0, 0] = np.nan
        feature_manipulations.ridge_predict_shared(
            source,
            invalid_target,
            shared_mask,
            alphas,
            seed=1,
        )
    with pytest.raises(ValueError, match="one-dimensional Boolean"):
        feature_manipulations.ridge_predict_shared(
            source,
            train_target,
            shared_mask.astype(np.int64),
            alphas,
            seed=1,
        )
    with pytest.raises(ValueError, match="one entry for each source sample"):
        feature_manipulations.ridge_predict_shared(
            source,
            train_target,
            shared_mask[:-1],
            alphas,
            seed=1,
        )
    with pytest.raises(ValueError, match="at least one held-out"):
        feature_manipulations.ridge_predict_shared(
            source,
            np.arange(12.0).reshape(6, 2),
            np.zeros(6, dtype=bool),
            alphas,
            seed=1,
        )
    with pytest.raises(ValueError, match="non-shared source rows"):
        feature_manipulations.ridge_predict_shared(
            source,
            train_target[:-1],
            shared_mask,
            alphas,
            seed=1,
        )
    with pytest.raises(ValueError, match="exactly one alpha"):
        feature_manipulations.ridge_predict_shared(
            source,
            train_target,
            shared_mask,
            alphas[:1],
            seed=1,
        )
    with pytest.raises(ValueError, match="NaN or infinite"):
        feature_manipulations.ridge_predict_shared(
            source,
            train_target,
            shared_mask,
            np.array([0.1, np.nan]),
            seed=1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        feature_manipulations.ridge_predict_shared(
            source,
            train_target,
            shared_mask,
            np.array([0.1, -1.0]),
            seed=1,
        )
    with pytest.raises(TypeError, match="seed must be an integer"):
        feature_manipulations.ridge_predict_shared(
            source,
            train_target,
            shared_mask,
            alphas,
            seed=True,
        )


def test_grouped_feature_manipulation_matches_composed_single_array_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check grouped ridge compatibility and delegation to reusable helpers.

    Inputs:
        tmp_path: Path provided by pytest for target and alpha artifacts.
        monkeypatch: pytest.MonkeyPatch used to spy on helper delegation.

    Outputs:
        None. Assertions validate identical held-out target and predictions.
    """

    rng = np.random.default_rng(41)
    target = rng.normal(size=(14, 5))
    source1 = rng.normal(size=(14, 4))
    source2 = rng.normal(size=(14, 6))
    shared_mask = np.array(
        [
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
        ]
    )
    target_context = {"shared1000_subj": shared_mask}
    pca = PCA(n_components=3, svd_solver="full").fit(target)
    pca_path = tmp_path / "target_pca.pkl"
    joblib.dump(pca, pca_path)
    alphas1_path = _write_alpha_archive(
        tmp_path / "model1_alphas.npz",
        alphas=np.array([0.05, 0.5, 5.0]),
        pc_indices=np.array([1, 2, 3]),
        model_name="model/one",
        layer_index=4,
    )
    alphas2_path = _write_alpha_archive(
        tmp_path / "model2_alphas.npz",
        alphas=np.array([0.1, 1.0, 10.0]),
        pc_indices=np.array([1, 2, 3]),
        model_name="model/two",
        layer_index=9,
    )

    train_target, expected_target, direct_mask = feature_manipulations.prepare_ridge_target(
        target,
        target_context,
        pca_path,
    )
    alphas1 = feature_manipulations.load_ridge_alphas(
        alphas1_path,
        model_name="model/one",
        expected_target_dim=3,
    )
    alphas2 = feature_manipulations.load_ridge_alphas(
        alphas2_path,
        model_name="model/two",
        expected_target_dim=3,
    )
    expected_source1 = feature_manipulations.ridge_predict_shared(
        source1,
        train_target,
        direct_mask,
        alphas1,
        seed=23,
    )
    expected_source2 = feature_manipulations.ridge_predict_shared(
        source2,
        train_target,
        direct_mask,
        alphas2,
        seed=23,
    )

    prepare_spy = Mock(wraps=feature_manipulations.prepare_ridge_target)
    load_spy = Mock(wraps=feature_manipulations.load_ridge_alphas)
    predict_spy = Mock(wraps=feature_manipulations.ridge_predict_shared)
    monkeypatch.setattr(feature_manipulations, "prepare_ridge_target", prepare_spy)
    monkeypatch.setattr(feature_manipulations, "load_ridge_alphas", load_spy)
    monkeypatch.setattr(feature_manipulations, "ridge_predict_shared", predict_spy)

    actual_source1, actual_source2, actual_target = (
        feature_manipulations.feature_manipulation_ridge(
            source1,
            source2,
            target,
            target_context,
            23,
            "model/one",
            "model/two",
            pca_path,
            alphas1_path,
            alphas2_path,
        )
    )

    assert prepare_spy.call_count == 1
    assert load_spy.call_count == 2
    assert predict_spy.call_count == 2
    np.testing.assert_allclose(actual_source1, expected_source1)
    np.testing.assert_allclose(actual_source2, expected_source2)
    np.testing.assert_allclose(actual_target, expected_target)


_REAL_RELEASE_MODEL_CONTEXT = ridge_pairwise_utils._release_model_context


def test_project_path_resolution_is_checkout_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check relative config paths follow the checkout and absolute paths do not.

    Inputs:
        tmp_path: Path used as a synthetic repository root and absolute path.
        monkeypatch: pytest.MonkeyPatch replacing the runner's repository root.

    Outputs:
        None. Assertions validate portable relative paths and unchanged
        absolute paths.
    """

    monkeypatch.setattr(ridge_pairwise_utils, "repo_root", tmp_path)
    absolute_path = tmp_path / "already-absolute.pkl"

    assert ridge_pairwise_utils._resolve_project_path(
        "pipeline/artifacts/model.pkl"
    ) == tmp_path / "pipeline/artifacts/model.pkl"
    assert ridge_pairwise_utils._resolve_project_path(absolute_path) == absolute_path


class _RunnerHarness:
    """Install deterministic fakes around the real pairwise runner control flow.

    Inputs:
        monkeypatch: pytest.MonkeyPatch used to replace external data/model/PID
            boundaries while preserving the runner and its real thread executor.
        tmp_path: Path used for empty artifact files and CSV checkpoints.
        model_names: list[str] of synthetic model identifiers available to runs.
        prefetch_enabled: bool controlling the runner's execution setting.
        coordinate_overlap: bool requesting deterministic worker/ridge overlap.
        fail_stage: Optional str naming ``worker``, ``ridge``, or ``pid`` as a
            synthetic failure boundary.

    Outputs:
        A configured harness instance whose ``run`` method invokes the real
        pairwise runner and whose counters/events expose observable behavior.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        model_names: list[str],
        *,
        prefetch_enabled: bool,
        coordinate_overlap: bool = False,
        fail_stage: str | None = None,
    ) -> None:
        """Create artifacts, state trackers, runner configuration, and patches.

        Inputs:
            monkeypatch: pytest.MonkeyPatch applying reversible test doubles.
            tmp_path: Path receiving synthetic empty artifact files.
            model_names: list[str] defining deterministic model values/order.
            prefetch_enabled: bool written to the execution configuration.
            coordinate_overlap: bool enabling synchronization of the first
                background load with the first ridge call.
            fail_stage: Optional str selecting a synthetic failure boundary.

        Outputs:
            None. The initialized object owns all test state and configuration.
        """

        if fail_stage not in {None, "worker", "ridge", "pid"}:
            raise ValueError(f"Unsupported synthetic failure stage: {fail_stage}")
        self.model_names = list(model_names)
        self.model_values = {
            model_name: float(model_index + 1)
            for model_index, model_name in enumerate(self.model_names)
        }
        self.prefetch_enabled = prefetch_enabled
        self.coordinate_overlap = coordinate_overlap
        self.fail_stage = fail_stage
        self.main_thread_ident = threading.get_ident()
        self.counts: Counter[str] = Counter()
        self.event_log: list[tuple[str, str, int]] = []
        self.pid_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.created_contexts: dict[str, list[dict[str, Any]]] = {
            model_name: [] for model_name in self.model_names
        }
        self.live_context_ids: set[int] = set()
        self.max_live_contexts = 0
        self.state_lock = threading.Lock()
        self.prefetch_started = threading.Event()
        self.pid_started = threading.Event()
        self.prefetch_terminal = threading.Event()
        self.coordinated_background_load = False
        self.first_pid_call = True
        self.worker_failure_raised = False
        self.ridge_failure_raised = False
        self.hdf_file = Mock(name="synthetic_hdf_file")
        self.shared_mask = np.array([False, False, False, False, True, True])
        self.test_target = np.array([[40.0, 41.0], [50.0, 51.0]])

        self.target_pca_path = tmp_path / "target_pca.pkl"
        self.layer_results_path = tmp_path / "best_layers.csv"
        self.target_pca_path.touch()
        self.layer_results_path.touch()
        self.artifact_dir = tmp_path / "model_artifacts"
        self.artifact_dir.mkdir()
        for model_name in self.model_names:
            safe_name = ridge_pairwise_utils.safe_model_name(model_name)
            alpha_path = self.artifact_dir / f"{safe_name}.alphas"
            alpha_path.touch()

        self.config: dict[str, Any] = {
            "functions": {
                "target_extraction": "synthetic_target",
                "feature_extraction": "synthetic_features",
                "pid_calculation": "synthetic_pid",
                "pid_report": "synthetic_report",
            },
            "metadata": {"subj_id": "synthetic-subject"},
            "execution": {"prefetch_ridge_predictions": prefetch_enabled},
            "target_kwargs": {"n_samples": 6},
            "choose_layer_kwargs": {
                "path_to_results": str(self.layer_results_path)
            },
            "feature_extraction_kwargs": {"batch_size_process": 2},
            "feature_manipulation_kwargs": {
                "pc_target_path": str(self.target_pca_path)
            },
            "artifact_templates": {
                "ridge_alphas": {
                    "directory_template": str(self.artifact_dir),
                    "filename_template": "{safe_model_name}.alphas",
                },
            },
            "pid_kwargs": {
                "method": "synthetic-method",
                "rng_seed": 73,
                "config": {"bias_correction": True},
            },
            "report_kwargs": {},
        }

        monkeypatch.setattr(
            ridge_pair_wise_comp,
            "resolve_pipeline_function",
            self.resolve_pipeline_function,
        )
        monkeypatch.setattr(
            ridge_pairwise_utils,
            "overall_best_layer",
            self.overall_best_layer,
        )
        monkeypatch.setattr(
            ridge_pairwise_utils,
            "load_ridge_alphas",
            self.load_ridge_alphas,
        )
        monkeypatch.setattr(
            ridge_pair_wise_comp,
            "prepare_ridge_target",
            self.prepare_ridge_target,
        )
        monkeypatch.setattr(
            ridge_pairwise_utils,
            "prepare_model_context",
            self.prepare_model_context,
        )
        monkeypatch.setattr(
            ridge_pairwise_utils,
            "ridge_predict_shared",
            self.ridge_predict_shared,
        )
        monkeypatch.setattr(
            ridge_pairwise_utils,
            "_release_model_context",
            self.release_model_context,
        )
        monkeypatch.setattr(ridge_pairwise_utils.gc, "collect", Mock(return_value=0))
        monkeypatch.setattr(
            ridge_pairwise_utils.torch.cuda,
            "is_available",
            Mock(return_value=False),
        )

    def record_event(self, event_name: str, model_name: str) -> None:
        """Append one thread-aware lifecycle event safely.

        Inputs:
            event_name: str describing load, extraction, ridge, or release.
            model_name: str identifying the synthetic model involved.

        Outputs:
            None. A tuple containing the event, model, and thread ID is stored.
        """

        with self.state_lock:
            self.event_log.append(
                (event_name, model_name, threading.get_ident())
            )

    def resolve_pipeline_function(
        self,
        function_config: dict[str, Any],
        registry: dict[str, Any],
        step_name: str,
        *,
        required: bool,
    ) -> Any:
        """Resolve runner pipeline phases to deterministic harness methods.

        Inputs:
            function_config: dict[str, Any] accepted for signature compatibility.
            registry: dict[str, Any] accepted for signature compatibility.
            step_name: str naming the requested configured phase.
            required: bool accepted for signature compatibility.

        Outputs:
            Bound callable for the requested phase, or None for an unknown phase.
        """

        del function_config, registry, required
        return {
            "target_extraction": self.target_extraction,
            "feature_extraction": self.feature_extraction,
            "pid_calculation": self.pid_calculation,
            "pid_report": self.pid_report,
        }.get(step_name)

    def target_extraction(self, **kwargs: Any) -> dict[str, Any]:
        """Return one small aligned target context with a closeable HDF handle.

        Inputs:
            **kwargs: Any configured target options, accepted but not used.

        Outputs:
            dict[str, Any] containing six target rows, a held-out Boolean mask,
            and the synthetic HDF handle checked by cleanup assertions.
        """

        del kwargs
        self.counts["target_extraction"] += 1
        return {
            "target": np.arange(18.0).reshape(6, 3),
            "shared1000_subj": self.shared_mask.copy(),
            "hdf_file": self.hdf_file,
            "img_ids": np.arange(6),
        }

    def overall_best_layer(
        self,
        model_name: str,
        path_to_results: str,
    ) -> dict[str, int]:
        """Select a deterministic valid layer for one synthetic model.

        Inputs:
            model_name: str identifying the requested model.
            path_to_results: str accepted for compatibility with layer lookup.

        Outputs:
            dict[str, int] containing a valid zero-based layer index under ``l``.
        """

        del path_to_results
        self.counts[f"layer:{model_name}"] += 1
        return {"l": self.model_names.index(model_name)}

    def load_ridge_alphas(
        self,
        alphas_path: str | Path,
        *,
        model_name: str,
        expected_target_dim: int,
        expected_layer_index: int | None = None,
    ) -> np.ndarray:
        """Return model-coded penalties while checking requested metadata.

        Inputs:
            alphas_path: str or Path to an existing empty test archive.
            model_name: str identifying the archived model.
            expected_target_dim: int defining the target-PC count.
            expected_layer_index: Optional int defining selected layer metadata.

        Outputs:
            np.ndarray with one deterministic positive alpha per target PC.
        """

        assert Path(alphas_path).is_file()
        assert expected_layer_index == self.model_names.index(model_name)
        self.counts[f"alphas:{model_name}"] += 1
        return np.full(expected_target_dim, self.model_values[model_name])

    def prepare_ridge_target(
        self,
        target: np.ndarray,
        target_context: dict[str, Any],
        pc_target_path: str | Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split one raw target into synthetic train/test PC arrays.

        Inputs:
            target: np.ndarray containing all six unstandardized target rows.
            target_context: dict[str, Any] containing the shared-image mask.
            pc_target_path: str or Path to the empty target-PCA artifact.

        Outputs:
            Tuple of training target, held-out target, and copied shared mask.
        """

        assert Path(pc_target_path) == self.target_pca_path
        np.testing.assert_array_equal(
            target,
            np.arange(18.0).reshape(6, 3),
        )
        np.testing.assert_array_equal(
            target_context["shared1000_subj"],
            self.shared_mask,
        )
        self.counts["prepare_target"] += 1
        train_target = np.asarray(target)[~self.shared_mask, :2].copy()
        return train_target, self.test_target.copy(), self.shared_mask.copy()

    def prepare_model_context(self, model_name: str) -> dict[str, Any]:
        """Create one tracked model context, coordinating optional overlap/failure.

        Inputs:
            model_name: str canonical synthetic model identifier to load.

        Outputs:
            dict[str, Any] with the model name and enough discovered layers for
            the configured layer index.

        Raises:
            RuntimeError: At the first background load when ``fail_stage`` is
            ``worker``.
        """

        is_background = threading.get_ident() != self.main_thread_ident
        coordinate_this_load = False
        with self.state_lock:
            if (
                is_background
                and self.coordinate_overlap
                and not self.coordinated_background_load
            ):
                self.coordinated_background_load = True
                coordinate_this_load = True
        self.counts[f"load:{model_name}"] += 1
        self.record_event("load_start", model_name)
        if coordinate_this_load:
            self.prefetch_started.set()
            if not self.pid_started.wait(timeout=5.0):
                raise AssertionError("Timed out waiting for PID overlap.")
            if self.fail_stage == "worker" and not self.worker_failure_raised:
                self.worker_failure_raised = True
                self.record_event("load_error", model_name)
                self.prefetch_terminal.set()
                raise RuntimeError("synthetic worker failure")

        context: dict[str, Any] = {
            "model_name": model_name,
            "layers_ordered": [object() for _ in self.model_names],
        }
        with self.state_lock:
            self.live_context_ids.add(id(context))
            self.max_live_contexts = max(
                self.max_live_contexts,
                len(self.live_context_ids),
            )
            self.created_contexts[model_name].append(context)
        self.record_event("load_end", model_name)
        return context

    def release_model_context(self, model_context: dict[str, Any] | None) -> None:
        """Track and perform the production context-clearing operation.

        Inputs:
            model_context: Optional dict[str, Any] representing a live context.

        Outputs:
            None. Live-context bookkeeping is updated and the context is cleared.
        """

        if model_context is not None:
            model_name = str(model_context.get("model_name", "unknown"))
            with self.state_lock:
                self.live_context_ids.discard(id(model_context))
            self.record_event("release", model_name)
        _REAL_RELEASE_MODEL_CONTEXT(model_context)

    def feature_extraction(
        self,
        *,
        source_context: dict[str, Any],
        layer_index: int,
        target_context: dict[str, Any],
        **kwargs: Any,
    ) -> np.ndarray:
        """Return model-coded features and record the extraction thread.

        Inputs:
            source_context: dict[str, Any] containing model and discovered layers.
            layer_index: int selected by the synthetic layer lookup.
            target_context: dict[str, Any] containing aligned sample identifiers.
            **kwargs: Any configured extraction options, accepted but not used.

        Outputs:
            np.ndarray with six rows and three deterministic feature columns.
        """

        del kwargs
        model_name = str(source_context["model_name"])
        assert layer_index == self.model_names.index(model_name)
        assert np.asarray(target_context["img_ids"]).shape == (6,)
        self.hdf_file.close.assert_not_called()
        self.counts[f"extract:{model_name}"] += 1
        self.record_event("extract", model_name)
        return np.full((6, 3), self.model_values[model_name])

    def ridge_predict_shared(
        self,
        source: np.ndarray,
        train_target: np.ndarray,
        shared_mask: np.ndarray,
        alphas: np.ndarray,
        *,
        seed: int,
    ) -> np.ndarray:
        """Return held-out model-coded predictions with optional failure overlap.

        Inputs:
            source: np.ndarray containing all unstandardized model feature rows.
            train_target: np.ndarray containing non-shared target PC rows.
            shared_mask: np.ndarray selecting held-out rows.
            alphas: np.ndarray whose first value identifies the synthetic model.
            seed: int configured ridge random seed.

        Outputs:
            np.ndarray of held-out predictions in target-PC space.

        Raises:
            RuntimeError: On the first call when ``fail_stage`` is ``ridge``.
        """

        assert seed == 73
        np.testing.assert_array_equal(shared_mask, self.shared_mask)
        model_value = float(np.asarray(alphas)[0])
        model_name = next(
            name for name, value in self.model_values.items() if value == model_value
        )
        np.testing.assert_array_equal(
            source,
            np.full((6, 3), model_value),
        )
        self.counts[f"ridge:{model_name}"] += 1

        is_background = threading.get_ident() != self.main_thread_ident
        self.record_event("ridge_start", model_name)
        if (
            self.fail_stage == "ridge"
            and is_background
            and not self.ridge_failure_raised
        ):
            self.ridge_failure_raised = True
            self.prefetch_terminal.set()
            raise RuntimeError("synthetic ridge failure")
        prediction = np.full(
            (int(np.asarray(shared_mask).sum()), train_target.shape[1]),
            model_value,
        )
        if is_background and self.coordinate_overlap:
            self.prefetch_terminal.set()
        return prediction

    def pid_calculation(
        self,
        target: np.ndarray,
        source_1: np.ndarray,
        source_2: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Record ordered PID inputs and return a deterministic result schema.

        Inputs:
            target: np.ndarray containing held-out target PCs.
            source_1: np.ndarray containing X1 held-out ridge predictions.
            source_2: np.ndarray containing X2 held-out ridge predictions.
            **kwargs: Any configured PID options, accepted for compatibility.

        Outputs:
            dict[str, Any] containing standard PID and MI component mappings.

        Raises:
            RuntimeError: On every call when ``fail_stage`` is ``pid``.
        """

        assert kwargs["rng_seed"] == 73
        np.testing.assert_allclose(target, self.test_target)
        self.counts["pid"] += 1
        self.pid_inputs.append(
            (
                np.asarray(target).copy(),
                np.asarray(source_1).copy(),
                np.asarray(source_2).copy(),
            )
        )
        coordinate_this_pid = self.coordinate_overlap and self.first_pid_call
        if coordinate_this_pid:
            self.first_pid_call = False
            if not self.prefetch_started.wait(timeout=5.0):
                raise AssertionError("Timed out waiting for prediction prefetch.")
            self.record_event("pid_start", "current-pair")
            self.pid_started.set()
            if not self.prefetch_terminal.wait(timeout=5.0):
                raise AssertionError("Timed out waiting for prefetched prediction.")
            self.record_event("pid_end", "current-pair")
        if self.fail_stage == "pid":
            raise RuntimeError("synthetic PID failure")
        source_1_value = float(np.asarray(source_1)[0, 0])
        source_2_value = float(np.asarray(source_2)[0, 0])
        return {
            "pid": {
                "red": 0.25,
                "unq1": source_1_value,
                "unq2": source_2_value,
                "syn": source_1_value + source_2_value,
            },
            "mi": {
                "bi_mi_1": source_1_value + 0.25,
                "bi_mi_2": source_2_value + 0.25,
                "tri_mi": source_1_value + source_2_value + 0.25,
            },
            "method": "synthetic-method",
        }

    def pid_report(
        self,
        pid_results: dict[str, Any],
        context: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Count reports and validate their X1/X2 context orientation.

        Inputs:
            pid_results: dict[str, Any] returned by ``pid_calculation``.
            context: dict[str, Any] containing target, ordered sources, and models.
            **kwargs: Any configured reporter options, accepted but not used.

        Outputs:
            None. Assertions validate context values and a counter is incremented.
        """

        del kwargs
        assert "pid" in pid_results
        source_1_value = self.model_values[str(context["model_1"])]
        source_2_value = self.model_values[str(context["model_2"])]
        assert float(np.asarray(context["source_1"])[0, 0]) == source_1_value
        assert float(np.asarray(context["source_2"])[0, 0]) == source_2_value
        np.testing.assert_allclose(context["target"], self.test_target)
        self.counts["report"] += 1

    def run(
        self,
        csv_path: str | Path,
        *,
        model_1_names: list[str] | None = None,
        model_2_names: list[str] | None = None,
    ) -> Path:
        """Invoke the real pairwise runner with the harness configuration.

        Inputs:
            csv_path: str or Path used for checkpoint output and resume.
            model_1_names: Optional list[str] overriding X1 candidates.
            model_2_names: Optional list[str] overriding X2 candidates.

        Outputs:
            Exact Path returned by ``run_pairwise_pid_pipeline``.
        """

        source_1_names = (
            self.model_names if model_1_names is None else model_1_names
        )
        source_2_names = (
            self.model_names if model_2_names is None else model_2_names
        )
        return ridge_pair_wise_comp.run_pairwise_pid_pipeline(
            model_1_names=list(source_1_names),
            model_2_names=list(source_2_names),
            otc_config=self.config,
            csv_path=csv_path,
        )


def test_pairwise_runner_prefetches_full_prediction_during_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check next-model preprocessing overlaps PID with one worker and context.

    Inputs:
        tmp_path: Path provided by pytest for artifacts and checkpoint output.
        monkeypatch: pytest.MonkeyPatch installing deterministic runner boundaries.

    Outputs:
        None. Assertions cover once-only work, lifecycle order, PID orientation,
        schema, checkpoint path, reporter calls, and HDF cleanup.
    """

    models = ["model-a", "model-b", "model-c"]
    harness = _RunnerHarness(
        monkeypatch,
        tmp_path,
        models,
        prefetch_enabled=True,
        coordinate_overlap=True,
    )
    csv_path = tmp_path / "nested" / "pairwise.csv"

    returned_path = harness.run(csv_path)

    assert returned_path == csv_path
    assert harness.counts["target_extraction"] == 1
    assert harness.counts["prepare_target"] == 1
    for model_name in models:
        assert harness.counts[f"layer:{model_name}"] == 1
        assert harness.counts[f"alphas:{model_name}"] == 1
        assert harness.counts[f"load:{model_name}"] == 1
        assert harness.counts[f"extract:{model_name}"] == 1
        assert harness.counts[f"ridge:{model_name}"] == 1
    assert harness.counts["pid"] == 3
    assert harness.counts["report"] == 3
    harness.hdf_file.close.assert_called_once_with()
    assert harness.max_live_contexts == 1
    assert harness.live_context_ids == set()
    assert all(
        context == {}
        for contexts in harness.created_contexts.values()
        for context in contexts
    )

    release_b_index = next(
        index
        for index, event in enumerate(harness.event_log)
        if event[:2] == ("release", "model-b")
    )
    load_c_index = next(
        index
        for index, event in enumerate(harness.event_log)
        if event[:2] == ("load_start", "model-c")
    )
    pid_start_index = next(
        index
        for index, event in enumerate(harness.event_log)
        if event[:2] == ("pid_start", "current-pair")
    )
    ridge_c_index = next(
        index
        for index, event in enumerate(harness.event_log)
        if event[:2] == ("ridge_start", "model-c")
    )
    pid_end_index = next(
        index
        for index, event in enumerate(harness.event_log)
        if event[:2] == ("pid_end", "current-pair")
    )
    assert release_b_index < load_c_index < pid_start_index < ridge_c_index
    assert ridge_c_index < pid_end_index

    extraction_threads_by_model = {
        model_name: thread_id
        for event_name, model_name, thread_id in harness.event_log
        if event_name == "extract"
    }
    background_load_threads = {
        thread_id
        for event_name, model_name, thread_id in harness.event_log
        if event_name == "load_start" and model_name == "model-c"
    }
    assert extraction_threads_by_model["model-a"] == harness.main_thread_ident
    assert extraction_threads_by_model["model-b"] == harness.main_thread_ident
    assert len(background_load_threads) == 1
    assert harness.main_thread_ident not in background_load_threads
    assert extraction_threads_by_model["model-c"] in background_load_threads

    results = pd.read_csv(csv_path)
    assert list(results.columns) == ridge_pairwise_utils.PAIRWISE_RESULT_COLUMNS
    assert list(zip(results["model_1"], results["model_2"], strict=True)) == [
        ("model-a", "model-b"),
        ("model-a", "model-c"),
        ("model-b", "model-c"),
    ]
    assert not (results["model_1"] == results["model_2"]).any()
    np.testing.assert_allclose(results["unq1"], [1.0, 1.0, 2.0])
    np.testing.assert_allclose(results["unq2"], [2.0, 3.0, 3.0])
    for target, source_1, source_2 in harness.pid_inputs:
        np.testing.assert_allclose(target, harness.test_target)
        assert source_1.shape == source_2.shape == harness.test_target.shape


def test_pairwise_runner_prefetch_toggle_produces_identical_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check enabled and sequential model loading preserve PID and CSV values.

    Inputs:
        tmp_path: Path provided by pytest for two independent synthetic runs.
        monkeypatch: pytest.MonkeyPatch replacing runner external boundaries.

    Outputs:
        None. Assertions compare complete CSV rows and ordered PID inputs.
    """

    models = ["model-a", "model-b", "model-c"]
    enabled_dir = tmp_path / "enabled"
    disabled_dir = tmp_path / "disabled"
    enabled_dir.mkdir()
    disabled_dir.mkdir()
    enabled_harness = _RunnerHarness(
        monkeypatch,
        enabled_dir,
        models,
        prefetch_enabled=True,
    )
    enabled_csv = enabled_dir / "results.csv"
    enabled_harness.run(enabled_csv)

    disabled_harness = _RunnerHarness(
        monkeypatch,
        disabled_dir,
        models,
        prefetch_enabled=False,
    )
    disabled_csv = disabled_dir / "results.csv"
    disabled_harness.run(disabled_csv)

    pd.testing.assert_frame_equal(
        pd.read_csv(enabled_csv),
        pd.read_csv(disabled_csv),
    )
    assert len(enabled_harness.pid_inputs) == len(disabled_harness.pid_inputs) == 3
    for enabled_inputs, disabled_inputs in zip(
        enabled_harness.pid_inputs,
        disabled_harness.pid_inputs,
        strict=True,
    ):
        for enabled_array, disabled_array in zip(
            enabled_inputs,
            disabled_inputs,
            strict=True,
        ):
            np.testing.assert_allclose(enabled_array, disabled_array)
    assert all(
        thread_id == disabled_harness.main_thread_ident
        for event_name, _, thread_id in disabled_harness.event_log
        if event_name == "load_start"
    )


def test_pairwise_runner_partial_and_full_resume_load_only_needed_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check reverse-pair resume, required-model filtering, and completed reruns.

    Inputs:
        tmp_path: Path provided by pytest for a pre-populated checkpoint.
        monkeypatch: pytest.MonkeyPatch installing deterministic runner boundaries.

    Outputs:
        None. Assertions show one missing unordered pair is appended, only its
        two models are processed, and a full resume performs no expensive work.
    """

    models = ["model-a", "model-b", "model-c", "model-d"]
    harness = _RunnerHarness(
        monkeypatch,
        tmp_path,
        models,
        prefetch_enabled=True,
    )
    csv_path = tmp_path / "resume.csv"
    completed_reverse_pairs = [
        ("model-b", "model-a"),
        ("model-c", "model-a"),
        ("model-d", "model-a"),
        ("model-c", "model-b"),
        ("model-d", "model-c"),
    ]
    checkpoint_rows: list[dict[str, Any]] = []
    for model_1, model_2 in completed_reverse_pairs:
        checkpoint_row = dict.fromkeys(
            ridge_pairwise_utils.PAIRWISE_RESULT_COLUMNS,
            0,
        )
        checkpoint_row.update(
            {
                "model_1": model_1,
                "model_2": model_2,
                "layer_1": models.index(model_1),
                "layer_2": models.index(model_2),
                "subj_id": "synthetic-subject",
                "pid_method": "synthetic-method",
            }
        )
        checkpoint_rows.append(checkpoint_row)
    pd.DataFrame(
        checkpoint_rows,
        columns=ridge_pairwise_utils.PAIRWISE_RESULT_COLUMNS,
    ).to_csv(csv_path, index=False)

    first_returned_path = harness.run(csv_path)

    assert first_returned_path == csv_path
    results = pd.read_csv(csv_path)
    assert len(results) == 6
    assert tuple(results.iloc[-1][["model_1", "model_2"]]) == (
        "model-b",
        "model-d",
    )
    for model_name in models:
        expected_count = 1 if model_name in {"model-b", "model-d"} else 0
        assert harness.counts[f"extract:{model_name}"] == expected_count
        assert harness.counts[f"ridge:{model_name}"] == expected_count
    assert harness.counts["target_extraction"] == 1
    assert harness.counts["pid"] == 1

    counts_after_partial_resume = harness.counts.copy()
    second_returned_path = harness.run(csv_path)

    assert second_returned_path == csv_path
    assert harness.counts == counts_after_partial_resume
    assert len(pd.read_csv(csv_path)) == 6
    unordered_pairs = [
        frozenset((model_1, model_2))
        for model_1, model_2 in zip(
            results["model_1"],
            results["model_2"],
            strict=True,
        )
    ]
    assert len(unordered_pairs) == len(set(unordered_pairs)) == 6
    assert all(len(pair) == 2 for pair in unordered_pairs)


@pytest.mark.parametrize(
    ("fail_stage", "expected_message"),
    [
        pytest.param(
            "worker",
            "Prefetched ridge prediction failed for 'model-c'",
            id="worker-failure",
        ),
        pytest.param(
            "ridge",
            "Prefetched ridge prediction failed for 'model-c'",
            id="ridge-failure",
        ),
        pytest.param(
            "pid",
            "synthetic PID failure",
            id="pid-failure",
        ),
    ],
)
def test_pairwise_runner_failures_release_contexts_and_close_hdf(
    fail_stage: str,
    expected_message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check worker, ridge, and PID exceptions preserve all cleanup guarantees.

    Inputs:
        fail_stage: str selecting the injected runner failure boundary.
        expected_message: str required in the propagated exception.
        tmp_path: Path provided by pytest for artifacts and checkpoint output.
        monkeypatch: pytest.MonkeyPatch installing deterministic failure fakes.

    Outputs:
        None. Assertions require HDF closure, context release/cancellation, and
        valid immediate checkpoints for PID pairs completed before the failure.
    """

    models = ["model-a", "model-b", "model-c"]
    harness = _RunnerHarness(
        monkeypatch,
        tmp_path,
        models,
        prefetch_enabled=True,
        coordinate_overlap=True,
        fail_stage=fail_stage,
    )
    csv_path = tmp_path / "failure.csv"

    with pytest.raises(RuntimeError, match=expected_message):
        harness.run(csv_path)

    harness.hdf_file.close.assert_called_once_with()
    assert harness.live_context_ids == set()
    assert harness.max_live_contexts == 1
    assert all(
        context == {}
        for contexts in harness.created_contexts.values()
        for context in contexts
    )
    results = pd.read_csv(csv_path)
    assert len(results) == (0 if fail_stage == "pid" else 1)
    if fail_stage == "ridge":
        assert len(harness.created_contexts["model-c"]) == 1
        assert any(
            event[:2] == ("release", "model-c")
            for event in harness.event_log
        )
    if fail_stage == "worker":
        assert harness.created_contexts["model-c"] == []
    if fail_stage == "pid":
        assert harness.counts["pid"] == 1
        assert harness.counts["extract:model-a"] == 1
        assert harness.counts["extract:model-b"] == 1
        assert harness.counts["extract:model-c"] == 1


def test_pairwise_main_plots_the_exact_runner_returned_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check plotting consumes the runner's returned checkpoint path verbatim.

    Inputs:
        tmp_path: Path provided by pytest for distinct configured/returned paths.
        monkeypatch: pytest.MonkeyPatch replacing YAML parsing, running, and plot.

    Outputs:
        None. Assertions validate that no second CSV path is constructed in main.
    """

    configured_csv = tmp_path / "configured.csv"
    returned_csv = tmp_path / "runner-returned.csv"
    plot_dir = tmp_path / "plots"
    config = {
        "models": ["model-a", "model-b"],
        "output": {
            "csv_path": str(configured_csv),
            "plot_dir": str(plot_dir),
        }
    }
    run_mock = Mock(return_value=returned_csv)
    plot_mock = Mock()
    monkeypatch.setattr(
        ridge_pair_wise_comp.yaml,
        "safe_load",
        Mock(return_value=config),
    )
    monkeypatch.setattr(
        ridge_pair_wise_comp,
        "run_pairwise_pid_pipeline",
        run_mock,
    )
    monkeypatch.setattr(
        ridge_pair_wise_comp,
        "plot_pairwise_pid_matrices",
        plot_mock,
    )

    ridge_pair_wise_comp.main()

    assert run_mock.call_args.kwargs["model_1_names"] == config["models"]
    assert run_mock.call_args.kwargs["model_2_names"] == config["models"]
    assert run_mock.call_args.kwargs["csv_path"] == configured_csv
    plot_mock.assert_called_once_with(
        csv_path=returned_csv,
        output_dir=plot_dir,
    )
