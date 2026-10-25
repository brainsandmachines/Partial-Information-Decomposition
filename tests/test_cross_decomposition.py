"""Small synthetic checks; never open NSD stores or launch real experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline.pca.cross_decomposition import cross_decomposition as cross
from pipeline.pca.cross_decomposition import compare_reconstruction as runner
from pipeline.testing.gsn import gsn_pca


@pytest.mark.parametrize("shape,k,solver", [
    ((18, 5), 5, "exact"), ((8, 25), 6, "exact"),
    ((18, 25), 3, "iterative"), ((18, 25), 3, "auto"),
])
def test_svd_against_explicit_covariance(shape: tuple, k: int, solver: str) -> None:
    """Inputs: synthetic array shape, int k and solver str; output: None/assertions."""
    rng = np.random.default_rng(10)
    first, second = rng.normal(size=shape), rng.normal(size=shape)
    fitted = cross.fit_cross_decomposition(first, second, k, seed=5, solver=solver)
    # (n,d) -> (d,); (d,n) @ (n,d) -> (d,d).
    a, b = first - first.mean(axis=0), second - second.mean(axis=0)
    covariance = a.T @ b / shape[0]
    # (d,d) -> (d,d), (d,), (d,d), an independent dense reference.
    u, s, vt = np.linalg.svd(covariance)
    np.testing.assert_allclose(fitted["singular_values"], s[:k], rtol=1e-8, atol=1e-10)
    assert np.all(np.diff(fitted["singular_values"]) <= 0)
    for basis, reference in ((fitted["left_basis"], u[:, :k]),
                             (fitted["right_basis"], vt[:k].T)):
        # (d,k) @ (k,d) -> (d,d).
        projection = basis @ basis.T
        np.testing.assert_allclose(projection, reference @ reference.T, atol=1e-8)
        np.testing.assert_allclose(projection, projection.T, atol=1e-12)
        np.testing.assert_allclose(projection @ projection, projection, atol=1e-10)
    np.testing.assert_allclose(covariance @ fitted["right_basis"],
                               fitted["left_basis"] * fitted["singular_values"], atol=1e-8)
    repeated = cross.fit_cross_decomposition(first, second, k, seed=5, solver=solver)
    np.testing.assert_allclose(repeated["left_basis"], fitted["left_basis"])


def test_zero_full_and_incremental_reconstruction() -> None:
    """No inputs; assert every efficient curve value equals literal reconstruction."""
    rng = np.random.default_rng(11)
    train_a, train_b = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
    fitted = cross.fit_cross_decomposition(train_a, train_b, 4)
    observed, target = rng.normal(size=(7, 4)), rng.normal(size=(7, 4))
    for basis, mean in ((fitted["left_basis"], fitted["first_mean"]),
                        (fitted["right_basis"], fitted["second_mean"])):
        curve = cross.normalized_reconstruction_curve(observed, target, mean, basis)
        for k in range(5):
            # (7,4) @ (4,k) @ (k,4) -> (7,4).
            reconstruction = mean + (observed - mean) @ basis[:, :k] @ basis[:, :k].T
            expected = np.mean((target - reconstruction) ** 2) / np.mean((target - mean) ** 2)
            np.testing.assert_allclose(curve[k], expected, atol=1e-12)
            if k == 0:
                np.testing.assert_allclose(reconstruction, np.broadcast_to(mean, observed.shape))
            if k == 4:
                np.testing.assert_allclose(reconstruction, observed, atol=1e-12)
        assert curve[0] == 1


def test_extra_noise_dimension_can_worsen_reconstruction() -> None:
    """No inputs; independent noise in a second dimension worsens held-out MSE."""
    rng = np.random.default_rng(14)
    signal = rng.normal(size=1000)
    # Two (1000,) arrays -> (1000,2); noise draws are independent across repeats.
    observed = np.column_stack((signal, rng.normal(size=1000)))
    target = np.column_stack((signal, rng.normal(size=1000)))
    curve = cross.normalized_reconstruction_curve(observed, target, np.zeros(2), np.eye(2))
    assert curve[1] < curve[0]
    assert curve[2] > curve[1]


@pytest.mark.parametrize("weighting", ["equal", "input_repeat_count"])
def test_same_protocol_as_unchanged_gsn(monkeypatch: pytest.MonkeyPatch, weighting: str) -> None:
    """Inputs: patch fixture and weighting str; output: None. Fix equal bases to
    isolate folds, repeat averaging, pooled means, normalization and selection.
    """
    responses = np.random.default_rng(2).normal(size=(12, 3, 3))
    gsn_training, cross_training = [], []

    def fixed_covariances(training: np.ndarray) -> dict:
        """Input: training responses (n,3,3); output: diagonal covariance dict."""
        gsn_training.append(training.copy())
        return {"cSb": np.diag([3., 2., 1.]), "cNb": np.eye(3)}

    def fixed_decomposition(first: np.ndarray, second: np.ndarray, *args) -> dict:
        """Inputs: training repeat means (n,3), unused options; output: identity bases."""
        cross_training.append((first.copy(), second.copy()))
        return {"left_basis": np.eye(3), "right_basis": np.eye(3)}

    monkeypatch.setattr(gsn_pca, "estimate_gsn_covariances", fixed_covariances)
    monkeypatch.setattr(cross, "fit_cross_decomposition", fixed_decomposition)
    options = dict(max_components=3, n_folds=3, seed=5, repeat_split_weighting=weighting)
    original = gsn_pca.select_gsn_pca_components(responses, **options)
    actual = cross.select_cross_decomposition_components(responses, **options)
    for key in original:
        np.testing.assert_allclose(actual[key], original[key], atol=1e-12)
    folds, splits, weights = cross.validation_design(12, 3, 3, 5, weighting)
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(12))
    pair_index = 0
    for fold_index, validation_indices in enumerate(folds):
        # (12,) -> (n_train,), disjoint complementary stimulus sets.
        training_indices = np.setdiff1d(np.arange(12), validation_indices)
        assert not np.intersect1d(training_indices, validation_indices).size
        np.testing.assert_array_equal(gsn_training[fold_index], responses[training_indices])
        for a, b in splits[::2]:
            # (n_train,3,n_group) -> (n_train,3).
            np.testing.assert_allclose(cross_training[pair_index][0],
                                       responses[training_indices][:, :, a].mean(axis=2))
            np.testing.assert_allclose(cross_training[pair_index][1],
                                       responses[training_indices][:, :, b].mean(axis=2))
            pair_index += 1
    expected = 2 / 3 if weighting == "input_repeat_count" else 1 / 2
    assert sum(w for (a, _), w in zip(splits, weights) if len(a) == 2) == pytest.approx(expected)


def test_two_directions_against_literal_training_only_reference() -> None:
    """No inputs; compare all folds to dense SVD and explicit U/V projections."""
    responses = np.random.default_rng(21).normal(size=(15, 3, 2))
    result = cross.select_cross_decomposition_components(responses, 3, 3, 6, solver="exact")
    folds, _, _ = cross.validation_design(15, 2, 3, 6, "input_repeat_count")
    for fold_index, validation_indices in enumerate(folds):
        # (15,3,2) -> (10,3,2), (5,3,2).
        training = np.delete(responses, validation_indices, axis=0)
        validation = responses[validation_indices]
        # (10,3,2) -> (3,), (10,3).
        mean = training.mean(axis=(0, 2))
        a, b = training[:, :, 0], training[:, :, 1]
        # (10,3) -> (3,3) -> two (3,3) bases and (3,) singular values.
        covariance = (a - a.mean(axis=0)).T @ (b - b.mean(axis=0)) / len(training)
        u, _, vt = np.linalg.svd(covariance)
        for direction, basis in enumerate((u, vt.T)):
            # (5,3,2) -> (5,3).
            observed, target = validation[:, :, direction], validation[:, :, 1 - direction]
            for k in range(1, 4):
                # (5,3) @ (3,k) @ (k,3) -> (5,3).
                estimate = mean + (observed - mean) @ basis[:, :k] @ basis[:, :k].T
                expected = np.mean((target - estimate) ** 2) / np.mean((target - mean) ** 2)
                assert result["direction_normalized_reconstruction_errors"][2 * fold_index + direction, k - 1] == pytest.approx(expected)
    # Full basis reconstructs observed, so no mean or fitted-basis dependence remains.
    assert np.isfinite(result["fold_normalized_reconstruction_errors"]).all()


def test_test_data_cannot_change_first_fold_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Input: patch fixture; output: None. Alter held-out rows, inspect fit inputs."""
    responses = np.random.default_rng(23).normal(size=(18, 4, 2))
    captured = []
    original = cross.fit_cross_decomposition

    def capture(first: np.ndarray, second: np.ndarray, *args) -> dict:
        """Inputs: training arrays (n,4), options; output: fitted SVD dict, recorded."""
        fitted = original(first, second, *args)
        captured.append(fitted)
        return fitted

    monkeypatch.setattr(cross, "fit_cross_decomposition", capture)
    cross.select_cross_decomposition_components(responses, 2, 3, seed=0)
    folds, _, _ = cross.validation_design(18, 2, 3, 0, "input_repeat_count")
    changed = responses.copy()
    changed[folds[0]] += 1000
    cross.select_cross_decomposition_components(changed, 2, 3, seed=0)
    for key in ("first_mean", "second_mean", "left_basis", "right_basis", "singular_values"):
        np.testing.assert_allclose(captured[0][key], captured[3][key])


def test_high_dimensional_selection_and_invalid_inputs() -> None:
    """No inputs; support d>n and fail explicitly on invalid grids/data/baselines."""
    responses = np.random.default_rng(1).normal(size=(12, 20, 3))
    result = cross.select_cross_decomposition_components(responses, 2, 3)
    assert result["fold_normalized_reconstruction_errors"].shape == (3, 2)
    assert 1 <= result["selected_n_components"] <= result["best_n_components"] <= 2
    with pytest.raises(ValueError, match="rank bound"):
        cross.select_cross_decomposition_components(responses, 8, 3)
    with pytest.raises(ValueError, match="finite"):
        cross.select_cross_decomposition_components(responses * np.nan, 2, 3)
    with pytest.raises(ValueError, match="two training"):
        cross.select_cross_decomposition_components(responses[:3], 1, 2)
    with pytest.raises(ValueError, match="reconstructable"):
        cross.normalized_reconstruction_curve(np.ones((3, 2)), np.zeros((3, 2)),
                                              np.zeros(2), np.eye(2))


def test_comparison_runner_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inputs: temp directory and patch fixture; output: None. Tiny real-GSN run
    checks loader reuse, resolved config, saved paired errors, plots and no overwrite.
    """
    if gsn_pca.GSN_IMPORT_ERROR is not None:
        pytest.skip("official GSN unavailable")
    rng = np.random.default_rng(8)
    # (30,3,1) + (30,3,3) -> (30,3,3).
    responses = rng.normal(size=(30, 3, 1)) + 0.4 * rng.normal(size=(30, 3, 3))
    calls = []

    def load_tiny(subject: str, **kwargs) -> dict:
        """Input: subject str/loader kwargs; output: synthetic responses and image IDs."""
        calls.append((subject, kwargs))
        # Scalar n_images -> (n_images,).
        return {"responses": responses, "image_ids": np.arange(30) + 100}

    monkeypatch.setattr(runner, "load_repeat_responses", load_tiny)
    source = {
        "functions": {"selection_function": "gsn_independent_repeat"},
        "subjects": ["subj99"], "image_set": "unique", "neural_data_dir": "unused",
        "pkl_info_path": "unused.pkl", "selection_kwargs": {
            "max_components": 3, "n_folds": 3, "seed": 7,
            "repeat_split_weighting": "input_repeat_count",
        },
    }
    source_path, config_path = tmp_path / "source.yaml", tmp_path / "comparison.yaml"
    source_path.write_text(yaml.safe_dump(source))
    config_path.write_text(yaml.safe_dump({"gsn_config_path": str(source_path),
                                         "output_dir": str(tmp_path / "results"), "svd_solver": "exact"}))
    first, = runner.run_comparison(config_path)
    second, = runner.run_comparison(config_path)
    assert first != second and first.exists() and second.exists()
    assert len(calls) == 2 and calls[0][1]["image_set"] == "unique"
    expected_files = {"gsn_pca.csv", "cross_decomposition.csv", "summary.csv",
                      "paired_differences.csv", "image_folds.csv", "repeat_splits.csv",
                      "run_config.yaml", "reconstruction_comparison.pdf"}
    assert {p.name for p in first.iterdir()} == expected_files
    assert (first / "reconstruction_comparison.pdf").read_bytes().startswith(b"%PDF")
    for filename in ("gsn_pca.csv", "cross_decomposition.csv", "image_folds.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(first / filename), pd.read_csv(second / filename))
    gsn = pd.read_csv(first / "gsn_pca.csv")
    cd = pd.read_csv(first / "cross_decomposition.csv")
    difference = pd.read_csv(first / "paired_differences.csv")
    for fold in range(1, 4):
        key = f"fold_normalized_reconstruction_errors_{fold}"
        np.testing.assert_allclose(difference[f"fold_cross_minus_gsn_{fold}"], cd[key] - gsn[key], atol=1e-12)
    np.testing.assert_allclose(gsn["mean_normalized_reconstruction_errors"].iloc[-1],
                               cd["mean_normalized_reconstruction_errors"].iloc[-1], atol=1e-12)
