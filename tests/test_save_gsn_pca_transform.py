"""Check that the saved GSN PCA artifact contains a reusable transform only."""

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pipeline.pca.choosing_minimal_best_pca import save_gsn_pca_transform


def test_save_gsn_pca_transform_uses_csv_count_and_supports_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save only a loadable GSN-basis transform at either requested count."""

    selection_csv = tmp_path / "selection.csv"
    with selection_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["subj_id", "image_set", "best_n_components"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "subj_id": "subj01",
                    "image_set": "all",
                    "best_n_components": 2,
                },
                {
                    "subj_id": "subj01",
                    "image_set": "all",
                    "best_n_components": 2,
                },
            ]
        )

    # (18,) -> (3, 3, 2)
    responses = np.arange(18.0).reshape(3, 3, 2)
    loaded = {}

    def fake_load_repeat_responses(*args, **kwargs) -> dict:
        """Return deterministic repeated responses without reading NSD files."""

        loaded["image_set"] = kwargs["image_set"]
        return {"responses": responses}

    def fake_estimate_gsn_covariances(input_responses: np.ndarray) -> dict:
        """Return diagonal signal and noise covariances for a known basis."""

        np.testing.assert_array_equal(input_responses, responses)
        # Two length-3 diagonals -> two (3, 3) covariance matrices.
        return {
            "cSb": np.diag([1.0, 4.0, 2.0]),
            "cNb": np.diag([3.0, 1.0, 2.0]),
        }

    monkeypatch.setattr(
        save_gsn_pca_transform,
        "load_otc_responses",
        fake_load_repeat_responses,
    )
    monkeypatch.setattr(
        save_gsn_pca_transform,
        "estimate_gsn_covariances",
        fake_estimate_gsn_covariances,
    )

    output_dir = tmp_path / "models"
    model_path = save_gsn_pca_transform.save_gsn_pca_transform(
        "subj01",
        selection_csv,
        output_dir,
    )
    model = joblib.load(model_path)

    assert loaded["image_set"] == "all"
    assert model.n_components_ == 2
    assert model.image_set_ == "all"
    assert model.selection_csv_path_ == str(selection_csv.resolve())
    assert model_path.name == "subj01_all_gsn_signal_pca_2_components.joblib"
    assert model.components_.shape == (2, 3)
    assert model.mean_.shape == (3,)
    # (3, 3) -> (2, 3)
    np.testing.assert_allclose(
        np.abs(model.components_),
        np.eye(3)[:, [1, 2]].T,
    )
    # (3, 3, 2) -> (3,)
    expected_mean = responses.mean(axis=(0, 2))
    np.testing.assert_allclose(model.mean_, expected_mean)
    samples = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    # (2, 3) -> (2, 2)
    projected = model.transform(samples)
    np.testing.assert_allclose(
        projected,
        (samples - expected_mean) @ model.components_.T,
    )

    override_path = save_gsn_pca_transform.save_gsn_pca_transform(
        "subj01",
        selection_csv,
        output_dir,
        n_components=1,
    )
    override_model = joblib.load(override_path)
    assert override_model.n_components_ == 1
    assert sorted(output_dir.iterdir()) == sorted([model_path, override_path])
