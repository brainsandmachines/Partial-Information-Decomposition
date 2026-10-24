"""Offline tests for the minimal-PC selection runner."""

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

zarr = pytest.importorskip("zarr")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pipeline.pca.choosing_minimal_best_pca import choosing_minimal_best_pca
from pipeline.pca.choosing_minimal_best_pca.choosing_minimal_best_pca import (
    choose_minimal_pcs,
    load_repeat_responses,
    raise_on_nonfinite_images,
    selection_results_to_table,
)


SUBJ_ID = "subj99"
IMAGE_IDS = np.array([10, 3, 7, 0, 5, 8])


def make_responses() -> np.ndarray:
    """Build hand-checkable, fully finite repeated responses.

    Inputs:
        None.

    Output:
        responses: np.ndarray with shape (6, 4, 3) in (image, voxel, repeat)
            order, holding the values 0 to 71.
    """

    # (72,) -> (6, 4, 3)
    return np.arange(72.0).reshape(6, 4, 3)


def write_store(neural_data_dir: Path, responses: np.ndarray) -> None:
    """Write responses as a ``{SUBJ_ID}_betas_img_vox_rep.zarr`` store.

    The array is stored in (repeat, image, voxel) order so that the loader's
    transpose to (image, voxel, repeat) is exercised.

    Inputs:
        neural_data_dir: Path, directory in which the store is created.
        responses: np.ndarray with shape (n_images, n_voxels, n_repeats).

    Output:
        None. The Zarr store is written to disk.
    """

    n_images, n_voxels, n_repeats = responses.shape
    betas = xr.DataArray(
        # (n_images, n_voxels, n_repeats) -> (n_repeats, n_images, n_voxels)
        responses.transpose(2, 0, 1),
        dims=("repeat", "image", "voxel"),
        coords={
            "repeat": np.arange(n_repeats),
            "image": IMAGE_IDS[:n_images],
            "voxel": np.arange(n_voxels),
        },
    )
    store_path = neural_data_dir / f"{SUBJ_ID}_betas_img_vox_rep.zarr"
    betas.to_dataset(name="betas_by_image_voxel_repeat").to_zarr(store_path, mode="w")


def patch_shared_flags(
    monkeypatch: pytest.MonkeyPatch,
    shared_ids: list[int],
) -> None:
    """Replace ``get_stim_info`` with fixed shared1000 flags.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, fixture used to patch the module.
        shared_ids: list[int], NSD image IDs flagged as shared1000.

    Output:
        None. The runner module's ``get_stim_info`` is patched.
    """

    shared1000_all = np.zeros(12, dtype=bool)
    shared1000_all[shared_ids] = True

    def fake_get_stim_info(pkl_info_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Return fixed NSD IDs and shared1000 flags.

        Inputs:
            pkl_info_path: Path, ignored stimulus-information path.

        Output:
            stim_info: tuple of np.ndarray, IDs of shape (12,) and Boolean
                shared1000 flags of shape (12,).
        """

        return np.arange(12), shared1000_all

    monkeypatch.setattr(choosing_minimal_best_pca, "get_stim_info", fake_get_stim_info)


def forbid_stim_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any ``get_stim_info`` call fail the test.

    Inputs:
        monkeypatch: pytest.MonkeyPatch, fixture used to patch the module.

    Output:
        None. The runner module's ``get_stim_info`` is patched to raise.
    """

    def failing_get_stim_info(pkl_info_path: Path) -> None:
        """Raise because the full image set must not read shared1000 flags.

        Inputs:
            pkl_info_path: Path, ignored stimulus-information path.

        Output:
            None. Always raises AssertionError.
        """

        raise AssertionError("get_stim_info must not be called for image_set='all'.")

    monkeypatch.setattr(choosing_minimal_best_pca, "get_stim_info", failing_get_stim_info)


def test_loader_all_keeps_every_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep all images, in stored order, without reading shared flags."""

    responses = make_responses()
    write_store(tmp_path, responses)
    forbid_stim_info(monkeypatch)
    report_dir = tmp_path / "Nonfinite"

    data = load_repeat_responses(
        SUBJ_ID,
        image_set="all",
        neural_data_dir=tmp_path,
        nonfinite_report_dir=report_dir,
    )

    np.testing.assert_array_equal(data["image_ids"], IMAGE_IDS)
    np.testing.assert_array_equal(data["responses"], responses)
    assert data["n_outside_set_removed"] == 0
    assert not report_dir.exists()


def test_loader_unique_removes_shared_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove shared1000 images and keep every other image in stored order."""

    responses = make_responses()
    write_store(tmp_path, responses)
    # Image id 7 (row 2) is in the store; 11 is not.
    patch_shared_flags(monkeypatch, [7, 11])

    data = load_repeat_responses(
        SUBJ_ID,
        image_set="unique",
        pkl_info_path=tmp_path / "unused.pkl",
        neural_data_dir=tmp_path,
        nonfinite_report_dir=tmp_path / "Nonfinite",
    )

    np.testing.assert_array_equal(data["image_ids"], [10, 3, 0, 5, 8])
    # (6, 4, 3) -> (5, 4, 3)
    np.testing.assert_array_equal(data["responses"], responses[[0, 1, 3, 4, 5]])
    assert data["n_outside_set_removed"] == 1


def test_loader_reports_nonfinite_images_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write one report row per bad (image, repeat) pair, then raise."""

    responses = make_responses()
    responses[1, :, 1] = np.nan  # image 3, repeat 1: every voxel missing
    responses[1, 0, 2] = np.inf  # image 3, repeat 2: one voxel
    responses[4, 1, 2] = np.nan  # image 5, repeat 2: one voxel
    write_store(tmp_path, responses)
    forbid_stim_info(monkeypatch)
    report_dir = tmp_path / "Nonfinite"

    with pytest.raises(ValueError, match="2 images have non-finite") as error:
        load_repeat_responses(
            SUBJ_ID,
            image_set="all",
            neural_data_dir=tmp_path,
            nonfinite_report_dir=report_dir,
        )

    message = str(error.value)
    assert "image 3: repeats [1, 2]" in message
    assert "image 5: repeats [2]" in message
    report = pd.read_csv(report_dir / f"{SUBJ_ID}_all_nonfinite.csv")
    expected = pd.DataFrame(
        {
            "subj_id": [SUBJ_ID] * 3,
            "image_set": ["all"] * 3,
            "image_id": [3, 3, 5],
            "repeat": [1, 2, 2],
            "n_nonfinite_voxels": [4, 1, 1],
            "n_voxels": [4, 4, 4],
        }
    )
    pd.testing.assert_frame_equal(report, expected)


def test_unique_set_ignores_nonfinite_shared_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never inspect shared images unless image_set is 'all'."""

    responses = make_responses()
    responses[2, 3, 0] = np.nan  # image 7, which is shared
    write_store(tmp_path, responses)
    patch_shared_flags(monkeypatch, [7])
    report_dir = tmp_path / "Nonfinite"

    data = load_repeat_responses(
        SUBJ_ID,
        image_set="unique",
        pkl_info_path=tmp_path / "unused.pkl",
        neural_data_dir=tmp_path,
        nonfinite_report_dir=report_dir,
    )
    np.testing.assert_array_equal(data["image_ids"], [10, 3, 0, 5, 8])
    assert not report_dir.exists()

    with pytest.raises(ValueError, match="image 7: repeats \\[0\\]"):
        load_repeat_responses(
            SUBJ_ID,
            image_set="all",
            neural_data_dir=tmp_path,
            nonfinite_report_dir=report_dir,
        )


def test_nonfinite_message_lists_limited_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List only the first images in the error; the CSV keeps all of them."""

    monkeypatch.setattr(choosing_minimal_best_pca, "MAX_LISTED_NONFINITE_IMAGES", 1)
    # (2, 1, 2) array with one NaN in each image.
    responses = np.array([[[np.nan, 1.0]], [[2.0, np.nan]]])

    with pytest.raises(ValueError, match="and 1 more") as error:
        raise_on_nonfinite_images(
            responses,
            np.array([40, 41]),
            np.array([0, 1]),
            SUBJ_ID,
            "all",
            report_dir=tmp_path,
        )

    assert "image 40: repeats [0]" in str(error.value)
    assert "image 41" not in str(error.value)
    report = pd.read_csv(tmp_path / f"{SUBJ_ID}_all_nonfinite.csv")
    assert report["image_id"].tolist() == [40, 41]


def test_loader_rejects_bad_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise on unknown image sets, missing stores, and unmatched shared IDs."""

    write_store(tmp_path, make_responses())
    patch_shared_flags(monkeypatch, [11])

    with pytest.raises(ValueError, match="No shared1000"):
        load_repeat_responses(
            SUBJ_ID,
            image_set="unique",
            pkl_info_path=tmp_path / "unused.pkl",
            neural_data_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="image_set"):
        load_repeat_responses(SUBJ_ID, image_set="shared", neural_data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_repeat_responses("subj00", image_set="all", neural_data_dir=tmp_path)


def test_selection_results_to_table_layout() -> None:
    """Write scalars, per-candidate vectors, and per-fold rows as columns."""

    results = {
        "selected_n_components": 2,
        "best_n_components": 3,
        "candidate_n_components": np.array([1, 2, 3]),
        "fold_errors": np.array([[0.9, 0.5, 0.4], [0.8, 0.6, 0.5]]),
        "mean_errors": np.array([0.85, 0.55, 0.45]),
        "threshold": 0.6,
    }

    table = selection_results_to_table(results, SUBJ_ID)

    assert list(table.columns) == [
        "subj_id",
        "candidate_n_components",
        "selected_n_components",
        "best_n_components",
        "fold_errors_1",
        "fold_errors_2",
        "mean_errors",
        "threshold",
        "is_selected",
    ]
    assert table.shape == (3, 9)
    assert (table["subj_id"] == SUBJ_ID).all()
    assert (table["selected_n_components"] == 2).all()
    np.testing.assert_allclose(table["fold_errors_1"], [0.9, 0.5, 0.4])
    np.testing.assert_allclose(table["fold_errors_2"], [0.8, 0.6, 0.5])
    np.testing.assert_allclose(table["threshold"], [0.6, 0.6, 0.6])
    assert table["is_selected"].tolist() == [False, True, False]


def test_selection_results_to_table_rejects_bad_results() -> None:
    """Raise instead of silently dropping entries that do not fit the table."""

    with pytest.raises(KeyError, match="selected_n_components"):
        selection_results_to_table({"candidate_n_components": np.array([1])}, SUBJ_ID)
    with pytest.raises(ValueError, match="bad"):
        selection_results_to_table(
            {
                "selected_n_components": 1,
                "candidate_n_components": np.array([1, 2]),
                "bad": np.zeros(3),
            },
            SUBJ_ID,
        )


def test_choose_minimal_pcs_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default to unique images, pass them unchanged, and save the CSV."""

    responses = make_responses()
    write_store(tmp_path, responses)
    patch_shared_flags(monkeypatch, [7])
    received: dict[str, Any] = {}

    def fake_selection(data: np.ndarray, max_components: int) -> dict[str, Any]:
        """Record the call and return a fixed two-candidate result.

        Inputs:
            data: np.ndarray with shape (n_images, n_voxels, n_repeats).
            max_components: int, recorded keyword argument.

        Output:
            results: dict[str, Any] with a selected count of 1 and per-fold
                errors of shape (2, 2).
        """

        received["data"] = data
        received["max_components"] = max_components
        return {
            "selected_n_components": 1,
            "candidate_n_components": np.array([1, 2]),
            "fold_errors": np.array([[0.5, 0.7], [0.6, 0.8]]),
        }

    output_dir = tmp_path / "out"
    report_dir = tmp_path / "Nonfinite"
    # image_set is omitted on purpose: the default must exclude shared images.
    results = choose_minimal_pcs(
        SUBJ_ID,
        fake_selection,
        selection_kwargs={"max_components": 2},
        neural_data_dir=tmp_path,
        pkl_info_path=tmp_path / "unused.pkl",
        output_dir=output_dir,
        nonfinite_report_dir=report_dir,
    )

    # (6, 4, 3) -> (5, 4, 3)
    np.testing.assert_array_equal(received["data"], responses[[0, 1, 3, 4, 5]])
    assert received["max_components"] == 2
    assert results["selected_n_components"] == 1
    assert results["image_set"] == "unique"
    np.testing.assert_array_equal(results["image_ids"], [10, 3, 0, 5, 8])
    assert results["csv_path"] == output_dir / f"{SUBJ_ID}_unique_fake_selection.csv"
    assert list(results["table"].columns[:3]) == [
        "subj_id",
        "image_set",
        "candidate_n_components",
    ]
    assert (results["table"]["image_set"] == "unique").all()
    pd.testing.assert_frame_equal(pd.read_csv(results["csv_path"]), results["table"])
    assert results["table"]["is_selected"].tolist() == [True, False]
    assert not report_dir.exists()
