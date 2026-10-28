"""Tests for the OTC loaders and the toy generator (loader tests ported from test_choosing_minimal_best_pca.py)."""

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

zarr = pytest.importorskip("zarr")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pipeline.rank_selection import data as choosing_minimal_best_pca
from pipeline.rank_selection.data import (

    load_otc_responses as load_repeat_responses,
    raise_on_nonfinite_images,

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
        load_repeat_responses(SUBJ_ID, image_set="bogus", neural_data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_repeat_responses("subj00", image_set="all", neural_data_dir=tmp_path)




def write_per_stimulus_store(neural_data_dir: Path, betas: np.ndarray, stim_ids: np.ndarray) -> None:
    """Write a ``{SUBJ_ID}_OTC_betas_per_stimulus.zarr`` store.

    Inputs:
        neural_data_dir: Path, directory in which the store is created.
        betas: np.ndarray with shape (n_images, n_voxels).
        stim_ids: np.ndarray with shape (n_images,), NSD image IDs.

    Output:
        None. The Zarr store is written to disk.
    """

    dataset = xr.Dataset(
        {"betas": (("image", "voxel"), betas), "stim_id": (("image",), stim_ids)}
    )
    dataset.to_zarr(neural_data_dir / f"{SUBJ_ID}_OTC_betas_per_stimulus.zarr", mode="w")


def test_per_stimulus_store_returns_one_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the per-stimulus store as a single repeat and apply the same shared mask."""

    # (24,) -> (6, 4)
    betas = np.arange(24.0).reshape(6, 4)
    write_per_stimulus_store(tmp_path, betas, IMAGE_IDS)
    patch_shared_flags(monkeypatch, [7, 0])

    shared = load_repeat_responses(
        SUBJ_ID, image_set="shared", store="per_stimulus",
        pkl_info_path=tmp_path / "unused.pkl", neural_data_dir=tmp_path,
    )
    unique = load_repeat_responses(
        SUBJ_ID, image_set="unique", store="per_stimulus",
        pkl_info_path=tmp_path / "unused.pkl", neural_data_dir=tmp_path,
    )

    np.testing.assert_array_equal(shared["image_ids"], [7, 0])
    # (6, 4) -> (2, 4, 1)
    np.testing.assert_array_equal(shared["responses"], betas[[2, 3]][:, :, None])
    np.testing.assert_array_equal(unique["image_ids"], [10, 3, 5, 8])
    with pytest.raises(ValueError, match="store"):
        load_repeat_responses(SUBJ_ID, image_set="all", store="bogus", neural_data_dir=tmp_path)


def test_load_otc_pair_aligns_common_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep only images both subjects saw, in the same row order for both."""

    first = np.random.default_rng(0).normal(size=(20, 3, 2))
    second = np.random.default_rng(1).normal(size=(20, 5, 2))
    first_ids = np.arange(20)
    second_ids = np.arange(20)[::-1] + 4  # 4..23, reversed; common ids 4..19

    for subject, responses, ids in (("subjA", first, first_ids), ("subjB", second, second_ids)):
        betas = xr.DataArray(
            responses, dims=("image", "voxel", "repeat"),
            coords={"image": ids, "voxel": np.arange(responses.shape[1]), "repeat": [0, 1]},
        )
        betas.to_dataset(name="betas_by_image_voxel_repeat").to_zarr(
            tmp_path / f"{subject}_betas_img_vox_rep.zarr", mode="w"
        )

    pair = choosing_minimal_best_pca.load_otc_pair(
        "subjA", "subjB", image_set="all", store="repeat", neural_data_dir=tmp_path
    )

    np.testing.assert_array_equal(pair.image_ids, np.arange(4, 20))
    np.testing.assert_array_equal(pair.responses, first[4:20])
    # Row i of the target must be the target's response to image id 4 + i.
    target_rows = [int(np.flatnonzero(second_ids == image_id)[0]) for image_id in range(4, 20)]
    np.testing.assert_array_equal(pair.target_responses, second[target_rows])
    assert pair.name == "subjA_to_subjB"


@pytest.mark.parametrize("noise_type", ["isotropic", "unequal_variance", "correlated"])
@pytest.mark.parametrize("private_rank", [0, 3])
def test_make_toy_has_the_stated_signal_rank(noise_type: str, private_rank: int) -> None:
    """Check the generated signal against its stated rank and covariance.

    The check uses only the generator's outputs: the noiseless signal's sample
    covariance must equal the stated covariance and have rank true_rank +
    private_rank; the two subjects' signals must share exactly true_rank
    dimensions; responses minus signal must have the stated noise structure.
    """

    toy = choosing_minimal_best_pca.make_toy(
        n_images=400, n_voxels=25, true_rank=4, private_rank=private_rank,
        n_repeats=3, noise_type=noise_type, noise_rank=5, n_subjects=2, seed=3,
    )
    signal, covariance = toy.info["signal"], toy.info["signal_covariance"]
    assert toy.responses.shape == (400, 25, 3)
    assert toy.target_responses.shape == (400, 25, 3)
    np.testing.assert_allclose(np.cov(signal, rowvar=False), covariance, atol=1e-10)
    # (25, 25) -> (25,)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    assert int(np.sum(eigenvalues > 1e-9 * eigenvalues[0])) == 4 + private_rank
    # (25, 400) @ (400, 25) -> (25, 25): only the shared dimensions covary.
    cross = (signal - signal.mean(0)).T @ (toy.info["target_signal"] - toy.info["target_signal"].mean(0))
    singular_values = np.linalg.svd(cross, compute_uv=False)
    assert int(np.sum(singular_values > 1e-9 * singular_values[0])) == 4
    assert toy.true_rank == 4

    noise = toy.responses - signal[:, :, None]  # (400, 25, 3)
    noise_covariance = toy.info["noise_covariance"]
    off_diagonal = noise_covariance - np.diag(np.diag(noise_covariance))
    assert np.allclose(off_diagonal, 0) == (noise_type != "correlated")
    # (400, 25, 3) -> (1200, 25): sample noise covariance near the stated one.
    sample = np.cov(noise.transpose(0, 2, 1).reshape(-1, 25), rowvar=False)
    assert np.abs(sample - noise_covariance).max() < 0.25 * np.abs(noise_covariance).max()


def test_make_toy_single_subject_truth_includes_private_signal() -> None:
    """A lone subject's stimulus-driven rank counts its private signal too."""

    toy = choosing_minimal_best_pca.make_toy(
        n_images=200, n_voxels=20, true_rank=3, private_rank=2, n_subjects=1, seed=0
    )
    assert toy.true_rank == 5
    assert toy.target_responses is None
    with pytest.raises(ValueError, match="second subject"):
        toy.averaged_target()
