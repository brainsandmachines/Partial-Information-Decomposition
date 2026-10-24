"""Choose the minimal number of target PCs from repeated NSD OTC responses."""

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr
import yaml

repo_root = Path(__file__).resolve().parents[3]
external_root = repo_root / "external"
for path in (repo_root, external_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from external.mayas_project.features_and_encoding.feat_ext_and_encoding import get_stim_info
from pipeline.testing.gsn.gsn_pca import select_gsn_pca_components
from pipeline.utils import resolve_pipeline_function


DEFAULT_NEURAL_DATA_DIR = Path(
    "/groups/golan_neurogroup/bml_group/datasets/nsddata/otc_betas/"
    "otc_betas_img_vox_rep/"
)
DEFAULT_PKL_INFO_PATH = Path(
    "/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata/"
    "experiments/nsd/nsd_stim_info_merged.pkl"
)
DEFAULT_OUTPUT_DIR = repo_root / "pipeline" / "pca" / "results" / "choosing_minimal_best_pca"
# Reports of images with non-finite responses, written before the run stops.
DEFAULT_NONFINITE_DIR = Path(__file__).resolve().parent / "Nonfinite"
# Number of non-finite images listed in the error message; the CSV lists all.
MAX_LISTED_NONFINITE_IMAGES = 20

# Selection functions that the YAML config may name.
SELECTION_FUNCTIONS = {
    "gsn_independent_repeat": select_gsn_pca_components,
}
# "unique": every image except shared1000 (default); "all": every OTC image.
# Shared images are used only when "all" is requested explicitly.
IMAGE_SETS = ("unique", "all")


def raise_on_nonfinite_images(
    responses: np.ndarray,
    image_ids: np.ndarray,
    repeat_ids: np.ndarray,
    subj_id: str,
    image_set: str,
    report_dir: str | Path = DEFAULT_NONFINITE_DIR,
) -> None:
    """Report images with non-finite responses and stop with an error.

    Inputs:
        responses: np.ndarray with shape (n_images, n_voxels, n_repeats).
        image_ids: np.ndarray with shape (n_images,), NSD image IDs of the rows.
        repeat_ids: np.ndarray with shape (n_repeats,), repeat labels of the
            last axis.
        subj_id: str, subject identifier used in the report and error.
        image_set: str, image set that was loaded, used in the report name.
        report_dir: str or Path, directory for the non-finite report.

    Output:
        None. When every value is finite nothing happens. Otherwise a CSV
        ``{subj_id}_{image_set}_nonfinite.csv`` with one row per
        (image, repeat) pair containing a non-finite value is written to
        report_dir, and a ValueError listing those images and repeats is raised.
    """

    # (n_images, n_voxels, n_repeats) -> (n_images, n_repeats)
    n_nonfinite_voxels = (~np.isfinite(responses)).sum(axis=1)
    if not n_nonfinite_voxels.any():
        return

    # (n_images, n_repeats) -> (n_bad_pairs,), (n_bad_pairs,)
    image_rows, repeat_columns = np.nonzero(n_nonfinite_voxels)
    report = pd.DataFrame(
        {
            "subj_id": subj_id,
            "image_set": image_set,
            "image_id": image_ids[image_rows],
            "repeat": repeat_ids[repeat_columns],
            "n_nonfinite_voxels": n_nonfinite_voxels[image_rows, repeat_columns],
            "n_voxels": responses.shape[1],
        }
    )
    report_path = Path(report_dir) / f"{subj_id}_{image_set}_nonfinite.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    # (n_images, n_repeats) -> (n_bad_images,)
    bad_rows = np.flatnonzero(n_nonfinite_voxels.any(axis=1))
    listed = [
        f"  image {image_ids[row].item()}: repeats {repeat_ids[n_nonfinite_voxels[row] > 0].tolist()}"
        for row in bad_rows[:MAX_LISTED_NONFINITE_IMAGES]
    ]
    if bad_rows.shape[0] > MAX_LISTED_NONFINITE_IMAGES:
        listed.append(f"  ... and {bad_rows.shape[0] - MAX_LISTED_NONFINITE_IMAGES} more")
    raise ValueError(
        f"{subj_id} ({image_set} images): {bad_rows.shape[0]} images have non-finite "
        "responses:\n" + "\n".join(listed) + f"\nFull report saved to {report_path}."
    )


def load_repeat_responses(
    subj_id: str,
    image_set: str = "unique",
    pkl_info_path: str | Path = DEFAULT_PKL_INFO_PATH,
    neural_data_dir: str | Path = DEFAULT_NEURAL_DATA_DIR,
    nonfinite_report_dir: str | Path = DEFAULT_NONFINITE_DIR,
) -> dict[str, Any]:
    """Load one subject's repeated OTC responses for the chosen image set.

    ``image_set="all"`` keeps every image in the store. ``image_set="unique"``
    removes the shared1000 images, which are the held-out evaluation set. If
    any loaded image has a non-finite value (e.g. a missing repeat stored as
    NaN), the affected images and repeats are reported and the run stops; no
    image is dropped silently.

    Inputs:
        subj_id: str, subject identifier, e.g. ``"subj01"``. The store
            ``{subj_id}_betas_img_vox_rep.zarr`` is read from neural_data_dir.
        image_set: str, ``"all"`` for the full OTC data set or ``"unique"``
            for every image except the shared1000 images.
        pkl_info_path: str or Path, NSD stimulus-information pickle that holds
            the shared1000 flags. It is read only when image_set is
            ``"unique"``.
        neural_data_dir: str or Path, directory with the per-subject
            ``*_betas_img_vox_rep.zarr`` stores.
        nonfinite_report_dir: str or Path, directory for the report written
            when non-finite responses are found.

    Output:
        data: dict[str, Any] with ``responses`` of shape
            (n_set_images, n_voxels, n_repeats), ``image_ids`` of shape
            (n_set_images,), and the int count ``n_outside_set_removed``.
            Raises ValueError when any response is non-finite.
    """

    if image_set not in IMAGE_SETS:
        raise ValueError(f"image_set must be one of {IMAGE_SETS}, got {image_set!r}.")
    store_path = Path(neural_data_dir) / f"{subj_id}_betas_img_vox_rep.zarr"
    if not store_path.exists():
        raise FileNotFoundError(f"Neural data store does not exist: {store_path}")
    dataset = xr.open_zarr(store_path)
    if "betas_by_image_voxel_repeat" not in dataset:
        raise KeyError(f"{store_path} has no 'betas_by_image_voxel_repeat' variable.")
    betas = dataset["betas_by_image_voxel_repeat"].transpose("image", "voxel", "repeat")
    image_ids = np.asarray(betas["image"].values)
    repeat_ids = np.asarray(betas["repeat"].values)

    if image_set == "unique":
        # Same image-ID convention as prepare_subject_context in mayas_project.
        _, shared1000_all = get_stim_info(Path(pkl_info_path))
        # (n_nsd_images,) -> (n_images,)
        shared_mask = np.asarray(shared1000_all[image_ids], dtype=bool)
        if not shared_mask.any():
            raise ValueError(
                f"No shared1000 images found for {subj_id}; check the image-ID convention."
            )
        set_mask = ~shared_mask
    else:
        # (n_images,) -> (n_images,)
        set_mask = np.ones(image_ids.shape[0], dtype=bool)

    # (n_images, n_voxels, n_repeats) -> (n_set_images, n_voxels, n_repeats)
    set_responses = np.asarray(betas.isel(image=set_mask).values, dtype=float)
    # (n_images,) -> (n_set_images,)
    set_image_ids = image_ids[set_mask]
    raise_on_nonfinite_images(
        set_responses,
        set_image_ids,
        repeat_ids,
        subj_id,
        image_set,
        report_dir=nonfinite_report_dir,
    )

    n_outside_set_removed = int((~set_mask).sum())
    print(
        f"{subj_id} ({image_set} images): {image_ids.shape[0]} images, removed "
        f"{n_outside_set_removed} outside the set, kept {set_image_ids.shape[0]}."
    )

    return {
        "responses": set_responses,
        "image_ids": set_image_ids,
        "n_outside_set_removed": n_outside_set_removed,
    }


def selection_results_to_table(results: dict[str, Any], subj_id: str) -> pd.DataFrame:
    """Convert a selection function's outputs to one row per candidate PC count.

    Every entry is written by its shape: scalars become constant columns,
    arrays of shape (n_candidates,) become one column, and arrays of shape
    (n_rows, n_candidates) become one column per row named ``{key}_{i}``
    with i starting at 1. Any other entry raises, so nothing is dropped.

    Inputs:
        results: dict[str, Any], selection-function output. It must contain
            ``selected_n_components`` (int) and ``candidate_n_components``
            (array of shape (n_candidates,)).
        subj_id: str, subject identifier written to every row.

    Output:
        table: pd.DataFrame with n_candidates rows, the ``subj_id`` and
            ``candidate_n_components`` columns, one or more columns per result
            entry, and a Boolean ``is_selected`` column.
    """

    for required_key in ("selected_n_components", "candidate_n_components"):
        if required_key not in results:
            raise KeyError(f"Selection results must contain '{required_key}'.")
    candidates = np.asarray(results["candidate_n_components"])
    if candidates.ndim != 1:
        raise ValueError("candidate_n_components must be one-dimensional.")
    n_candidates = candidates.shape[0]

    columns: dict[str, Any] = {
        "subj_id": subj_id,
        "candidate_n_components": candidates,
    }
    for key, value in results.items():
        if key == "candidate_n_components":
            continue
        array = np.asarray(value)
        if array.ndim == 0 and array.dtype != object:
            columns[key] = array.item()
        elif array.shape == (n_candidates,):
            columns[key] = array
        elif array.ndim == 2 and array.shape[1] == n_candidates:
            # (n_rows, n_candidates) -> n_rows columns of shape (n_candidates,)
            for row_index, row in enumerate(array, start=1):
                columns[f"{key}_{row_index}"] = row
        else:
            raise ValueError(
                f"Cannot write result '{key}' with shape {array.shape} to a table "
                f"with {n_candidates} candidate rows."
            )
    columns["is_selected"] = candidates == results["selected_n_components"]

    return pd.DataFrame(columns)


def choose_minimal_pcs(
    subj_id: str,
    selection_function: Callable[..., dict[str, Any]],
    image_set: str = "unique",
    selection_kwargs: dict[str, Any] | None = None,
    neural_data_dir: str | Path = DEFAULT_NEURAL_DATA_DIR,
    pkl_info_path: str | Path = DEFAULT_PKL_INFO_PATH,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    nonfinite_report_dir: str | Path = DEFAULT_NONFINITE_DIR,
) -> dict[str, Any]:
    """Load a subject's repeated responses and choose the minimal PC count.

    The run stops with a ValueError, after writing a report to
    nonfinite_report_dir, if any loaded image has a non-finite response.

    Inputs:
        subj_id: str, subject identifier, e.g. ``"subj01"``.
        selection_function: callable taking responses of shape
            (n_images, n_voxels, n_repeats) plus keyword arguments and
            returning a dict with ``selected_n_components`` and
            ``candidate_n_components``, e.g. ``select_gsn_pca_components``.
        image_set: str, ``"all"`` for the full OTC data set or ``"unique"``
            for every image except the shared1000 images.
        selection_kwargs: dict or None, keyword arguments passed to
            selection_function.
        neural_data_dir: str or Path, directory with the per-subject
            ``*_betas_img_vox_rep.zarr`` stores.
        pkl_info_path: str or Path, NSD stimulus-information pickle.
        output_dir: str, Path, or None, directory for the CSV of selection
            outputs. When None, nothing is written.
        nonfinite_report_dir: str or Path, directory for the report written
            when non-finite responses are found.

    Output:
        results: dict[str, Any] with ``selected_n_components`` (int),
            ``image_set`` (str), ``selection_results`` (the selection
            function's dict), ``table`` (pd.DataFrame), ``csv_path`` (Path or
            None), and ``image_ids`` of shape (n_set_images,).
    """

    selection_kwargs = {} if selection_kwargs is None else dict(selection_kwargs)

    data = load_repeat_responses(
        subj_id,
        image_set=image_set,
        pkl_info_path=pkl_info_path,
        neural_data_dir=neural_data_dir,
        nonfinite_report_dir=nonfinite_report_dir,
    )
    selection_results = selection_function(data["responses"], **selection_kwargs)
    table = selection_results_to_table(selection_results, subj_id)
    table.insert(1, "image_set", image_set)

    csv_path = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / f"{subj_id}_{image_set}_{selection_function.__name__}.csv"
        table.to_csv(csv_path, index=False)
        print(f"Saved selection outputs to {csv_path}.")

    return {
        "selected_n_components": int(selection_results["selected_n_components"]),
        "image_set": image_set,
        "selection_results": selection_results,
        "table": table,
        "csv_path": csv_path,
        "image_ids": data["image_ids"],
    }


if __name__ == "__main__":
    with open(Path(__file__).with_name("choosing_minimal_best_pca.yaml"), "r") as config_file:
        config = yaml.safe_load(config_file)

    selection_function = resolve_pipeline_function(
        config["functions"],
        SELECTION_FUNCTIONS,
        "selection_function",
        required=True,
    )
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    for subj_id in config["subjects"]:
        subject_results = choose_minimal_pcs(
            subj_id,
            selection_function,
            image_set=config["image_set"],
            selection_kwargs=config["selection_kwargs"],
            neural_data_dir=config["neural_data_dir"],
            pkl_info_path=config["pkl_info_path"],
            output_dir=output_dir,
        )
        print(
            f"{subj_id} ({subject_results['image_set']} images): "
            f"selected {subject_results['selected_n_components']} PCs."
        )
