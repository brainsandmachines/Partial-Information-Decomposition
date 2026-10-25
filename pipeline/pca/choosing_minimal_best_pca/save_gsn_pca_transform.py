"""Fit and save a selected-image-set GSN PCA transform without projections."""

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA


repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.pca.choosing_minimal_best_pca.choosing_minimal_best_pca import (
    DEFAULT_NEURAL_DATA_DIR,
    DEFAULT_PKL_INFO_PATH,
    IMAGE_SETS,
    load_repeat_responses,
)
from pipeline.testing.gsn.gsn_pca import (
    estimate_gsn_covariances,
    pca_signal_noise_covariances,
)


# Edit these values before running this file. Set N_COMPONENTS to None to use
# best_n_components from SELECTION_CSV_PATH, or set it to an integer to override.
SUBJ_ID = "subj01"
SELECTION_CSV_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "choosing_minimal_best_pca"
    / "subj01_unique_select_gsn_pca_components.csv"
)
N_COMPONENTS: int | None = None
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "fitted_gsn_pca"
NEURAL_DATA_DIR = DEFAULT_NEURAL_DATA_DIR
PKL_INFO_PATH = DEFAULT_PKL_INFO_PATH


def selection_from_csv(
    selection_csv_path: str | Path,
    subj_id: str,
) -> tuple[int, str]:
    """Read one subject's component count and image set from a result CSV.

    Inputs:
        selection_csv_path: str or Path, CSV written by
            ``choosing_minimal_best_pca.py``. Every row must describe the same
            subject and image set.
        subj_id: str, expected subject identifier, for example ``"subj01"``.

    Output:
        selection: tuple[int, str] containing the single positive value in the
            CSV's ``best_n_components`` column and its image-set name.
    """

    csv_path = Path(selection_csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Selection CSV does not exist: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"Selection CSV contains no result rows: {csv_path}")
    required_columns = {"subj_id", "image_set", "best_n_components"}
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        raise ValueError(
            f"Selection CSV is missing columns: {sorted(missing_columns)}."
        )
    if {row["subj_id"] for row in rows} != {subj_id}:
        raise ValueError(
            f"Selection CSV must contain only subj_id={subj_id!r}: {csv_path}"
        )
    image_sets = {row["image_set"] for row in rows}
    if len(image_sets) != 1:
        raise ValueError(
            "Selection CSV must contain exactly one image_set value."
        )
    image_set = image_sets.pop()
    if image_set not in IMAGE_SETS:
        raise ValueError(f"image_set must be one of {IMAGE_SETS}, got {image_set!r}.")

    try:
        values = {float(row["best_n_components"]) for row in rows}
    except (TypeError, ValueError) as error:
        raise ValueError("best_n_components must contain numeric values.") from error
    if len(values) != 1:
        raise ValueError(
            "Selection CSV must contain exactly one best_n_components value."
        )
    value = values.pop()
    if not np.isfinite(value) or not value.is_integer() or value < 1:
        raise ValueError("best_n_components must be a positive integer.")
    return int(value), image_set


def save_gsn_pca_transform(
    subj_id: str,
    selection_csv_path: str | Path,
    output_dir: str | Path,
    n_components: int | None = None,
    neural_data_dir: str | Path = DEFAULT_NEURAL_DATA_DIR,
    pkl_info_path: str | Path = DEFAULT_PKL_INFO_PATH,
) -> Path:
    """Fit GSN on the CSV's image set and save only its PCA transform.

    The saved joblib artifact is a scikit-learn ``PCA`` object whose retained
    components are the leading eigenvectors of GSN's estimated signal
    covariance. It contains the fitted response mean, projection parameters,
    source image set, and selection-CSV path, but no neural observations,
    covariance matrices, or projected scores. The existing pipeline can
    therefore apply it with ``transform``.

    Inputs:
        subj_id: str, subject identifier, for example ``"subj01"``.
        selection_csv_path: str or Path, component-selection CSV. Its
            ``best_n_components`` value is used when n_components is None.
        output_dir: str or Path, directory in which to save the joblib model.
        n_components: int or None, explicit retained component count. None
            loads the count from selection_csv_path.
        neural_data_dir: str or Path, directory containing the subject's
            ``*_betas_img_vox_rep.zarr`` store.
        pkl_info_path: str or Path, NSD stimulus-information pickle containing
            the shared1000 flags used when the CSV's image_set is ``unique``.

    Output:
        model_path: Path, saved ``*_gsn_signal_pca_*_components.joblib`` file.
            Loading it with joblib returns an object exposing ``transform``
            from shape (n_samples, n_voxels) to
            (n_samples, n_components).
    """

    csv_n_components, image_set = selection_from_csv(selection_csv_path, subj_id)
    chosen_n_components = csv_n_components if n_components is None else n_components
    if isinstance(chosen_n_components, bool) or not isinstance(
        chosen_n_components, (int, np.integer)
    ):
        raise TypeError("n_components must be an integer or None.")
    chosen_n_components = int(chosen_n_components)

    data = load_repeat_responses(
        subj_id,
        image_set=image_set,
        pkl_info_path=pkl_info_path,
        neural_data_dir=neural_data_dir,
    )
    responses = np.asarray(data["responses"], dtype=float)
    if responses.ndim != 3:
        raise ValueError(
            "responses must have shape (n_images, n_voxels, n_repeats)."
        )
    if not np.all(np.isfinite(responses)):
        raise ValueError("responses must be finite.")
    n_voxels = responses.shape[1]
    if not 1 <= chosen_n_components <= n_voxels:
        raise ValueError(
            f"n_components must be between 1 and {n_voxels}, got "
            f"{chosen_n_components}."
        )

    gsn_results = estimate_gsn_covariances(responses)
    pca_results = pca_signal_noise_covariances(
        gsn_results["cSb"],
        gsn_results["cNb"],
    )

    transform = PCA(n_components=chosen_n_components, whiten=False)
    # (n_voxels, n_voxels) -> (n_components, n_voxels)
    transform.components_ = pca_results["signal_eigenvectors"][
        :, :chosen_n_components
    ].T.copy()
    # (n_images, n_voxels, n_repeats) -> (n_voxels,)
    transform.mean_ = responses.mean(axis=(0, 2))
    # (n_voxels,) -> (n_components,)
    transform.explained_variance_ = pca_results["signal_eigenvalues"][
        :chosen_n_components
    ].copy()
    transform.n_components_ = chosen_n_components
    transform.n_features_in_ = n_voxels
    transform.image_set_ = image_set
    transform.selection_csv_path_ = str(Path(selection_csv_path).resolve())

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / (
        f"{subj_id}_{image_set}_gsn_signal_pca_"
        f"{chosen_n_components}_components.joblib"
    )
    joblib.dump(transform, model_path)
    count_source = "CSV best_n_components" if n_components is None else "override"
    print(
        f"Saved {chosen_n_components}-component GSN PCA transform for "
        f"image_set={image_set!r} ({count_source}) to {model_path}."
    )
    return model_path


if __name__ == "__main__":
    save_gsn_pca_transform(
        SUBJ_ID,
        SELECTION_CSV_PATH,
        OUTPUT_DIR,
        n_components=N_COMPONENTS,
        neural_data_dir=NEURAL_DATA_DIR,
        pkl_info_path=PKL_INFO_PATH,
    )
