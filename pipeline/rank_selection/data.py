"""Shared inputs for every rank-selection method: OTC loaders and one toy generator.

Every method receives a ``RankData``: responses shaped (images, voxels, repeats)
for one subject, optionally paired with a second subject's responses to the
same images. Toy datasets also carry their known ground truth.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

repo_root = Path(__file__).resolve().parents[2]
external_root = repo_root / "external"
for path in (repo_root, external_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from external.mayas_project.features_and_encoding.feat_ext_and_encoding import get_stim_info

OTC_ROOT = Path("/groups/golan_neurogroup/bml_group/datasets/nsddata/otc_betas")
# The two OTC stores do not agree (per-voxel correlation ~0.48 for subj01), so
# every loader requires the store to be named explicitly.
STORE_DIRS = {
    "repeat": OTC_ROOT / "otc_betas_img_vox_rep",
    "per_stimulus": OTC_ROOT / "otc_betas_per_stim",
}
DEFAULT_PKL_INFO_PATH = Path(
    "/groups/golan_neurogroup/bml_group/datasets/nsddata/nsddata/"
    "experiments/nsd/nsd_stim_info_merged.pkl"
)
DEFAULT_NONFINITE_DIR = Path(__file__).resolve().parent / "results" / "nonfinite_reports"
# "unique": every image except shared1000; "shared": only shared1000 (the PID
# evaluation images); "all": every image.
IMAGE_SETS = ("unique", "shared", "all")
NOISE_TYPES = ("isotropic", "unequal_variance", "correlated")
MAX_LISTED_NONFINITE_IMAGES = 20


@dataclass
class RankData:
    """Responses of one subject, optionally paired with a second subject.

    Fields:
        name: str, dataset label used in result folders.
        responses: np.ndarray (n_images, n_voxels, n_repeats).
        image_ids: np.ndarray (n_images,), identifiers shared by both subjects.
        subject: str or None.
        target_responses: np.ndarray (n_images, n_target_voxels,
            n_target_repeats) or None; the second subject for pair methods.
        target_subject: str or None.
        true_rank: int or None. Toy single-subject data: rank of the subject's
            stimulus-driven signal (shared + private). Toy pair data: shared rank.
        info: dict[str, Any], provenance and toy ground truth (``signal``,
            ``signal_covariance``, ``shared_rank``, ``private_rank``, ...).
    """

    name: str
    responses: np.ndarray
    image_ids: np.ndarray
    subject: str | None = None
    target_responses: np.ndarray | None = None
    target_subject: str | None = None
    true_rank: int | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def averaged(self) -> np.ndarray:
        """Average the responses over repeats.

        Input: none.
        Output: np.ndarray (n_images, n_voxels).
        """
        return self.responses.mean(axis=2)  # (n_images, n_voxels, n_repeats) -> (n_images, n_voxels)

    def averaged_target(self) -> np.ndarray:
        """Average the second subject's responses over repeats.

        Input: none.
        Output: np.ndarray (n_images, n_target_voxels). Raises ValueError when
            the data have no second subject.
        """
        if self.target_responses is None:
            raise ValueError(f"Dataset {self.name!r} has no second subject.")
        return self.target_responses.mean(axis=2)  # (n, v_t, r_t) -> (n, v_t)


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
        repeat_ids: np.ndarray with shape (n_repeats,), repeat labels.
        subj_id: str, subject identifier used in the report and error.
        image_set: str, image set that was loaded, used in the report name.
        report_dir: str or Path, directory for the non-finite report.

    Output:
        None. When every value is finite nothing happens. Otherwise a CSV
        ``{subj_id}_{image_set}_nonfinite.csv`` with one row per (image, repeat)
        pair containing a non-finite value is written to report_dir, and a
        ValueError listing those images and repeats is raised.
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


def load_otc_responses(
    subj_id: str,
    image_set: str = "unique",
    store: str = "repeat",
    neural_data_dir: str | Path | None = None,
    pkl_info_path: str | Path = DEFAULT_PKL_INFO_PATH,
    nonfinite_report_dir: str | Path = DEFAULT_NONFINITE_DIR,
) -> dict[str, Any]:
    """Load one subject's OTC responses for an image set from a named store.

    ``store="repeat"`` reads ``{subj}_betas_img_vox_rep.zarr`` (variable
    ``betas_by_image_voxel_repeat``). ``store="per_stimulus"`` reads
    ``{subj}_OTC_betas_per_stimulus.zarr`` (variable ``betas``) and returns a
    single repeat axis. Shared1000 flags use the NSD convention
    ``shared1000[image_id]`` for both stores. Non-finite responses in the
    selected images stop the run with a report; no image is dropped silently.

    Inputs:
        subj_id: str, subject identifier, e.g. ``"subj01"``.
        image_set: str in IMAGE_SETS.
        store: str, ``"repeat"`` or ``"per_stimulus"``.
        neural_data_dir: str, Path, or None; None uses STORE_DIRS[store].
        pkl_info_path: str or Path, NSD stimulus-information pickle; read only
            when image_set is not ``"all"``.
        nonfinite_report_dir: str or Path for the non-finite report.

    Output:
        data: dict[str, Any] with ``responses`` (n_set_images, n_voxels,
            n_repeats), ``image_ids`` (n_set_images,), and the int
            ``n_outside_set_removed``.
    """
    if image_set not in IMAGE_SETS:
        raise ValueError(f"image_set must be one of {IMAGE_SETS}, got {image_set!r}.")
    if store not in STORE_DIRS:
        raise ValueError(f"store must be one of {tuple(STORE_DIRS)}, got {store!r}.")
    directory = Path(STORE_DIRS[store] if neural_data_dir is None else neural_data_dir)

    if store == "repeat":
        store_path = directory / f"{subj_id}_betas_img_vox_rep.zarr"
        if not store_path.exists():
            raise FileNotFoundError(f"Neural data store does not exist: {store_path}")
        dataset = xr.open_zarr(store_path)
        if "betas_by_image_voxel_repeat" not in dataset:
            raise KeyError(f"{store_path} has no 'betas_by_image_voxel_repeat' variable.")
        betas = dataset["betas_by_image_voxel_repeat"].transpose("image", "voxel", "repeat")
        image_ids = np.asarray(betas["image"].values)
        repeat_ids = np.asarray(betas["repeat"].values)
    else:
        store_path = directory / f"{subj_id}_OTC_betas_per_stimulus.zarr"
        if not store_path.exists():
            raise FileNotFoundError(f"Neural data store does not exist: {store_path}")
        dataset = xr.open_zarr(store_path)
        betas = dataset["betas"]
        image_ids = np.asarray(dataset["stim_id"].values)
        repeat_ids = np.array([0])

    if image_set == "all":
        # (n_images,) -> (n_images,)
        set_mask = np.ones(image_ids.shape[0], dtype=bool)
    else:
        _, shared1000_all = get_stim_info(Path(pkl_info_path))
        # (n_nsd_images,) -> (n_images,)
        shared_mask = np.asarray(shared1000_all[image_ids], dtype=bool)
        if not shared_mask.any():
            raise ValueError(
                f"No shared1000 images found for {subj_id}; check the image-ID convention."
            )
        set_mask = shared_mask if image_set == "shared" else ~shared_mask

    if store == "repeat":
        # (n_images, n_voxels, n_repeats) -> (n_set_images, n_voxels, n_repeats)
        set_responses = np.asarray(betas.isel(image=set_mask).values, dtype=float)
    else:
        # (n_images, n_voxels) -> (n_set_images, n_voxels, 1)
        set_responses = np.asarray(betas.values[set_mask], dtype=float)[:, :, None]
    set_image_ids = image_ids[set_mask]  # (n_images,) -> (n_set_images,)
    raise_on_nonfinite_images(
        set_responses, set_image_ids, repeat_ids, subj_id, image_set,
        report_dir=nonfinite_report_dir,
    )
    n_outside_set_removed = int((~set_mask).sum())
    print(
        f"{subj_id} ({image_set} images, {store} store): {image_ids.shape[0]} images, "
        f"removed {n_outside_set_removed} outside the set, kept {set_image_ids.shape[0]}."
    )
    return {
        "responses": set_responses,
        "image_ids": set_image_ids,
        "n_outside_set_removed": n_outside_set_removed,
    }


def load_otc(
    subject: str,
    image_set: str = "unique",
    store: str = "repeat",
    neural_data_dir: str | Path | None = None,
    pkl_info_path: str | Path = DEFAULT_PKL_INFO_PATH,
) -> RankData:
    """Load one subject's OTC responses as RankData.

    Inputs: subject, image_set, store, neural_data_dir, pkl_info_path as in
        ``load_otc_responses``.
    Output: RankData named ``{subject}`` with responses (n_images, n_voxels,
        n_repeats) and no ground truth.
    """
    loaded = load_otc_responses(subject, image_set, store, neural_data_dir, pkl_info_path)
    return RankData(
        name=subject,
        responses=loaded["responses"],
        image_ids=loaded["image_ids"],
        subject=subject,
        info={"source": "otc", "image_set": image_set, "store": store},
    )


def load_otc_pair(
    source_subject: str,
    target_subject: str,
    image_set: str = "shared",
    store: str = "repeat",
    neural_data_dir: str | Path | None = None,
    pkl_info_path: str | Path = DEFAULT_PKL_INFO_PATH,
) -> RankData:
    """Load two subjects' responses to their common images, aligned by image ID.

    Inputs: source_subject and target_subject (str); image_set, store,
        neural_data_dir, pkl_info_path as in ``load_otc_responses``. Only images
        seen by both subjects are kept, i.e. the shared images in NSD.
    Output: RankData named ``{source}_to_{target}`` whose ``responses`` belong
        to the source and ``target_responses`` to the target, rows in the same
        image order.
    """
    source = load_otc_responses(source_subject, image_set, store, neural_data_dir, pkl_info_path)
    target = load_otc_responses(target_subject, image_set, store, neural_data_dir, pkl_info_path)
    # (n_source,), (n_target,) -> (n_common,)
    image_ids, source_rows, target_rows = np.intersect1d(
        source["image_ids"], target["image_ids"], return_indices=True
    )
    if image_ids.shape[0] < 10:
        raise ValueError(
            f"Only {image_ids.shape[0]} images are common to {source_subject} and "
            f"{target_subject}; use image_set='shared'."
        )
    print(f"{source_subject} and {target_subject}: {image_ids.shape[0]} common images.")
    return RankData(
        name=f"{source_subject}_to_{target_subject}",
        responses=source["responses"][source_rows],  # (n_source, v_s, r) -> (n_common, v_s, r)
        image_ids=image_ids,
        subject=source_subject,
        target_responses=target["responses"][target_rows],  # (n_target, v_t, r) -> (n_common, v_t, r)
        target_subject=target_subject,
        info={"source": "otc", "image_set": image_set, "store": store},
    )


def _orthonormal_loadings(
    rng: np.random.Generator,
    n_voxels: int,
    rank: int,
    loading_corr: float,
) -> np.ndarray:
    """Draw unit-norm voxel loadings with a fixed equicorrelation between columns.

    Inputs: rng (np.random.Generator); n_voxels and rank (int); loading_corr
        (float), the correlation between any two loading columns, in
        (-1 / (rank - 1), 1).
    Output: np.ndarray (n_voxels, rank) with column Gram matrix equal to the
        equicorrelation matrix (identity when loading_corr = 0).
    """
    if rank == 0:
        return np.zeros((n_voxels, 0))
    lower_bound = -1.0 / (rank - 1) if rank > 1 else -np.inf
    if not lower_bound < loading_corr < 1.0:
        raise ValueError(f"loading_corr={loading_corr} is invalid for rank={rank}.")
    # (n_voxels, rank) -> (n_voxels, rank) with orthonormal columns
    basis, _ = np.linalg.qr(rng.normal(size=(n_voxels, rank)))
    correlation = np.full((rank, rank), loading_corr)
    np.fill_diagonal(correlation, 1.0)
    # (n_voxels, rank) @ (rank, rank) -> (n_voxels, rank)
    return basis @ np.linalg.cholesky(correlation).T


def _orthogonal_scores(
    rng: np.random.Generator,
    n_images: int,
    variances: np.ndarray,
) -> np.ndarray:
    """Draw centered latent scores whose sample covariance is exactly diagonal.

    Inputs: rng; n_images (int); variances (np.ndarray (rank,)), the sample
        variance of each latent dimension.
    Output: np.ndarray (n_images, rank) with mean zero and sample covariance
        diag(variances), so the signal rank is exact in every sample.
    """
    rank = variances.shape[0]
    if rank == 0:
        return np.zeros((n_images, 0))
    latent = rng.normal(size=(n_images, rank))
    latent -= latent.mean(axis=0, keepdims=True)
    # (n_images, rank) -> (n_images, rank) with orthonormal columns
    latent_basis, _ = np.linalg.qr(latent)
    return latent_basis * np.sqrt((n_images - 1) * variances)


def _subject_noise(
    rng: np.random.Generator,
    n_images: int,
    n_voxels: int,
    n_repeats: int,
    noise_std: float,
    noise_type: str,
    noise_rank: int,
    noise_corr_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw measurement noise, independent across images and repeats.

    Inputs: rng; sizes (int); noise_std (float), the per-voxel scale;
        noise_type in NOISE_TYPES; noise_rank (int) and noise_corr_strength
        (float), the rank and scale of the voxel-correlated part used by
        ``"correlated"``.
    Output: noise np.ndarray (n_images, n_voxels, n_repeats) and its true
        voxel covariance np.ndarray (n_voxels, n_voxels).
    """
    standard = rng.normal(size=(n_images, n_voxels, n_repeats))
    if noise_type == "isotropic":
        return noise_std * standard, noise_std ** 2 * np.eye(n_voxels)
    if noise_type == "unequal_variance":
        voxel_scale = rng.uniform(0.3, 2.0, size=n_voxels)  # (n_voxels,)
        # (1, n_voxels, 1) * (n_images, n_voxels, n_repeats) -> same shape
        noise = noise_std * voxel_scale[None, :, None] * standard
        return noise, np.diag((noise_std * voxel_scale) ** 2)
    if noise_type == "correlated":
        mixing = noise_corr_strength * rng.normal(size=(noise_rank, n_voxels))  # (noise_rank, n_voxels)
        #Mixing determines how strongly each latent noise source contributes to the noise of each voxel.
        shared = rng.normal(size=(n_images, noise_rank, n_repeats))
        # (n_images, noise_rank, n_repeats) x (noise_rank, n_voxels) -> (n_images, n_voxels, n_repeats)
        correlated_part = np.einsum("nkr,kv->nvr", shared, mixing)
        covariance = noise_std ** 2 * (np.eye(n_voxels) + mixing.T @ mixing)
        return noise_std * (standard + correlated_part), covariance
    raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got {noise_type!r}.")


def make_toy(
    n_images: int = 500,
    n_voxels: int = 60,
    true_rank: int = 5,
    n_repeats: int = 3,
    noise_std: float = 1.0,
    noise_type: str = "isotropic",
    noise_rank: int = 10,
    noise_corr_strength: float = 0.5,
    private_rank: int = 0,
    n_subjects: int = 1,
    n_target_voxels: int | None = None,
    loading_corr: float = 0.0,
    signal_max_variance: float = 9.0,
    signal_min_variance: float = 1.0,
    seed: int = 0,
) -> RankData:
    """Generate repeated toy responses with a known stimulus-driven signal rank.

    Each subject's response to image i, repeat r is
    ``signal_i + noise_ir``. The signal is the same in every repeat and equals
    shared latent scores (rank ``true_rank``, common to both subjects) mapped
    by subject-specific loadings, plus a subject-private signal of rank
    ``private_rank``. Latent sample variances decrease geometrically from
    ``signal_max_variance`` to ``signal_min_variance``, so the signal covariance
    has exact, known rank. Noise is independent across images, repeats, and
    subjects, with the voxel structure given by ``noise_type``.

    Inputs:
        n_images, n_voxels, true_rank, n_repeats (int): sizes; true_rank is the
            shared signal rank.
        noise_std (float), noise_type (str in NOISE_TYPES), noise_rank (int),
            noise_corr_strength (float): measurement noise, see ``_subject_noise``.
        private_rank (int): extra signal rank of each subject alone.
        n_subjects (int): 1, or 2 for pair methods such as cross-subject PLS.
        n_target_voxels (int or None): second subject's voxel count (default
            n_voxels).
        loading_corr (float): equicorrelation of the shared loading columns.
        signal_max_variance, signal_min_variance (float): latent variances.
        seed (int): data seed.

    Output:
        RankData. ``true_rank`` is true_rank + private_rank for one subject and
        true_rank (the shared rank) for two. ``info`` holds the noiseless
        ``signal`` (n_images, n_voxels), the true ``signal_covariance`` and
        ``noise_covariance`` (n_voxels, n_voxels) of the first subject, the
        second subject's noiseless ``target_signal`` (n_images,
        n_target_voxels) or None, and the generator settings.
    """
    if n_subjects not in (1, 2):
        raise ValueError("n_subjects must be 1 or 2.")
    if not 1 <= true_rank + private_rank < n_voxels:
        raise ValueError("true_rank + private_rank must be between 1 and n_voxels - 1.")
    if n_images <= true_rank + n_subjects * private_rank + 1:
        raise ValueError("n_images must exceed the number of latent signal dimensions + 1.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    n_target_voxels = n_voxels if n_target_voxels is None else n_target_voxels
    settings = dict(
        n_images=n_images, n_voxels=n_voxels, true_rank=true_rank, n_repeats=n_repeats,
        noise_std=noise_std, noise_type=noise_type, noise_rank=noise_rank,
        noise_corr_strength=noise_corr_strength, private_rank=private_rank,
        n_subjects=n_subjects, n_target_voxels=n_target_voxels,
        loading_corr=loading_corr, signal_max_variance=signal_max_variance,
        signal_min_variance=signal_min_variance, seed=seed,
    )
    rng = np.random.default_rng(seed)
    total_rank = true_rank + private_rank
    # (total_rank,) latent variances, shared dimensions first
    variances = np.geomspace(signal_max_variance, signal_min_variance, total_rank)
    # All latent dimensions (shared, then each subject's private ones) are drawn
    # jointly orthogonal, so the sample signal covariance is exact and the two
    # subjects share exactly true_rank dimensions.
    all_variances = np.concatenate([variances[:true_rank]] + [variances[true_rank:]] * n_subjects)
    all_scores = _orthogonal_scores(rng, n_images, all_variances)  # (n_images, true_rank + n_subjects * private_rank)
    shared_scores = all_scores[:, :true_rank]  # (n_images, true_rank)

    subjects = []
    for subject_index, voxels in enumerate([n_voxels, n_target_voxels][:n_subjects]):
        start = true_rank + subject_index * private_rank
        private_scores = all_scores[:, start:start + private_rank]  # (n_images, private_rank)
        shared_loadings = _orthonormal_loadings(rng, voxels, true_rank, loading_corr)  # (voxels, true_rank)
        private_loadings = _orthonormal_loadings(rng, voxels, private_rank, 0.0)  # (voxels, private_rank)
        # (n_images, rank) @ (rank, voxels) -> (n_images, voxels)
        signal = shared_scores @ shared_loadings.T + private_scores @ private_loadings.T
        loadings = np.concatenate([shared_loadings, private_loadings], axis=1)  # (voxels, total_rank)
        # (voxels, total_rank) @ (total_rank, voxels) -> (voxels, voxels)
        signal_covariance = (loadings * variances) @ loadings.T
        noise, noise_covariance = _subject_noise(
            rng, n_images, voxels, n_repeats, noise_std, noise_type,
            noise_rank, noise_corr_strength,
        )
        # (n_images, voxels) -> (n_images, voxels, n_repeats)
        subjects.append((signal[:, :, None] + noise, signal, signal_covariance, noise_covariance))

    responses, signal, signal_covariance, noise_covariance = subjects[0]
    label = ",".join(f"{key}={value}" for key, value in settings.items())
    return RankData(
        name=f"toy[{label}]",
        responses=responses,
        image_ids=np.arange(n_images),
        subject="toy_subject_1",
        target_responses=subjects[1][0] if n_subjects == 2 else None,
        target_subject="toy_subject_2" if n_subjects == 2 else None,
        true_rank=true_rank if n_subjects == 2 else total_rank,
        info={
            "source": "toy",
            "shared_rank": true_rank,
            "private_rank": private_rank,
            "signal_rank": total_rank,
            "signal": signal,
            "signal_covariance": signal_covariance,
            "noise_covariance": noise_covariance,
            "target_signal": subjects[1][1] if n_subjects == 2 else None,
            **settings,
        },
    )
