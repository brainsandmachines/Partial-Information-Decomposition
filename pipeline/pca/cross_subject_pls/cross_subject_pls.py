"""Choose a cross-subject PLS rank from shared-image OTC responses.

One subject's repeat-averaged responses (source) predict another subject's
responses (target) to the same shared images through PLS regression. For each
rank k, the normalized held-out MSE is averaged over image folds; the rank with
the lowest mean MSE is k*.
"""

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cross_decomposition import PLSRegression

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.pca.subject_pca import split_unique_shared


def load_shared_pair(
    source_subject: str,
    target_subject: str,
    hdf_path: str | Path,
    pkl_info_path: str | Path,
    neural_data_dir: str | Path,
) -> dict[str, Any]:
    """Load two subjects' responses to their common shared images, aligned by image ID.

    Inputs:
        source_subject: str, e.g. ``"subj01"``; its responses are the predictors.
        target_subject: str, e.g. ``"subj02"``; its responses are predicted.
        hdf_path: str or Path, NSD stimulus HDF5 file.
        pkl_info_path: str or Path, NSD stimulus-information pickle.
        neural_data_dir: str or Path, directory with the per-subject
            ``{subject}_OTC_betas_per_stimulus.zarr`` stores (repeat-averaged).

    Output:
        pair: dict[str, Any] with ``source`` of shape (n_common, n_source_voxels),
            ``target`` of shape (n_common, n_target_voxels), and ``image_ids``
            of shape (n_common,), rows in the same image order.
    """
    shared = {}
    for subject in (source_subject, target_subject):
        split = split_unique_shared(
            subject,
            hdf_path,
            pkl_info_path,
            Path(neural_data_dir) / f"{subject}_OTC_betas_per_stimulus.zarr",
        )
        shared[subject] = (
            np.asarray(split["shared_images"]),
            np.asarray(split["shared_neural_data"], dtype=float),
        )

    source_ids, source_data = shared[source_subject]
    target_ids, target_data = shared[target_subject]
    if len(np.unique(source_ids)) != len(source_ids) or len(np.unique(target_ids)) != len(target_ids):
        raise ValueError("Each subject must have one row per shared image.")
    # (n_source_shared,), (n_target_shared,) -> (n_common,)
    image_ids, source_rows, target_rows = np.intersect1d(
        source_ids, target_ids, return_indices=True
    )
    if image_ids.shape[0] < 10:
        raise ValueError(
            f"Only {image_ids.shape[0]} shared images are common to "
            f"{source_subject} and {target_subject}; check the image-ID convention."
        )
    print(
        f"{source_subject}: {source_ids.shape[0]} shared images, "
        f"{target_subject}: {target_ids.shape[0]}, common: {image_ids.shape[0]}."
    )
    return {
        "source": source_data[source_rows],  # (n_source_shared, v_s) -> (n_common, v_s)
        "target": target_data[target_rows],  # (n_target_shared, v_t) -> (n_common, v_t)
        "image_ids": image_ids,
    }


def pls_rank_curve(
    source: np.ndarray,
    target: np.ndarray,
    max_rank: int,
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Cross-validate the PLS rank for predicting target responses from source responses.

    Images are split into folds. In each fold, source and target are centered
    with training means (no scaling), one PLS regression with ``max_rank``
    components is fit, and held-out predictions for every rank k come from its
    first k components. The held-out MSE is normalized by the MSE of predicting
    the training mean, so rank 0 has error 1.

    Inputs:
        source: np.ndarray with shape (n_images, n_source_voxels).
        target: np.ndarray with shape (n_images, n_target_voxels), same image order.
        max_rank: int, largest PLS rank tested; at most n_training_images - 1.
        n_folds: int, number of held-out image folds.
        seed: int, random seed used to assign images to folds.

    Output:
        results: dict[str, Any] with ``rank`` of shape (max_rank + 1,) starting
            at 0, ``fold_normalized_mse`` of shape (n_folds, max_rank + 1),
            ``mean_normalized_mse`` and ``standard_error`` across folds,
            ``best_rank`` (k*, lowest mean), and ``one_se_rank`` (smallest rank
            whose mean is within one standard error of the lowest).
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        raise ValueError("source and target must be 2D with the same number of images.")
    if not (np.all(np.isfinite(source)) and np.all(np.isfinite(target))):
        raise ValueError("source and target must be finite.")
    n_images = source.shape[0]
    if not 2 <= n_folds <= n_images:
        raise ValueError("n_folds must be between 2 and n_images.")
    min_training = n_images - int(np.ceil(n_images / n_folds))
    if not 1 <= max_rank <= min_training - 1:
        raise ValueError(f"max_rank must be between 1 and {min_training - 1}.")

    rng = np.random.default_rng(seed)
    # (n_images,) -> n_folds arrays with shape (n_validation_images,)
    validation_folds = np.array_split(rng.permutation(n_images), n_folds)
    fold_normalized_mse = np.empty((n_folds, max_rank + 1))

    for fold_index, validation_indices in enumerate(validation_folds):
        print(f"Cross-subject PLS: fold {fold_index + 1} of {n_folds}...")
        training_mask = np.ones(n_images, dtype=bool)
        training_mask[validation_indices] = False

        # 1. Center with training means only.
        source_mean = source[training_mask].mean(axis=0)  # (n_train, v_s) -> (v_s,)
        target_mean = target[training_mask].mean(axis=0)  # (n_train, v_t) -> (v_t,)
        source_train = source[training_mask] - source_mean
        target_train = target[training_mask] - target_mean
        source_test = source[validation_indices] - source_mean
        target_test = target[validation_indices] - target_mean

        # 2. One PLS fit with max_rank components; the k-rank model uses its
        #    first k x-rotations and y-loadings.
        pls = PLSRegression(n_components=max_rank, scale=False)
        pls.fit(source_train, target_train)
        # (n_test, v_s) @ (v_s, max_rank) -> (n_test, max_rank)
        test_scores = source_test @ pls.x_rotations_

        # 3. Held-out MSE for every rank, normalized by the rank-0 error.
        baseline_mse = np.mean(target_test ** 2)
        if baseline_mse == 0:
            raise ValueError("Held-out target responses have no variance.")
        prediction = np.zeros_like(target_test)  # (n_test, v_t)
        fold_normalized_mse[fold_index, 0] = 1.0
        for rank_index in range(max_rank):
            # (n_test, 1) * (1, v_t) -> (n_test, v_t)
            prediction += (
                test_scores[:, rank_index, None]
                * pls.y_loadings_[None, :, rank_index]
            )
            fold_normalized_mse[fold_index, rank_index + 1] = (
                np.mean((target_test - prediction) ** 2) / baseline_mse
            )

    # (n_folds, max_rank + 1) -> (max_rank + 1,)
    mean_normalized_mse = fold_normalized_mse.mean(axis=0)
    standard_error = fold_normalized_mse.std(axis=0, ddof=1) / np.sqrt(n_folds)
    best_rank = int(np.argmin(mean_normalized_mse))
    threshold = mean_normalized_mse[best_rank] + standard_error[best_rank]
    one_se_rank = int(np.flatnonzero(mean_normalized_mse <= threshold)[0])
    return {
        "rank": np.arange(max_rank + 1),
        "fold_normalized_mse": fold_normalized_mse,
        "mean_normalized_mse": mean_normalized_mse,
        "standard_error": standard_error,
        "best_rank": best_rank,
        "one_se_rank": one_se_rank,
    }


def save_rank_curve(
    results: dict[str, Any],
    source_subject: str,
    target_subject: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save the rank curve as a CSV and a PDF plot.

    Inputs:
        results: dict[str, Any] returned by ``pls_rank_curve``.
        source_subject: str, source subject identifier.
        target_subject: str, target subject identifier.
        output_dir: str or Path, directory for the outputs; created if missing.

    Output:
        paths: dict[str, Path] with the ``csv`` and ``plot`` file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"pls_rank_{source_subject}_to_{target_subject}"

    table = pd.DataFrame({
        "source_subject": source_subject,
        "target_subject": target_subject,
        "rank": results["rank"],
        "mean_normalized_mse": results["mean_normalized_mse"],
        "standard_error": results["standard_error"],
        "is_best_rank": results["rank"] == results["best_rank"],
        "is_one_se_rank": results["rank"] == results["one_se_rank"],
    })
    # (n_folds, max_rank + 1) -> n_folds columns of shape (max_rank + 1,)
    for fold_index, fold_mse in enumerate(results["fold_normalized_mse"], start=1):
        table[f"fold_{fold_index}_normalized_mse"] = fold_mse
    csv_path = stem.with_suffix(".csv")
    table.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    mean = results["mean_normalized_mse"]
    error = results["standard_error"]
    axis.plot(results["rank"], mean, color="tab:blue", label="mean over folds")
    axis.fill_between(results["rank"], mean - error, mean + error,
                      color="tab:blue", alpha=0.25, label="± 1 SE")
    axis.axvline(results["best_rank"], color="tab:red", linestyle="--",
                 label=f"k* = {results['best_rank']}")
    axis.axvline(results["one_se_rank"], color="tab:gray", linestyle=":",
                 label=f"1-SE rank = {results['one_se_rank']}")
    axis.set(xlabel="PLS rank k", ylabel="normalized held-out MSE",
             title=f"Cross-subject PLS: {source_subject} → {target_subject}")
    axis.legend()
    plot_path = stem.with_suffix(".pdf")
    figure.savefig(plot_path, bbox_inches="tight")
    plt.close(figure)
    return {"csv": csv_path, "plot": plot_path}


if __name__ == "__main__":
    with open(Path(__file__).with_name("cross_subject_pls.yaml"), "r") as config_file:
        config = yaml.safe_load(config_file)

    pair = load_shared_pair(
        config["source_subject"],
        config["target_subject"],
        config["hdf_path"],
        config["pkl_info_path"],
        config["neural_data_dir"],
    )
    results = pls_rank_curve(
        pair["source"],
        pair["target"],
        max_rank=config["max_rank"],
        n_folds=config["n_folds"],
        seed=config["seed"],
    )
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    paths = save_rank_curve(
        results, config["source_subject"], config["target_subject"], output_dir
    )
    print(
        f"k* = {results['best_rank']} (1-SE rank = {results['one_se_rank']}). "
        f"Saved {paths['csv']} and {paths['plot']}."
    )
