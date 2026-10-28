"""Tiny no-data example for debugging cross-subject PLS rank selection.

Two fake subjects respond to the same images. Both responses contain a shared
stimulus signal of rank TRUE_SHARED_RANK; each subject also has a private
signal (seen by that subject only) and independent noise. Only the shared
signal lets the source predict the target.

Expected: with PRIVATE_RANK = 0, k* equals TRUE_SHARED_RANK. With source
private signal, k* can exceed it by a few ranks: the extra PLS components carry
source-private directions that improve prediction only by removing that private
variance from the shared estimate (suppression). The 1-SE rank stays at or
near TRUE_SHARED_RANK.
"""

import sys
import types
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# cross_subject_pls imports subject_pca, which imports the R-backed missMDA
# wrapper. The toy does not use R, so a stub lets it run on machines without R.
missmda_stub = types.ModuleType("pipeline.pca.missmda_ncp")
missmda_stub.estimate_ncp_pca = None
sys.modules.setdefault("pipeline.pca.missmda_ncp", missmda_stub)

from pipeline.pca.cross_subject_pls.cross_subject_pls import (
    pls_rank_curve,
    save_rank_curve,
)

N_IMAGES = 1000           # like the shared1000 images
N_SOURCE_VOXELS = 300
N_TARGET_VOXELS = 250
TRUE_SHARED_RANK = 10      # signal both subjects share -> expected k*
PRIVATE_RANK = 100          # signal each subject has alone -> should not count
NOISE_STD = 3.0
MAX_RANK = 40
N_FOLDS = 5
SEED = 0
OUTPUT_DIR = repo_root / "pipeline" / "pca" / "results" / "cross_subject_pls" / "toy"


def make_toy_subjects(seed: int = SEED) -> dict[str, np.ndarray]:
    """Create two fake subjects' responses to the same images.

    Inputs:
        seed: int, random seed.

    Output:
        toy: dict[str, np.ndarray] with ``source`` of shape
            (N_IMAGES, N_SOURCE_VOXELS), ``target`` of shape
            (N_IMAGES, N_TARGET_VOXELS), and the noiseless ``shared_signal``
            of shape (N_IMAGES, TRUE_SHARED_RANK).
    """
    rng = np.random.default_rng(seed)
    shared_signal = rng.normal(size=(N_IMAGES, TRUE_SHARED_RANK))
    source_private = rng.normal(size=(N_IMAGES, PRIVATE_RANK))
    target_private = rng.normal(size=(N_IMAGES, PRIVATE_RANK))

    # Each subject maps the latent signals to its own voxels.
    # (N_IMAGES, rank) @ (rank, n_voxels) -> (N_IMAGES, n_voxels)
    source = (
        shared_signal @ rng.normal(size=(TRUE_SHARED_RANK, N_SOURCE_VOXELS))
        + source_private @ rng.normal(size=(PRIVATE_RANK, N_SOURCE_VOXELS))
        + NOISE_STD * rng.normal(size=(N_IMAGES, N_SOURCE_VOXELS))
    )
    target = (
        shared_signal @ rng.normal(size=(TRUE_SHARED_RANK, N_TARGET_VOXELS))
        + target_private @ rng.normal(size=(PRIVATE_RANK, N_TARGET_VOXELS))
        + NOISE_STD * rng.normal(size=(N_IMAGES, N_TARGET_VOXELS))
    )
    return {"source": source, "target": target, "shared_signal": shared_signal}


def check_truncated_ranks(source: np.ndarray, target: np.ndarray, max_rank: int = 10) -> float:
    """Check that the first k components of one PLS fit equal a separate k-component fit.

    pls_rank_curve relies on this to get every rank from one fit.

    Inputs:
        source: np.ndarray with shape (n_images, n_source_voxels).
        target: np.ndarray with shape (n_images, n_target_voxels).
        max_rank: int, ranks 1..max_rank are compared.

    Output:
        worst: float, largest relative difference between the two predictions.
    """
    n_train = int(0.8 * source.shape[0])
    source_mean = source[:n_train].mean(axis=0)
    target_mean = target[:n_train].mean(axis=0)
    source_train = source[:n_train] - source_mean
    target_train = target[:n_train] - target_mean
    source_test = source[n_train:] - source_mean

    full = PLSRegression(n_components=max_rank, scale=False).fit(source_train, target_train)
    worst = 0.0
    for rank in range(1, max_rank + 1):
        # (n_test, v_s) @ (v_s, rank) @ (rank, v_t) -> (n_test, v_t)
        truncated = source_test @ full.x_rotations_[:, :rank] @ full.y_loadings_[:, :rank].T
        separate = PLSRegression(n_components=rank, scale=False).fit(
            source_train, target_train
        ).predict(source_test) - target_train.mean(axis=0)
        worst = max(worst, float(np.max(np.abs(truncated - separate)) / np.max(np.abs(separate))))
    return worst


if __name__ == "__main__":
    toy = make_toy_subjects()
    print(f"source {toy['source'].shape}, target {toy['target'].shape}, "
          f"true shared rank {TRUE_SHARED_RANK}, private rank {PRIVATE_RANK}")

    worst = check_truncated_ranks(toy["source"], toy["target"])
    print(f"Truncated vs separate PLS fits, max relative difference: {worst:.1e}")

    results = pls_rank_curve(
        toy["source"], toy["target"], max_rank=MAX_RANK, n_folds=N_FOLDS, seed=SEED
    )
    print("Mean normalized MSE, ranks 0-10:",
          np.round(results["mean_normalized_mse"][:11], 4))
    print(f"k* = {results['best_rank']}, 1-SE rank = {results['one_se_rank']} "
          f"(true shared rank {TRUE_SHARED_RANK}; k* may exceed it when PRIVATE_RANK > 0)")

    paths = save_rank_curve(results, "toy_source", "toy_target", OUTPUT_DIR)
    print(f"Saved {paths['csv']} and {paths['plot']}")
