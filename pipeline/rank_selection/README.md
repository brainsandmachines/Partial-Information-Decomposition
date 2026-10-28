# Rank selection

How many PCs (or ranks) of the OTC responses should the PID target keep? Every
method that answers this lives here, takes the same input, returns the same
output, and runs from one entry point on toy data or OTC data.

## Methods

| Name (`methods:` key) | Options | What counts as "predictable" | Selected rank |
|---|---|---|---|
| `variance_threshold` | `variance_threshold`, `folds` | variance of all images (no held-out data); a naive held-out projection curve is reported alongside | smallest count reaching the threshold |
| `eigenvector` | `basis: pca \| gsn`, `folds: K \| loo`, `max_components`, `device`, `checkpoint_path` | each voxel of a held-out image predicted from its other voxels (same measurement, so voxel-correlated noise counts) | minimum MSEP |
| `repeat_reconstruction` | `basis: gsn \| pca`, `folds`, `max_components`, `repeat_split_weighting` | a held-out image's independent repeats (noise does not carry over) | smallest count within 1 SE of the minimum (the original GSN rule) |
| `mle` | — | Minka's evidence under probabilistic PCA with isotropic noise; unequal or correlated noise inflates it | sklearn `PCA(n_components="mle")` |
| `cross_decomposition` | `n_permutations`, `folds`, `randomized`, binning | the authors' held-out cross-covariance between repeats 0 and 1 vs a permutation null (Gauthaman et al., 2025) | leading consecutive significant components |
| `pls` | `max_rank`, `folds` | a second subject's responses to the same images | minimum normalized MSE (k\*); the 1-SE rank is also reported |

`eigenvector` and `repeat_reconstruction` form a 2 × 2 design that changes one
thing at a time:

| basis \\ criterion | same measurement (`eigenvector`) | independent repeat (`repeat_reconstruction`) |
|---|---|---|
| PCA | `basis: pca` | `basis: pca` |
| GSN signal covariance | `basis: gsn` | `basis: gsn` |

Every result reports `selected_rank` (the method's own rule) and, where folds
exist, `best_rank` (optimum) and `one_se_rank`.

## Running

```bash
# from the repository root
python -m pipeline.rank_selection.run --config pipeline/rank_selection/configs/smoke.yaml
python -m pipeline.rank_selection.run --config <config> --list          # numbered task list
python -m pipeline.rank_selection.run --config <config> --task-index 3  # one task
python -m pipeline.rank_selection.run --config <config> --only pls mle  # some methods
python -m pipeline.rank_selection.run --config <config> --summarize     # rebuild summary + plots
```

On the cluster, `slurm/submit.sh` submits one array task per config task plus a
final summary job (run it from the repository root with the conda env active):

```bash
bash pipeline/rank_selection/slurm/submit.sh pipeline/rank_selection/configs/otc.yaml pid_env
ONLY=cross_decomposition bash pipeline/rank_selection/slurm/submit.sh \
    pipeline/rank_selection/configs/otc.yaml cross_decomposition --gpus=1 --partition=rtx6000 --qos=golan-neuro
```

**Environments.** `cross_decomposition` needs the authors' `lib` and `bonner`
(conda env `cross_decomposition`); the GSN methods need `gsn` (`pid_env`). In an
environment that lacks a method's dependencies the runner reports it as
unavailable and continues; nothing is saved, so it runs later in the right
environment against the same results folder.

## Configs

`configs/smoke.yaml` (seconds), `configs/toy_grid.yaml` (controlled comparison),
`configs/otc.yaml` (real data; choose `subjects` in the file). A config has:

- `experiment_name`, optional `output_root` (default `pipeline/rank_selection/results/`, git-ignored)
- `data`:
  - toy: `fixed:` and `grid:` parameters of `data.make_toy`; each grid combination is one dataset
  - otc: `subjects`, `image_set` (`unique | shared | all`), `store` (`repeat | per_stimulus`),
    `pairs` (`all_ordered` or a list) and `pair_image_set` for pair methods
- `methods`: `{name: options}` or `{name: [options, ...]}`; list-valued options expand into variants
- `seeds`: fold/permutation seeds (seed-free methods run once)

## Outputs

```
results/<experiment_name>/
  config.yaml      exact copy of the config that ran (a different config is refused)
  run_info.yaml    git commit, host, date, package versions, SLURM IDs
  summary.csv      one row per dataset x method x seed
  rank_recovery.pdf  toy only: selected minus true rank
  <dataset>/
    <method>_seed<k>.csv  .json  .pdf    curve, scalars, plot
    overlay.pdf                         all methods on this dataset
```

Rerunning the same config skips finished results.

## Data

`data.RankData` holds responses `(images, voxels, repeats)`, image IDs, and
optionally a second subject (`target_responses`) for pair methods.

- `load_otc(subject, image_set, store)` / `load_otc_pair(source, target, ...)`.
  **The two OTC stores disagree** (subj01: correlation 0.48 between the
  per-stimulus store and the repeat-store average), so `store` is always explicit.
- `make_toy(...)`: shared signal of rank `true_rank` (+ `private_rank` per
  subject) with exact, known covariance; noise `isotropic`, `unequal_variance`,
  or `correlated` (rank `noise_rank`), independent across repeats and subjects.
  Single-subject truth = `true_rank + private_rank`; pair truth = `true_rank`.

## Layout

`data.py`, `result.py` (`RankResult`), `press.py` (shared leave-one-voxel-out
kernel, r_j / (1 − h_j)), `methods/`, `registry.py`, `run.py`, `plotting.py`,
`configs/`, `slurm/`, `archive/` (missMDA wrapper and row-wise LOO PCA, kept but
not maintained). Tests: `tests/rank_selection/`.
