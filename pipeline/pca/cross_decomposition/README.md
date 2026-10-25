# Cross-decomposition versus GSN+PCA reconstruction

This comparison asks whether estimating directions from training cross-covariance
changes the component count that best reconstructs an independent repetition of
held-out stimuli. It does not run DNN encoding models or PID, and does not save
projected neural data.

## Existing pipeline inspected

- `pipeline/pca/choosing_minimal_best_pca/choosing_minimal_best_pca.py` loads
  `{subject}_betas_img_vox_rep.zarr`, variable `betas_by_image_voxel_repeat`, as
  `(images, voxels, repeats)`. The ROI is already OTC in these stores.
- `image_set: unique` removes shared1000 using stimulus IDs and NSD metadata;
  `all` keeps shared images too. Nonfinite responses cause a report and failure,
  with no silent removal. The comparison calls this loader unchanged.
- No z-scoring or variance scaling is applied by the loader or selector.
  These are the stored beta units; the code does not undo upstream processing.
- `pipeline/testing/gsn/gsn_pca.py::select_gsn_pca_components` partitions a seeded
  permutation of image rows with `np.array_split`. A fold's training images are
  the complement of its validation images. GSN fits `cSb` using training images
  and all their repetitions; leading eigenvectors define the projection.
- Each validation split averages complementary disjoint repetition groups. With
  three repetitions, all three 1-versus-2 partitions are scored in both directions.
  The current `input_repeat_count` weighting assigns total weight 2/3 to two-repeat
  inputs and 1/3 to one-repeat inputs. `equal` remains available.
- For each directed split, the score is
  `mean((target - reconstruction)**2) / mean((target - training_mean)**2)`.
  `training_mean` pools training images and repetitions, voxel by voxel. Ratios
  are averaged across directed splits with the configured weights, then across
  folds equally. Standard errors are fold sample SD divided by `sqrt(n_folds)`.
- Candidates are every integer from 1 to `max_components`; K=0 is not selected.
  `best_n_components` minimizes the mean curve. `selected_n_components` is the
  smallest count within one SE of that minimum. Both are reported separately.
- The original runner saves scalar/per-candidate/per-fold results to a CSV,
  including the seed in its name. It does not produce a reconstruction-curve
  comparison plot and has no nested or independent final evaluation stage.

## New method and comparison choices

`cross_decomposition.py` is independently importable and reusable. For each
training fold and unordered repetition partition, it averages each group and
centers each resulting `(n_train, d)` matrix by its own training-image mean. It
fits `C = A.T @ B / n_train = U diag(s) V.T`. Singular values are descending.
One SVD provides U for the forward direction and V for the reverse direction.

Reconstruction uses `mu + (observed - mu) @ basis[:, :K] @ basis[:, :K].T`, with
the **same pooled training mean and same baseline as GSN**. The separate means
used to form the cross-covariance are not substituted into the reconstruction
metric. This keeps the reconstruction/evaluation protocol identical between
methods. Both bases and their training means are available from
`fit_cross_decomposition` for separate uses.

The basis estimation follows [Gauthaman, Ménard & Bonner (2025), Methods §4.2](https://doi.org/10.1371/journal.pcbi.1013714).
Their original endpoint is a held-out covariance spectrum. This implementation
uses the requested orthogonal reconstruction endpoint and the existing GSN
preprocessing/metric; it does not reproduce their full spectral analysis or add
their voxel z-scoring.

The selector computes the same projection errors efficiently through
`SSE(K) = SSE(0) + sum_j<=K (||scores_j||² - 2 scores_j.T target_scores_j)`.
This identity requires orthonormal basis columns. No reconstruction/projector
matrix is built for every K. K=0 is available in the metric helper as a diagnostic
baseline (normalized error 1), but excluded from both selection grids.

### Computation and limits

- `svd_solver: auto` uses SciPy's converged iterative SVD on a `LinearOperator`
  when K is less than half the theoretical centered rank bound. The operator
  computes `A.T @ (B @ v) / n_train`; no voxel-square covariance is allocated.
  A seeded starting vector ensures reproducibility. Solver failure is surfaced,
  not replaced silently with another method.
- Otherwise, reduced QR of `A.T` and `B.T` reduces the exact SVD to a matrix of
  side `min(n_train, d)`. For `d > n_train`, this avoids a d-by-d matrix. Exact
  decomposition can still be expensive for large training folds.
- Every candidate must satisfy `K <= min(d, smallest_n_train - 1)`. Unsupported
  caps fail explicitly rather than completing an unidentifiable null basis or
  quietly giving the methods different grids. Reduce the cap in the source YAML
  for **both methods** if needed. Degenerate singular-value subspaces need not
  have uniquely defined individual directions.
- The unchanged GSN side still constructs voxel-square covariances; its original
  memory requirements remain. No shrinkage or regularization was added to the
  cross-decomposition method, and no new dependency was installed.

## Run on the same OTC data

From the repository root, in the existing environment with official GSN and
the original loader's dependencies installed:

```bash
conda activate PID_env
python -m pipeline.pca.cross_decomposition.compare_reconstruction
```

No argparse is used. Edit `compare_reconstruction.yaml` in this directory. Its
`gsn_config_path` defaults to the existing `choosing_minimal_best_pca.yaml`; that
file supplies subjects, data paths, image set, seed, fold count, repetition
weighting, and component cap for both methods. Relative paths resolve from the
repository root. The defaults are subj01, unique images, 5 folds, seed 0, and
K=1..1000. Keep `unique` when preserving shared1000 for later evaluation.

The runner reloads neither images nor external saved GSN curves between methods:
it passes the same response array and selection settings to both. GSN is rerun,
so stale CSVs with different preprocessing, weighting, or folds are not compared.
It also seeds/restores NumPy's global RNG around GSN for its internal randomness.

## Outputs

Fresh directories under `pipeline/pca/results/cross_decomposition_comparison/`
contain subject, image set, seed, and UTC timestamp in their names. Existing
GSN outputs are never overwritten. Each completed directory contains:

| File | Contents |
| --- | --- |
| `gsn_pca.csv`, `cross_decomposition.csv` | Every candidate's mean, fold scores, SD, SE, argmin and one-SE choices. Cross CSV also includes directed errors. |
| `summary.csv` | Both choices, minimum validation normalized MSE and variability for each method. |
| `paired_differences.csv` | Cross minus GSN at each K, per fold and averaged, with paired fold SD/SE. Negative means lower cross-decomposition error. |
| `reconstruction_comparison.pdf` | Both curves with SE shading and argmin markers; lower panel shows paired differences. |
| `image_folds.csv` | Image IDs in loaded order and 1-based validation-fold membership. Training images are each fold's complement. |
| `repeat_splits.csv` | Ordered input/target groups using 0-based repetition-axis indices and their weights. |
| `run_config.yaml` | Source configuration/path, resolved selector settings, data shape, centering conventions and NumPy/SciPy versions. |

In the cross CSV, `direction_normalized_reconstruction_errors_i` uses 1-based
fold-major indexing: `i = (fold - 1)*n_directions + direction`. In both method
CSVs, `fold_normalized_reconstruction_errors_i` denotes an image validation fold,
after averaging its repetition-direction scores.

These are **component-selection validation curves**. Their minima are selected
on those same folds and are not unbiased final-test performance estimates.
Fold SD/SE describe variability; overlapping training sets mean the SE shading
is not an independent-fold confidence interval or significance test. For a final
generalization claim, freeze the chosen K and evaluate on a separate untouched
stimulus set, or add nested validation. This runner leaves shared1000 untouched
when `image_set: unique` and does not claim final-test performance.

## Validation and changes to existing code

```bash
python -m pytest tests/test_cross_decomposition.py -q
```

Tests compare exact/iterative decompositions to independent dense SVD; verify
descending spectra, symmetric/idempotent projectors, zero/full reconstruction,
literal normalized MSE, both U/V directions, training-only fitting and disjoint
stimuli, worsening error when adding independent noise, `d > n_train`, rank-bound
validation, and exact protocol agreement with the unchanged GSN selector given
identical bases. A tiny real-GSN synthetic integration test checks reproducible
CSV/plot outputs and fresh run directories; it skips if GSN is not installed.
No NSD experiment is launched by these tests.

No existing GSN function, shared utility, configuration, or result was modified.
The existing loader, table serializer and GSN selector are reused. Because the
fold/repetition logic is internal to the existing selector, the new module mirrors
that small block; the protocol-equivalence test guards against future drift.
No existing plotting helper implements these reconstruction/difference curves,
so the new runner draws the comparison directly. New callables are documented in
`FUNCTION_REGISTRY.md`.
