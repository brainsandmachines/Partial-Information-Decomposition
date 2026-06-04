# Duplicate Function Audit

Generated: 2026-06-02

This audit covers the actionable thesis code: `PID/`, `library_wrappers/`,
`Partial_Information_Decomposition/`, `encoding_model/`, `supression_effect/`,
`toy_examples/`, `Simulations/`, `data/`, `uni_tests/`, and root utility/test
files. I treated `external/`, `gpid/`, and `source_conwell_code/` as
vendor/reference code, not deletion candidates.

I found 84 scoped Python files, 4 scoped R files, and 589 Python/R functions or
classes. "Erase" below means erase only after updating imports/callers and
running tests. No source files were changed by this audit.

## Executive Summary

Highest-value cleanup targets:

1. Centralize the regression scoring and commonality-analysis code. The same
   OLS/Ridge/Lasso R2 helpers and `commonality_analysis` appear in core code and
   three toy scripts.
2. Move copied Idep demo helpers (`ones`, `equicorr_blocks`, `build_full_cov`,
   `run_one`, `main`) out of the main Idep class files.
3. Consolidate wrapper helpers in `library_wrappers/wrapper_utils.py`.
   `parse_sizes`, `csv_shape`, and Rscript discovery are copied across wrappers.
4. Refactor M7/M8 simulation files into parameterized whitened/not-whitened
   variants instead of maintaining parallel files with copied orchestration.
5. Archive or delete stale scripts such as `Partial_Information_Decomposition/Dump.py`,
   root `test.py`, and the hard-coded R example scripts once you confirm they
   are not part of current experiments.

## Duplicate And Same-Logic Groups

### G01: Regression R2 Helpers

Functions:

- `encoding_model/encoding_utils.py:441` `compute_ols_cv_r2`
- `toy_examples/toy_example.py:14` `compute_ols_cv_r2`
- `toy_examples/toy_example_feature_correlation.py:5` `compute_ols_cv_r2`
- `toy_examples/toy_example_new.py:5` `compute_ols_cv_r2`
- `Partial_Information_Decomposition/PID_util.py:33` `compute_ridge_cv_r2`
- `encoding_model/encoding_utils.py:460` `compute_ridge_cv_r2`
- `toy_examples/toy_example.py:33` `compute_ridge_cv_r2`
- `toy_examples/toy_example_feature_correlation.py:24` `compute_ridge_cv_r2`
- `toy_examples/toy_example_new.py:24` `compute_ridge_cv_r2`
- `encoding_model/encoding_utils.py:487` `compute_r2`
- `toy_examples/toy_example.py:60` `compute_r2`
- `toy_examples/toy_example_feature_correlation.py:51` `compute_r2`
- `toy_examples/toy_example_new.py:51` `compute_r2`
- `encoding_model/encoding_utils.py:502` `compute_lasso_cv_r2`
- `toy_examples/toy_example.py:75` `compute_lasso_cv_r2`

What they do: fit OLS/Ridge/Lasso regressions and return in-sample or
cross-validated R2 scores for commonality analysis.

Similarity: exact or near-exact. The main behavioral difference is return
shape: some versions return only a score, while `toy_examples/toy_example.py`
returns `(model, score)`, and `PID_util.compute_ridge_cv_r2` returns
`(score, ridge_cv)` while also converting torch tensors to NumPy.

Keep: keep `encoding_model/encoding_utils.py:441`, `:460`, `:487`, `:502` as
the current canonical implementation for Python package users. Keep
`Partial_Information_Decomposition/PID_util.py:33` temporarily because existing
PID code expects the model-returning/torch-aware behavior.

Erase/merge: erase the copies in `toy_examples/toy_example_feature_correlation.py`
and `toy_examples/toy_example_new.py`. Merge the model-returning behavior from
`toy_examples/toy_example.py` into the canonical helper, for example with a
`return_model` option, then erase those toy copies too.

Important call sites checked: `encoding_model/suppression_core.py` imports the
encoding helpers; `toy_examples/toy_example*.py` use local copies;
`Partial_Information_Decomposition/Idep_Simulations/lr_Idep.py:23` and
`PID_util.py:1024` call the `PID_util` Ridge version.

### G02: Commonality Analysis

Functions:

- `encoding_model/suppression_core.py:152` `commonality_analysis`
- `toy_examples/toy_example.py:93` `commonality_analysis`
- `toy_examples/toy_example_feature_correlation.py:67` `commonality_analysis`
- `toy_examples/toy_example_new.py:67` `commonality_analysis`

What they do: compute R2 for source 1, source 2, and joint sources, then
decompose variance into unique source 1, unique source 2, common/redundant, and
unexplained components.

Similarity: same logic. The toy versions include shape normalization and
sometimes variance scaling or beta outputs. `toy_examples/toy_example.py` is the
richest version because it can return regression coefficients and supports
`lasso_cv`.

Keep: keep `encoding_model/suppression_core.py:152` as the current canonical
core function because it is imported by `supression_effect/Suppresed_Encoder.py`
and tested by `uni_tests/test_supression_core.py`.

Erase/merge: move the useful beta-output and optional Lasso behavior from
`toy_examples/toy_example.py:93` into the core implementation, then replace the
three toy definitions with imports or delete them. `toy_example_new.py` and
`toy_example_feature_correlation.py` should not keep private copies.

Important call sites checked: `supression_effect/Suppresed_Encoder.py:116`,
`encoding_model/suppression_core.py:195`, `toy_examples/toy_example.py:373`,
`toy_examples/toy_example_new.py:176`, `toy_examples/toy_example_feature_correlation.py:189`,
and `uni_tests/test_supression_core.py:74`.

### G03: Suppression/Permutation Model Builders

Functions:

- `encoding_model/suppression_core.py:64` `permutate_models`
- `toy_examples/toy_example.py:165` `permutate_models`
- `toy_examples/toy_example.py:305` `run_experiment`
- `toy_examples/toy_example_feature_correlation.py:145` `run_experiment`
- `toy_examples/toy_example_new.py:133` `run_experiment`
- `encoding_model/suppression_core.py:186` `run_all_methods`
- `toy_examples/toy_example.py:396` `run_all_methods`
- `toy_examples/toy_example_feature_correlation.py:212` `run_all_methods`
- `toy_examples/toy_example_new.py:199` `run_all_methods`

What they do: generate two source feature matrices that share/permutate signal
and nuisance dimensions, then run the commonality-analysis methods.

Similarity: `permutate_models` is a near-copy; the toy version only adds local
variables for `spurious_M1`/`spurious_M2`. The `run_experiment` and
`run_all_methods` functions share the same experiment skeleton with different
data generators.

Keep: keep `encoding_model/suppression_core.py:64` and
`encoding_model/suppression_core.py:186` as the canonical suppression pipeline.

Erase/merge: erase `toy_examples/toy_example.py:165` after updating the toy to
import `encoding_model.suppression_core.permutate_models`. Merge
`toy_example_new.py` and `toy_example_feature_correlation.py` into a single
toy/example driver with a generator parameter, then delete the duplicate
`run_experiment`/`run_all_methods` copies.

Important call sites checked: `encoding_model/suppression_core.py:97`,
`encoding_model/suppression_core.py:135`, `toy_examples/toy_example.py:341`,
and `uni_tests/test_supression_core.py:46`.

### G04: Standardization Helpers

Functions:

- `Partial_Information_Decomposition/PID_util.py:377` `standardize`
- `supression_effect/supp_gauss_multivariate.py:39` `standardize`
- `Simulations/turned_off_unqiue.py:88` `standardize`

What they do: column-standardize arrays to zero mean and unit variance.

Similarity: `PID_util.py:377` and `supp_gauss_multivariate.py:39` are exact
Torch copies. `turned_off_unqiue.py:88` is the same idea but uses NumPy despite
the torch type annotation.

Keep: keep `Partial_Information_Decomposition/PID_util.py:377` for Torch PID
code. If NumPy standardization is needed, create one clearly named helper such
as `standardize_np` in `utils.py`.

Erase/merge: erase `supression_effect/supp_gauss_multivariate.py:39`; replace
uses with the PID utility. Replace `Simulations/turned_off_unqiue.py:88` with a
NumPy helper and fix its misleading type annotation.

Important call sites checked: `Simulations/turned_off_unqiue.py:101`,
`:102`, `:103`, `:156`, `:157`, and `:158`.

### G05: Correlation/Singularity/Diagnostic Plot Helpers

Functions:

- `Partial_Information_Decomposition/PID_util.py:413` `correlation_matrix`
- `encoding_model/encoding_utils.py:521` `correlation_matrix`
- `Partial_Information_Decomposition/PID_util.py:442` `singularity_report`
- `encoding_model/encoding_utils.py:529` `singularity_report`
- `Partial_Information_Decomposition/PID_util.py:467` `diagnostic_plots`
- `encoding_model/encoding_utils.py:555` `diagnostic_plots`
- nested `cross_correlation` inside `PID_util.py:468`
- nested `cross_correlation` inside `encoding_utils.py:556`

What they do: compute column correlations, inspect singular/ill-conditioned
blocks, and plot blockwise source/target correlation matrices.

Similarity: `correlation_matrix`, `diagnostic_plots`, and nested
`cross_correlation` are exact copies. `singularity_report` is same-purpose but
not exact: the PID version uses `block_singularity_check` and returns
`(report, printing_required)`, while the encoding version returns only the
report and computes eigenvalues through `correlation_matrix`.

Keep: keep `Partial_Information_Decomposition/PID_util.py:413` and
`PID_util.py:467` as canonical mathematical diagnostics. Keep
`PID_util.py:442` as the canonical singularity function after normalizing its
return type.

Erase/merge: erase `encoding_model/encoding_utils.py:521` and `:555` after
imports are updated. Do not erase `encoding_model/encoding_utils.py:529` until
callers are adapted to the PID return contract or the contracts are unified.

Important call sites checked: `encoding_model/suppression_core.py:7`,
`toy_examples/toy_example.py:11`, `toy_examples/toy_example.py:363`, and Idep
imports in `Idep_multivariate_gauss.py` and `jacknife_Idep_multivariate_gauss.py`.

### G06: Result-Comparison And Suppression-Effect Printers

Functions:

- `Partial_Information_Decomposition/PID_util.py:558` `compare_results`
- `supression_effect/supp_gauss_multivariate.py:185` `compare_results`
- `supression_effect/gauss_unvariate.py:124` `compare_results`
- `Partial_Information_Decomposition/Dump.py:5` `check_supression_effect`
- `supression_effect/gauss_unvariate.py:75` `check_supression_effect`

What they do: print side-by-side variance partitioning, MI, and PID components,
and identify suppression effects.

Similarity: `PID_util.py:558` and `supp_gauss_multivariate.py:185` are exact
copies. The univariate version originally differed only because it used older
A/B-style result keys instead of the current X1/X2-style result keys.

Keep: keep `Partial_Information_Decomposition/PID_util.py:558` for multivariate
PID/MI output. After the G06 follow-up, the univariate script also uses the
same X1/X2 result-key contract and imports the PID utility printer.

Erase/merge: erase `supression_effect/supp_gauss_multivariate.py:185` after
importing the PID utility. Erase `Partial_Information_Decomposition/Dump.py:5`
or delete the whole `Dump.py` file; I found no imports or active call sites for
it.

Important call sites checked: `supression_effect/supp_gauss_multivariate.py:293`
and later calls; `supression_effect/gauss_unvariate.py:150`, `:169`, `:174`,
`:179`, `:184`, `:189`, and `:194`.

### G07: Evil-Twin Toy Helpers

Functions:

- `evil_twin/covariance_example.py:6` `empirical_covariance_matrix_torch`
- `evil_twin/covariance_example.py:24` `check_evil_twin_covariances_torch`
- `evil_twin/covariance_example.py:70` `evil_twin_example_torch`
- `evil_twin/covariance_example.py:128` `run_covariance_comparison`

What they do: generate the sonic/shadow evil-twin Gaussian examples, validate
their covariance structure, and provide a small demonstration entry point for
comparing the Sonic and Shadow covariance matrices.

Similarity: the old `toy_examples/evil_twin.py` script mixed reusable
Sonic/Shadow covariance helpers with top-level PID execution. The reusable
covariance helpers now live in the dedicated root-level `evil_twin/` package.

Keep: keep `evil_twin_example_torch` and the covariance comparison helpers in
`evil_twin/covariance_example.py`.

Erase/merge: `toy_examples/evil_twin.py` was deleted after moving the reusable
covariance example into the new package. The old `evil_twin_idep` and
`evil_twin_tilde_pid` script-local helpers were not kept because this G07 scope
is the Sonic/Shadow covariance example the thesis still needs to show.

Important call sites checked: no scoped files imported the old
`toy_examples/evil_twin.py`. New imports should use
`from evil_twin import evil_twin_example_torch`.

### G08: Python Wrapper Helper Copies

Functions:

- `library_wrappers/wrapper_utils.py:11` `parse_sizes`
- `library_wrappers/wrapper_utils.py:22` `csv_shape`
- `library_wrappers/wrapper_utils.py:42` `find_rscript`
- `library_wrappers/Delta_PID.py`, `IG_R.py`, `Idep_R.py`, `Tilde_PID.py`,
  and `check_evil_twin_all.py` import these shared helpers.

What they do: parse block-size CLI arguments, validate CSV shapes, locate
Rscript, and build default simple-Gaussian CLI args.

Similarity: `parse_sizes`, `csv_shape`, and `find_rscript` were repeated across
the Python wrappers. They now have one shared implementation.

Keep: keep `library_wrappers/wrapper_utils.py:11` as canonical for
`parse_sizes`. `csv_shape` and `find_rscript` now also live in
`wrapper_utils.py`.

Erase/merge: local copies were removed from `IG_R.py`, `Idep_R.py`, and
`check_evil_twin_all.py`; `Delta_PID.py` and `Tilde_PID.py` now also use the
shared `parse_sizes`.
Do not erase all `simple_example_args` blindly; they are same-purpose but
wrapper-specific. Factor shared defaults if desired, but keep per-wrapper args
until a shared helper can preserve each wrapper's fields.

Important call sites checked: active Python wrappers import from
`wrapper_utils.py` with a relative import plus script-style fallback.

### G09: R Idep Wrapper Responsibility Split

Functions/classes:

- `library_wrappers/Idep_R.py:40` embedded R `load_idep_gauss`
- `library_wrappers/r_idep_client.py:24` embedded R `load_idep_gauss`
- `library_wrappers/Idep_R.py:266` `main`
- `library_wrappers/r_idep_client.py:89` `RIdePResult`
- `library_wrappers/r_idep_client.py:144` `run_idep_from_covariance`
- `library_wrappers/r_idep_client.py:216` `run_idep_for_cases`
- `library_wrappers/r_idep_client.py:228` `atoms_as_ordered_values`

What they do: call JWKay/PID `idepGM` from Python through Rscript. The
library wrapper is CLI/report oriented; the toy wrapper is a programmatic API
used by the evil-twin comparison script.

Similarity: same responsibility, different interface.

Keep: keep `library_wrappers/Idep_R.py` as the CLI home. The programmatic API
now lives in `library_wrappers/r_idep_client.py`.

Erase/merge: `toy_examples/r_idep_wrapper.py` is removed. The old comparison
script is also not active in the current worktree, so no shim was needed.

Important call sites checked:
no active scoped files import `toy_examples/r_idep_wrapper.py`.

### G10: R IG Functions With Shared Core Logic

Functions:

- `PID/IGFuns.R:3` `IG_GaussM_Core`
- `PID/IGFuns.R:119` `IG_GaussM_PQR`
- `PID/IGFuns.R:126` `IG_GaussM_Dat`
- `PID/IGFuns.R:166` `IG_GaussU_pqr`

What they do: compute IG PID for Gaussian variables. `IG_GaussM_PQR` accepts
already-whitened P/Q/R blocks; `IG_GaussM_Dat` accepts a full covariance matrix
and first derives P/Q/R; `IG_GaussU_pqr` is the scalar trivariate version.

Similarity: `IG_GaussM_PQR` and `IG_GaussM_Dat` share most of their core after
P/Q/R are available. `IG_GaussU_pqr` is related but dimension-specific.

Keep: keep all three public R functions because they are different input APIs.

Erase/merge: no public R function was erased. The shared multivariate core now
lives in `IG_GaussM_Core(sizes, P, Q, R)`, and both `IG_GaussM_PQR` and
`IG_GaussM_Dat` call it.

Related note: `PID/IGscript.R` and `PID/IdepGscript.R` are hard-coded example
scripts with old absolute paths. They were moved to `PID/examples/`.

### G11: Idep Class Demo Helpers And Class Name Collision

Status: cleaned in `G11_IDEP_DEMO_HELPER_CLEANUP.md`.

Kept implementation classes:

- `Partial_Information_Decomposition/Idep_multivariate_gauss.py:30` class
  `Idep_multivariate_gauss`
- `Partial_Information_Decomposition/jacknife_Idep_multivariate_gauss.py:23`
  class `JackknifeIdepMultivariateGauss`
- `Partial_Information_Decomposition/jacknife_Idep_multivariate_gauss.py:291`
  compatibility alias `Idep_multivariate_gauss = JackknifeIdepMultivariateGauss`
- `Partial_Information_Decomposition/parallel_Idep_multivariate_gauss.py:23`
  class `para_Idep_multivariate_gauss`

Kept demo helpers in one example module:

- `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:27` `ones`
- `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:31`
  `equicorr_blocks`
- `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:48`
  `build_full_cov`
- `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:80`
  `run_serial_example`
- `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:95`
  `run_jackknife_example`
- `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:114`
  `run_parallel_example`
- `Partial_Information_Decomposition/examples/idep_equicorr_demo.py:131` `main`

What they do: the class methods implement Idep for different settings; the
example helpers build canonical equicorrelation examples and run demo checks.

Similarity: the old helper functions were exact duplicates across three solver
files. The solver classes remain separate because they differ by bias
correction, jackknife behavior, and batched/parallel tensor shape.

Kept/erased: kept the main solver, renamed the jackknife class explicitly, kept
the parallel solver, moved the reusable demo helpers into
`examples/idep_equicorr_demo.py`, and erased the copied bottom-of-file demo
helpers from all three solver files. `pretty` was erased because it was only
local demo formatting.

Important call sites checked: production code imports
`Partial_Information_Decomposition.Idep_multivariate_gauss.Idep_multivariate_gauss`;
no active code imported the old duplicated demo helpers.

### G12: M7/M8 Simulation Orchestration

Functions:

- `Partial_Information_Decomposition/Idep_Simulations/logdet_m7_m8.py:18`
  `simulate_m7_m8_log_det`
- `Partial_Information_Decomposition/Idep_Simulations/logdet_m7_m8_notwhiten.py:18`
  `simulate_m7_m8_log_det`
- `Partial_Information_Decomposition/Idep_Simulations/logdet_m7_m8.py:200`
  `calculate_bias`
- `Partial_Information_Decomposition/Idep_Simulations/logdet_m7_m8_notwhiten.py:201`
  `calculate_bias`
- `Partial_Information_Decomposition/Idep_Simulations/logdet_m7_m8.py:234`
  `simulation_wrapper`
- `Partial_Information_Decomposition/Idep_Simulations/logdet_m7_m8_notwhiten.py:235`
  `simulation_wrapper`
- `Partial_Information_Decomposition/Idep_Simulations/logdet_m7_m8.py:257`
  `sort_m7_m8_results`
- `Partial_Information_Decomposition/Idep_Simulations/logdet_m7_m8_notwhiten.py:258`
  `sort_m7_m8_results`
- `Partial_Information_Decomposition/Idep_Simulations/mi_m7_m8.py:23`
  `simulate_m7_m8_mi`
- `Partial_Information_Decomposition/Idep_Simulations/mi_m7_m8_notwhiten.py:24`
  `simulate_m7_m8_mi`
- `Partial_Information_Decomposition/Idep_Simulations/mi_m7_m8.py:335`
  `sort_m7_m8_results`
- `Partial_Information_Decomposition/Idep_Simulations/mi_m7_m8_notwhiten.py:375`
  `sort_m7_m8_results`
- `Partial_Information_Decomposition/Idep_Simulations/mi_m7_m8.py:358`
  `simulation_wrapper`
- `Partial_Information_Decomposition/Idep_Simulations/mi_m7_m8_notwhiten.py:406`
  `simulation_wrapper`
- `Partial_Information_Decomposition/Idep_Simulations/unique_m7_m8.py:311`
  `sort_m7_m8_results`
- `Partial_Information_Decomposition/Idep_Simulations/unique_m7_m8.py:329`
  `simulation_wrapper`

What they do: run bias simulations for M7/M8 log determinants, mutual
information, and unique-node estimates.

Similarity: exact copied orchestration in the logdet whitened/not-whitened
files; high same-purpose similarity in MI files. The actual statistics differ
between whitened and not-whitened versions, so deleting a whole file would be
risky.

Keep: keep the newest/most complete statistic implementations for each concept,
but move shared orchestration to `Simulation_utils.py` or
`simulation_wrapper.py`.

Erase/merge: erase the exact copied `simulation_wrapper` and
`sort_m7_m8_results` functions from one side after a shared helper exists.
Merge whitened/not-whitened behavior behind a config flag such as
`normalization="whitened"` vs `"raw_covariance"`, then delete the parallel
file copies if tests confirm identical outputs for the old modes.

Do not delete yet: `mi_m7_m8.py` and `mi_m7_m8_notwhiten.py` are similar but
their bias terms and numerator/denominator breakdown differ.

### G13: Config And Filename Utilities

Status: cleaned in `G13_CONFIG_FILENAME_CLEANUP.md`.

Functions:

- `Partial_Information_Decomposition/Idep_Simulations/Simulation_utils.py:905`
  `make_pre_config`
- `Partial_Information_Decomposition/output_utils.py:4` `safe_filename`

What they do: merge simulation YAML config fragments and normalize/sanitize
names for output paths.

Similarity: both old pairs were exact duplicates.

Kept/erased: kept `Simulation_utils.make_pre_config` as the canonical config
merge helper and erased the local copy from
`Partial_Information_Decomposition/Idep_Simulations/idep_sim_runs.py`. Moved
`safe_filename` into lightweight `Partial_Information_Decomposition/output_utils.py`
and erased the local copies from `Simulation_utils.py` and `heatmap_plot.py`.

Important call sites checked: `idep_sim_runs.py:36`, `idep_sim_runs.py:111`,
and `idep_sim_runs.py:137` now use the imported `make_pre_config`.
`Simulation_utils.py:173`, `Simulation_utils.py:174`, and
`heatmap_plot.py:494` now use the imported `safe_filename`.

### G14: Test Fixtures

Status: cleaned in `G14_TEST_FIXTURE_CLEANUP.md`.

Fixtures:

- `uni_tests/conftest.py:6` `dims`
- `uni_tests/conftest.py:11` `random_data`

What they do: build random torch tensors for tests.

Similarity: the old test-local fixtures were duplicated. The shared fixture now
preserves the old behavior: univariate tests receive 1D tensors, while
`test_idep_multivariate.py` receives the larger multivariate dimensions.

Kept/erased: kept one shared `random_data` fixture and one shared `dims`
fixture in `uni_tests/conftest.py`. Erased the local fixture copies from
`uni_tests/test_Idep.py`, `uni_tests/test_PID_util.py`, and
`uni_tests/test_idep_multivariate.py`.

### G15: Multi-Seed Experiment Boilerplate

Status: cleaned in `G15_MULTI_SEED_BOILERPLATE_CLEANUP.md`.

Functions:

- `utils.py:592` `run_configured_multiseed`
- `supression_effect/Suppresed_Encoder.py:37` `get_run_config`
- `Simulations/Both_Unique_Encodrs.py:20` `get_run_config`
- `Simulations/both_unique.py:23` `get_run_config`
- `Simulations/turned_off_unqiue.py:30` `get_run_config`
- `supression_effect/Suppresed_Encoder.py:149` `run_single_seed`
- `Simulations/Both_Unique_Encodrs.py:76` `run_single_seed`
- `Simulations/both_unique.py:153` `run_single_seed`
- `Simulations/turned_off_unqiue.py:133` `run_single_seed`
- `supression_effect/Suppresed_Encoder.py:199` `main`
- `Simulations/Both_Unique_Encodrs.py:105` `main`
- `Simulations/both_unique.py:164` `main`
- `Simulations/turned_off_unqiue.py:290` `main`

What they do: define experiment configs, run one seed, run
`utils.run_multi_seed_experiment`, and save/print seed summaries.

Similarity: the old `main` functions repeated the same summary/save/print
orchestration around experiment-specific `run_single_seed` bodies.

Kept/erased: kept the experiment-specific config dictionaries and
`run_single_seed` logic. Added `utils.run_configured_multiseed` and changed the
four listed `main` functions to delegate the duplicated multi-seed reporting
boilerplate to it.

Do not merge yet: the experiment-specific `run_single_seed` bodies still encode
different datasets/generators and should not be erased until their shared and
unique parts are separated.

## Same Name But Do Not Delete Yet

- `safe_logdet`: `Partial_Information_Decomposition/mi_functions.py:73` is a
  Torch implementation that handles batched tensors; `Partial_Information_Decomposition/Idep_Simulations/premutations_bias_corr.py:85`
  is a NumPy implementation with jitter; `Partial_Information_Decomposition/mi_functions.py:91`
  `np_safe_logdet` is another NumPy version. Keep both tensor/NumPy modes or rename to
  `torch_safe_logdet` and `np_safe_logdet` to make the difference obvious.
- `permutation_null_debias`: `Partial_Information_Decomposition/bias_functions.py:62`
  is config/tensor based and used by PID bias correction; `Partial_Information_Decomposition/Idep_Simulations/premutations_bias_corr.py:24`
  is a standalone NumPy simulation helper. Same idea, different API.
- `calculate_bias` in the M7/M8 simulation files has the same role but different
  mathematical assumptions. Refactor later, but do not delete solely by name.
- `parse_args` and `main` appear many times because they are script entrypoints.
  They are not duplicate logic unless the surrounding wrapper flow is also the
  same.
- `Idep_univariate_gauss.compute_Idep`, `Idep_multivariate_gauss.compute_Idep`,
  jackknife `compute_Idep`, and parallel `compute_Idep` share the mathematical
  structure but differ by dimensionality and correction strategy. Factor common
  math only after tests cover all variants.
- `Simulations/both_unique.py:101` `feature_creation` and
  `Simulations/turned_off_unqiue.py:50` `feature_creation` are same-purpose
  generators but create different experiments. Move them to a generator module;
  do not delete either as a duplicate.

## Other Cleanup Candidates

- `Partial_Information_Decomposition/Dump.py`: unused duplicate/older
  suppression-effect checker. Recommended erase.
- Root `test.py`: contains `rotate_in_random_plane` plus example code that runs
  at import time. Move the function to a real utility or toy example, then
  erase the root script.
- `toy_examples/toy_example_new.py`: mostly a reduced/older version of
  `toy_examples/toy_example.py` with encoding artifacts in text. Recommended
  erase after any unique experiment notes are copied into the main toy example.
- `toy_examples/toy_example_feature_correlation.py`: keep only the
  `generate_correlated_features` idea, preferably as a generator option in the
  main toy example. Then erase the duplicate script.
- `PID/examples/IGscript.R` and `PID/examples/IdepGscript.R`: hard-coded external paths and old
  example workflow. Move to `PID/examples/` or archive.
- `Partial_Information_Decomposition/NotNeedeRightNow/`: the folder name and
  unfinished TODOs make it look like archive/dead code. Move to an explicit
  `archive/` folder or remove from the importable package path.
- Typo-heavy file/function names should be fixed in a separate rename pass:
  `supression_effect`, `Suppresed_Encoder.py`, `suppresion_model.py`,
  `jacknife_Idep_multivariate_gauss.py`, `calcualte_mi`, `permutate_models`,
  `creature_featurs`, and `Theortical_cov_toy_example.py`. Do not mix this with
  deletion because imports will need careful updates.

## Recommended Cleanup Order

1. Add or choose canonical shared helpers for regression metrics,
   commonality analysis, wrapper parsing, and standardization.
2. Replace imports in examples/tests/simulations to use canonical helpers.
3. Delete the now-unused duplicate toy/example/helper definitions.
4. Move Idep demo helpers into tests/examples and remove them from class files.
5. Refactor M7/M8 simulation orchestration after tests cover whitened and
   not-whitened behavior.
6. Archive stale scripts and rename typo-heavy modules in a separate, tested
   pass.
