# G03 Suppression Model Cleanup

Generated: 2026-06-02

This document records the source changes made for duplicate group G03 from
`DUPLICATE_FUNCTION_AUDIT.md`.

## Kept

- `encoding_model/suppression_core.py:63` `permutate_models`
- `encoding_model/suppression_core.py:150` `run_all_methods`
- `toy_examples/suppression_toy_runner.py:80` `run_toy_experiment`
- `toy_examples/suppression_toy_runner.py:128` `run_all_toy_methods`
- `toy_examples/suppression_toy_runner.py:172` `run_default_factorial_scenarios`

Reason: `encoding_model/suppression_core.py` remains the canonical suppression
pipeline for encoding/fMRI code. The new `toy_examples/suppression_toy_runner.py`
is the canonical home for the repeated toy-example experiment skeleton.

## Erased

Removed duplicate local implementations from:

- `toy_examples/toy_example.py`
  - removed local `permutate_models`
- `toy_examples/toy_example_new.py`
  - removed local `run_experiment`
  - removed local `run_all_methods`
- `toy_examples/toy_example_feature_correlation.py`
  - moved `generate_correlated_features` into
    `toy_examples/suppression_toy_runner.py:12`
  - removed local `run_experiment`
  - removed local `run_all_methods`

Reason: these functions repeated the same suppression/commonality experiment
skeleton with only small differences in source generation.

## Changed

- Added `toy_examples/suppression_toy_runner.py`.
- `toy_examples/toy_example.py:11` now imports canonical
  `permutate_models` from `encoding_model.suppression_core`.
- `toy_examples/toy_example.py:130` keeps its specialized `run_experiment`
  because it still supports nonstandard modes:
  - `simple`
  - `permuted`
  - `only_unq2_zero`
  - `unq2_zero_with_red_unq1_syn`
- `toy_examples/toy_example.py:226` renamed the old generic
  `run_all_methods` wrapper to `run_ridge_toy_method`, because it only runs
  ridge-CV for that specialized toy.
- `toy_examples/toy_example_new.py:6` now imports and runs
  `run_default_factorial_scenarios(..., experiment_kind=SPLIT_SIGNAL)`.
- `toy_examples/toy_example_feature_correlation.py:6` now imports and runs
  `run_default_factorial_scenarios(..., experiment_kind=FEATURE_CORRELATION)`.
- `supression_effect/supp_gauss_multivariate.py:13` now imports
  `commonality_analysis` directly from `encoding_model.commonality`, and only
  imports the specialized `run_experiment` from `toy_examples.toy_example`.
- `Simulations/both_unique.py:9` now imports `commonality_analysis` directly
  from `encoding_model.commonality`.

## Intentionally Left Alone

- `toy_examples/toy_example.py:130` `run_experiment` was not removed because it
  is a broader special-case generator, not just the duplicated two-source toy
  skeleton from `toy_example_new.py` and `toy_example_feature_correlation.py`.
- `supression_effect/gauss_unvariate.py` still imports from a non-existent
  `examples.toy_example` package. That looks like a separate old/broken import
  path and was not part of G03's duplicated function cluster.

## Verification

- Remaining-definition check shows:
  - one canonical `permutate_models`, in `encoding_model/suppression_core.py`
  - one core `run_all_methods`, in `encoding_model/suppression_core.py`
  - toy runner functions centralized in
    `toy_examples/suppression_toy_runner.py`
- Syntax parse check passed for:
  - `toy_examples/suppression_toy_runner.py`
  - `toy_examples/toy_example.py`
  - `toy_examples/toy_example_new.py`
  - `toy_examples/toy_example_feature_correlation.py`
  - `encoding_model/suppression_core.py`
  - `supression_effect/supp_gauss_multivariate.py`
  - `Simulations/both_unique.py`
- Import smoke check passed for:
  - `toy_examples.suppression_toy_runner`
  - `toy_examples.toy_example_new`
  - `toy_examples.toy_example_feature_correlation`
- Toy runner smoke check passed by monkeypatching the shared score helpers,
  avoiding the local scikit-learn/NumPy binary incompatibility.
- Direct import of `toy_examples.toy_example` is still blocked in this
  environment by missing `nilearn`, through `encoding_model.encoding_utils`.
