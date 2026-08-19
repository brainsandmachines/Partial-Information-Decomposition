# G02 Commonality Analysis Cleanup

Generated: 2026-06-02

This document records the source changes made for duplicate group G02 from
`DUPLICATE_FUNCTION_AUDIT.md`.

## Kept

- `encoding_model/commonality.py:26` `commonality_analysis`
- `encoding_model/suppression_core.py:8` import/re-export of
  `commonality_analysis`

Reason: `encoding_model/commonality.py` is now the canonical lightweight home
for commonality analysis. `suppression_core.py` still exposes the same name for
older callers that import `commonality_analysis` from the suppression module or
through `encoding_model/suppresion_model.py`.

## Erased

Removed duplicate local `commonality_analysis` definitions from:

- `encoding_model/suppression_core.py`
- `toy_examples/toy_example.py`
- `toy_examples/toy_example_new.py`
- `toy_examples/toy_example_feature_correlation.py`

Also removed the beta-output path from `toy_examples/toy_example.py`:

- Removed `betas_X1`, `betas_X2`, and `betas_X12` from the commonality result
  path by deleting the local beta-returning implementation.
- Removed `plot_coefficients`, because it only plotted beta outputs that are no
  longer produced.
- Simplified `run_all_methods` and `gauss_simple_example` to return/use only
  the variance decomposition dictionary.

## Changed

- Added `encoding_model/commonality.py` with one shared non-beta implementation.
- Standardized result keys to:
  - `R²_X1`
  - `R²_X2`
  - `R²_X12`
  - `unique_X1`
  - `unique_X2`
  - `common`
  - `unexplained`
- Kept `lasso_cv` support without beta outputs by discarding fitted models and
  retaining only the score.
- Added `scale_by_target_variance=False` to the shared function.
  `toy_examples/toy_example_new.py:53` passes
  `scale_by_target_variance=True` to preserve its previous variance-scaled
  component behavior.
- Added `**_ignored_kwargs` to the shared function so older callers that pass
  unused metadata such as `snr` do not break.
- Updated imports:
  - `toy_examples/toy_example.py:7`
  - `toy_examples/toy_example_new.py:7`
  - `toy_examples/toy_example_feature_correlation.py:7`
  - `encoding_model/suppression_core.py:8`
- Updated `utils.py:747` so histogram grouping uses the shared
  `CA_R²_X1`/`CA_R²_X2`/`CA_R²_X12` and `CA_unique_X1`/`CA_unique_X2` result
  names.

## Intentionally Left Alone

- `utils.extract_all_components(..., betas_dict=None)` was not removed because
  separate simulation code still references that optional argument.
- PID/R-wrapper names such as `unique_X1` and `unique_X2` were not changed
  because those names belong to PID atom conventions, not the duplicated
  commonality analysis functions cleaned up in G02.

## Verification

- Remaining-definition check shows only one `def commonality_analysis`, in
  `encoding_model/commonality.py`.
- Syntax parse check passed for all G02-touched Python files.
- Import smoke check passed for:
  - `encoding_model.commonality`
  - `toy_examples.toy_example_new`
  - `toy_examples.toy_example_feature_correlation`
- Commonality math smoke check passed by monkeypatching the shared score
  helpers, avoiding the local scikit-learn/NumPy binary incompatibility.
- Direct scikit-learn runtime checks are still blocked by the current
  environment issue already documented in `G01_REGRESSION_HELPER_CLEANUP.md`.
