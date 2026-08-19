# G01 Regression Helper Cleanup

Generated: 2026-06-02

This document records the source changes made for duplicate group G01 from
`DUPLICATE_FUNCTION_AUDIT.md`.

## Kept

- `encoding_model/regression_metrics.py:6` `compute_ols_cv_r2`
- `encoding_model/regression_metrics.py:32` `compute_ridge_cv_r2`
- `encoding_model/regression_metrics.py:64` `compute_r2`
- `encoding_model/regression_metrics.py:87` `compute_lasso_cv_r2`
- `encoding_model/encoding_utils.py:20` re-export of the four helpers above
- `Partial_Information_Decomposition/PID_util.py:33` `compute_ridge_cv_r2`

Reason: `encoding_model/regression_metrics.py` is now the canonical lightweight
home for the shared NumPy/scikit-learn scoring helpers. This avoids importing
the heavy plotting/fMRI dependencies in `encoding_model/encoding_utils.py` just
to compute regression scores.

The `encoding_utils.py` re-export is kept so older imports in encoding code keep
working. `PID_util.compute_ridge_cv_r2` was kept because PID simulation code
currently depends on its different return shape and torch-to-NumPy conversion
behavior.

## Erased

Removed duplicate local helper definitions from:

- `toy_examples/toy_example.py`
  - `compute_ols_cv_r2`
  - `compute_ridge_cv_r2`
  - `compute_r2`
  - `compute_lasso_cv_r2`
- `toy_examples/toy_example_new.py`
  - `compute_ols_cv_r2`
  - `compute_ridge_cv_r2`
  - `compute_r2`
- `toy_examples/toy_example_feature_correlation.py`
  - `compute_ols_cv_r2`
  - `compute_ridge_cv_r2`
  - `compute_r2`
- `encoding_model/encoding_utils.py`
  - removed the in-file implementations of the four helpers and replaced them
    with imports from `encoding_model.regression_metrics`

Reason: these functions duplicated the same regression scoring logic. Keeping
one implementation removes drift risk between examples and encoding code.

## Changed

- Added `encoding_model/regression_metrics.py` as the canonical implementation
  module for G01.
- Added `return_model=False` to `compute_ols_cv_r2`, `compute_ridge_cv_r2`, and
  `compute_r2`.
- Default behavior is unchanged: existing callers still receive only the R2
  score.
- When `return_model=True`, those helpers return `(model, score)`. This
  preserves the previous behavior of `toy_examples/toy_example.py`, which uses
  model coefficients in its `commonality_analysis` output.
- Left `compute_lasso_cv_r2` returning `(model, score)`, matching its previous
  toy-example behavior.
- Moved the scikit-learn imports inside the helper functions so importing the
  helper module does not fail before a helper is actually called.
- Updated `toy_examples/toy_example.py:7`,
  `toy_examples/toy_example_new.py:7`, and
  `toy_examples/toy_example_feature_correlation.py:7` to import helpers from
  `encoding_model.regression_metrics`.
- Left existing compatibility imports in `encoding_model/suppresion_model.py:17`
  and `encoding_model/suppression_core.py:13` unchanged because
  `encoding_utils.py` still re-exports the helpers.

## Remaining For Later Groups

- The three toy files still contain duplicate `commonality_analysis` logic.
  That belongs to G02.
- The toy experiment runners still overlap heavily. That belongs to G03.

## Verification

- Remaining definitions check shows only the canonical implementations in
  `encoding_model/regression_metrics.py` plus the intentionally separate
  `Partial_Information_Decomposition/PID_util.py:33` helper.
- Syntax parse check passed for:
  - `encoding_model/regression_metrics.py`
  - `encoding_model/encoding_utils.py`
  - `toy_examples/toy_example.py`
  - `toy_examples/toy_example_new.py`
  - `toy_examples/toy_example_feature_correlation.py`
- Import smoke check passed for:
  - `encoding_model.regression_metrics`
  - `toy_examples.toy_example_new`
  - `toy_examples.toy_example_feature_correlation`
- Runtime function-call smoke check could not complete in the current
  environment because importing scikit-learn triggers `pyarrow`, `numexpr`, and
  `bottleneck` binaries compiled against NumPy 1.x while this environment has
  NumPy 2.2.4.
