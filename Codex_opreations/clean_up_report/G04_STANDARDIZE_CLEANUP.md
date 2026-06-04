# G04 Standardize Helper Cleanup

Generated: 2026-06-04

This document records the source changes made for duplicate group G04 from
`DUPLICATE_FUNCTION_AUDIT.md`.

## Kept

- `Partial_Information_Decomposition/PID_util.py:377` `standardize`
- `utils.py:70` `standardize_np`

Reason: `PID_util.standardize` is the canonical torch tensor helper used by PID
code. `utils.standardize_np` is the canonical NumPy helper for simulations that
work with NumPy arrays before converting to torch tensors.

## Erased

Removed duplicate local standardization helpers from:

- `supression_effect/supp_gauss_multivariate.py`
  - removed local torch `standardize`
- `Simulations/turned_off_unqiue.py`
  - removed local NumPy-backed `standardize`

Reason: the suppression-effect helper was an exact duplicate of
`PID_util.standardize`, and the simulation helper was a NumPy standardizer with
a misleading torch type annotation.

## Changed

- Added `utils.py:70` `standardize_np(X, eps=1e-12)`.
- Updated `Simulations/turned_off_unqiue.py:91`, `:92`, `:93`, `:146`,
  `:147`, and `:148` to call `standardize_np`.
- Removed stale commented-out `standardize(...)` lines from
  `supression_effect/supp_gauss_multivariate.py`.

## Intentionally Left Alone

- `Partial_Information_Decomposition/PID_util.py:377` was not moved because PID
  torch utilities already live in that file.
- Other cleanup opportunities in `Simulations/turned_off_unqiue.py`, such as
  duplicate imports and beta-output code, were left for later groups because
  they are not part of G04.

## Verification

- Remaining-definition check shows:
  - one torch `standardize`, in `Partial_Information_Decomposition/PID_util.py`
  - one NumPy `standardize_np`, in `utils.py`
- Syntax parse check passed for:
  - `utils.py`
  - `Simulations/turned_off_unqiue.py`
  - `supression_effect/supp_gauss_multivariate.py`
  - `Partial_Information_Decomposition/PID_util.py`
- `standardize_np` smoke check passed by extracting and executing just that
  function body, avoiding full `utils.py` import-time dependencies.
