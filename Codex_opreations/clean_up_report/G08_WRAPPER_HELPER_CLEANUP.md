# G08 Wrapper Helper Cleanup

## Summary

G08 is complete. Shared Python wrapper helpers now live in:

```text
library_wrappers/wrapper_utils.py
```

## Kept

| Function | Location | What it does | Why keep it |
| --- | --- | --- | --- |
| `parse_sizes` | `library_wrappers/wrapper_utils.py:11` | Parses `n_source1,n_source2,n_target` CLI size strings. | Canonical size parser for active wrappers. |
| `csv_shape` | `library_wrappers/wrapper_utils.py:22` | Validates a numeric no-header CSV and returns `(rows, columns)`. | Shared by wrappers that need covariance/block shape checks. |
| `find_rscript` | `library_wrappers/wrapper_utils.py:42` | Resolves explicit `Rscript` paths/names or finds `Rscript` on `PATH`. | Canonical Rscript discovery helper. |

## Removed Local Copies

| File | Removed local helper(s) | Replacement |
| --- | --- | --- |
| `library_wrappers/IG_R.py` | `parse_sizes`, `csv_shape`, `find_rscript` | Imports from `wrapper_utils.py`. |
| `library_wrappers/Idep_R.py` | `parse_sizes`, `csv_shape`, `find_rscript` | Imports from `wrapper_utils.py`. |
| `library_wrappers/check_evil_twin_all.py` | `find_rscript` | Imports from `wrapper_utils.py`. |
| `library_wrappers/Delta_PID.py` | `parse_sizes` | Imports from `wrapper_utils.py`. |
| `library_wrappers/Tilde_PID.py` | `parse_sizes` | Imports from `wrapper_utils.py`. |

The imports use a relative import with a script-style fallback, so the wrappers still work both when imported as modules and when run directly.

## Intentionally Left Alone

Wrapper-specific example/default argument builders were not merged. They look similar, but each wrapper has different fields and defaults.

## Verification

Passed syntax parse for:

```text
library_wrappers/wrapper_utils.py
library_wrappers/IG_R.py
library_wrappers/Idep_R.py
library_wrappers/check_evil_twin_all.py
library_wrappers/Delta_PID.py
library_wrappers/Tilde_PID.py
library_wrappers/r_idep_client.py
```

Passed smoke checks:

```text
parse_sizes('1, 2, 3') -> (1, 2, 3)
csv_shape(library_wrappers/evil_twin_whitened_correlation_1_1_1.csv) -> (3, 3)
module imports for IG_R, Idep_R, check_evil_twin_all, Delta_PID, Tilde_PID
```

Search check after cleanup shows only the canonical helper definitions in `wrapper_utils.py`.

`git diff --check` passed.
