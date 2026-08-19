# G09 R Idep Client Cleanup

## Summary

G09 is complete. The programmatic R Idep API was moved out of `toy_examples/` and into:

```text
library_wrappers/r_idep_client.py
```

The CLI/report wrapper remains in:

```text
library_wrappers/Idep_R.py
```

## Kept

| Function/Class | Location | What it does | Why keep it |
| --- | --- | --- | --- |
| `RIdePResult` | `library_wrappers/r_idep_client.py:89` | Stores Idep/MMI atoms plus R stdout/stderr. | Useful structured return type for Python callers. |
| `run_idep_from_covariance` | `library_wrappers/r_idep_client.py:144` | Calls R `idepGM(sizes, sigma)` from Python and returns named atoms. | Canonical programmatic API. |
| `run_idep_for_cases` | `library_wrappers/r_idep_client.py:216` | Runs R Idep for multiple named covariance matrices. | Convenience API for comparisons. |
| `atoms_as_ordered_values` | `library_wrappers/r_idep_client.py:228` | Converts atom dictionaries to the canonical atom order. | Keeps ordering logic out of callers. |
| `main` | `library_wrappers/Idep_R.py` | CLI/report entry point. | Kept as the command-line wrapper home. |

## Removed/Moved

| Old item | Replacement | Reason |
| --- | --- | --- |
| `toy_examples/r_idep_wrapper.py` | `library_wrappers/r_idep_client.py` | Programmatic wrapper logic belongs with other library wrappers, not toy examples. |
| Local `_find_rscript` from the toy wrapper | `library_wrappers/wrapper_utils.py:42` | Rscript lookup is shared by G08. |

The old evil-twin comparison script is already not active in the current worktree, so no import shim was needed.

## Verification

Passed syntax parse for `library_wrappers/r_idep_client.py`.

Passed smoke check:

```text
atoms_as_ordered_values(dict(zip(ATOMS, range(4)))) -> [0.0, 1.0, 2.0, 3.0]
```

Full R execution was not run because `Rscript` is not installed in this environment.

`git diff --check` passed.
