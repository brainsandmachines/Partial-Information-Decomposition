# G10 IGFuns Core Cleanup

## Summary

G10 is complete. The public R APIs remain available:

```text
IG_GaussM_PQR
IG_GaussM_Dat
IG_GaussU_pqr
```

The duplicated multivariate implementation was factored into one internal helper:

```text
PID/IGFuns.R:3 IG_GaussM_Core
```

## Kept

| Function | Location | What it does | Why keep it |
| --- | --- | --- | --- |
| `IG_GaussM_Core` | `PID/IGFuns.R:3` | Shared multivariate IG PID computation once P/Q/R blocks are available. | Removes duplicate core logic from the public multivariate APIs. |
| `IG_GaussM_PQR` | `PID/IGFuns.R:119` | Public API for already-whitened P/Q/R blocks. | Different input contract from covariance-data API. |
| `IG_GaussM_Dat` | `PID/IGFuns.R:126` | Public API for a full covariance matrix; derives P/Q/R and calls the core. | Different input contract from P/Q/R API. |
| `IG_GaussU_pqr` | `PID/IGFuns.R:166` | Scalar trivariate IG PID function. | Related but dimension-specific, so it remains separate. |

## Moved

| Old path | New path | Reason |
| --- | --- | --- |
| `PID/IGscript.R` | `PID/examples/IGscript.R` | Hard-coded example script, not core library code. |
| `PID/IdepGscript.R` | `PID/examples/IdepGscript.R` | Hard-coded example script, not core library code. |

## Verification

`PID/` is a nested git repository, so its status is separate from the outer thesis repo:

```text
M  IGFuns.R
D  IGscript.R
D  IdepGscript.R
?? examples/
```

R syntax execution was not run because `Rscript` is not installed in this environment:

```text
/bin/bash: line 1: Rscript: command not found
```

The Python-side `git diff --check` passed.

## Recommendation

Keep `IG_GaussM_Core` internal and continue exposing only the three public functions for callers. If the old examples are needed, update their hard-coded paths before treating them as active runnable scripts.
