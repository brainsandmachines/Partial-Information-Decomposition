---
name: function-registry-reuse
description: Prevent duplicate functions and keep FUNCTION_REGISTRY.md synchronized in this project. Use before creating, modifying, moving, renaming, or deleting functions; when refactoring duplicated logic; when deciding whether to reuse an existing helper; or when the user asks to update, refresh, rebuild, or check FUNCTION_REGISTRY.md.
---

# Function Registry Reuse

## Purpose

Prevent duplicated logic by checking `FUNCTION_REGISTRY.md` before creating or changing functions. Prefer existing project functions when their purpose, behavior, inputs, outputs, assumptions, or mathematical role match the task.

## Core Rule

Before creating a function, ask: "Does this function already exist in `FUNCTION_REGISTRY.md`, or can an existing function be reused?"

Only create a new function when the answer is clearly no.

## Before Creating A Function

1. Open and inspect `FUNCTION_REGISTRY.md`.
2. Search for similar names, purposes, inputs, outputs, assumptions, mathematical roles, or implementation logic.
3. If a relevant function appears in the registry, inspect the actual source code.
4. Verify the function is correct for the current use case.
5. Check assumptions carefully: input order, tensor shape, numpy versus torch, device and dtype, raw covariance versus whitened covariance, samples versus covariance matrices, biased versus unbiased estimators, log base, PID convention, and source-target ordering.
6. Reuse or wrap the existing function if it is appropriate.
7. Create a new function only when no existing function can be safely reused, wrapped, or extended without changing its meaning.

## Using Existing Functions

Do not assume a registry match is correct automatically. Verify that:

- The implementation matches the registry description.
- The docstring is consistent with the code.
- The assumptions match the current task.
- The output is the needed quantity.
- The function does not silently use a different convention.

If the function appears incorrect, outdated, badly documented, or risky, report the issue clearly before using or changing it.

## Modifying Existing Functions

Ask the user before modifying an existing function. Explain:

- Which function would change and where it is located.
- Why the current function is insufficient.
- What change is proposed.
- Which files or callers may be affected.
- Whether the change is backward compatible.

Do not change an existing function without approval.

## Placing New Functions

Keep task-specific functions close to the script or experiment that uses them. Put reusable functions in the appropriate utility module:

- General helpers: `utils.py` or the nearest existing utility file.
- PID helpers: an existing PID utility file or a clearly scoped PID utility.
- Covariance, whitening, Gaussian MI, or log-determinant helpers: the existing covariance, Gaussian, or MI utility module.
- Plotting helpers: plotting utilities.
- Simulation bookkeeping, CSV writing, and experiment loops: simulation or experiment utilities.

Ask before placing a function when the right location is unclear.

## Registry Updates

Whenever a function is created, modified, moved, renamed, or deleted, update `FUNCTION_REGISTRY.md`.

Include the function name, file path, purpose, inputs, outputs, important assumptions, whether it is general-purpose or task-specific, related functions, and relevant convention notes.

When the user asks to update, refresh, rebuild, or check the registry:

1. Re-scan the project for function definitions.
2. Compare code against `FUNCTION_REGISTRY.md`.
3. Add missing functions.
4. Update changed signatures, paths, purposes, or assumptions.
5. Mark or remove functions that no longer exist.
6. Preserve useful manual notes that remain correct.
7. Report what was added, changed, or removed.

## Project-Specific Warning

Similar function names may still hide different mathematical conventions. Always verify:

- Variable order, such as `[X1, X2, T]`, `[T, X1, X2]`, or `[M, X, Y]`.
- Raw covariance versus whitened covariance.
- Covariance versus precision matrix.
- Sample data versus covariance input.
- Gaussian versus discrete assumptions.
- Biased versus unbiased covariance estimation.
- Natural logarithm versus log base 2.
- BROJA, Idep, Gaussian PID, or other PID conventions.
- Source-target ordering.

If conventions differ, do not treat functions as interchangeable.
