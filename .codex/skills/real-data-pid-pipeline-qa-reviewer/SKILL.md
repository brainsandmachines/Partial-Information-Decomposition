---
name: real-data-pid-pipeline-qa-reviewer
description: "Review, debug, inspect, and QA-check code for the real-data PID pipeline against the Real_Data_PID_Pipeline_Plan. Use when Codex is asked to review pipeline code, check PID-agnostic design, inspect data/preprocessing/feature-manipulation/estimation/PID/results layers, verify schemas and PID identities, propose deterministic dummy examples, or produce a QA report. This skill is review-first: do not modify production code unless the user explicitly asks for fixes after the review."
---

# Real Data PID Pipeline QA Reviewer

## Role

Act as a careful reviewer and testing assistant for the real-data PID pipeline. Review code against the Real_Data_PID_Pipeline_Plan, identify mismatches and likely bugs, suggest improvements, and propose or create small deterministic dummy examples that the user can run.

Do not modify production code automatically. First explain the issue and suggest a fix plan. Modify code only if the user explicitly asks to apply a fix after seeing the review.

## Primary Reference

Find and read the Real_Data_PID_Pipeline_Plan before judging design conformance. If the plan is not obvious, search the repo for `Real_Data_PID_Pipeline_Plan`, `real data pid pipeline`, `PID pipeline plan`, and related filenames. If the plan cannot be found, state that clearly and mark plan-dependent judgments as unclear.

## Pipeline Assumptions

Use the bivariate PID structure:

- Target `T`: neural responses.
- Source `X1`: representation or prediction from model 1.
- Source `X2`: representation or prediction from model 2.

Support both representation-level PID and prediction-level PID:

- Representation-level PID: `X1` and `X2` are model features or transformed model features, and `T` is neural activity.
- Prediction-level PID: `X1` and `X2` are cross-validated predictions of neural responses from two encoding models, and `T` is measured neural activity.

Prefer prediction-level PID when feasible because `X1`, `X2`, and `T` live in comparable neural-response space. Still require the pipeline to remain general enough to support both modes.

## Review Workflow

1. Identify the pipeline folder and list relevant files.
2. Map each file to its intended pipeline layer.
3. Check whether each file has one clear responsibility.
4. Flag responsibility mixing that violates the plan.
5. Check imports and dependencies.
6. Check whether existing utilities are reused instead of duplicating logic.
7. Check the input and output contract of every important function.
8. Check whether shapes are documented and enforced.
9. Check whether `T`, `X1`, and `X2` ordering is consistent everywhere.
10. Check whether covariance block ordering is consistent everywhere.
11. Check whether values are in bits or nats and whether the unit is stored in metadata.
12. Check whether random seeds are controlled in dummy examples and stochastic procedures.
13. Check whether cross-validation leakage is possible.
14. Check whether the pipeline can run with an external dummy PID function.
15. Check whether diagnostics and warnings are informative.
16. Produce a QA summary with pass/fail/unclear status for each major point.

## Layer Checks

### Layer 1: Data Loading

Verify that this layer loads and organizes raw neural data, DNN model features, prediction files, and metadata.

Flag PCA, regression, MI, PID, covariance estimation, final analysis, or other downstream analysis in this layer.

Expected outputs include neural responses, model features or predictions, image or trial metadata, subject identifiers, brain-area identifiers, session identifiers, model identifiers, and layer identifiers when relevant.

### Layer 2: Preprocessing

Verify that this layer cleans and aligns data before analysis.

Allowed responsibilities include sample alignment, z-scoring, bad-unit removal, repeat averaging, and missing-value handling.

Expected reporting includes sample counts before and after cleaning, feature counts before and after cleaning, excluded units or images, and whether z-scoring or repeat averaging was applied.

### Layer 3: Feature Manipulation

Verify that optional transformations are modular and separated from the rest of the pipeline.

Expected transform types include identity transforms, PCA, random projections, CCA-like methods, or future transforms. Check that transforms can be replaced without changing the rest of the pipeline.

Flag possible leakage if PCA, scaling, regression, or any learned transform is fitted on data that should be held out. Check whether learned transforms are fitted inside cross-validation when relevant.

Expected outputs include transform name, number of components, explained variance, effective dimension before and after transformation, and whether the transform was fitted inside cross-validation.

### Layer 4: Estimation And Mutual Information

Verify that covariance estimation and classical information-theoretic quantities are computed independently of the chosen PID definition.

Flag MI quantities hidden only inside the PID callable unless they are also stored separately.

Recommended MI and diagnostic outputs:

- `I(T;X1)`
- `I(T;X2)`
- `I(T;X1,X2)`
- `I(X1;X2)`
- conditional MI values when relevant
- co-information
- covariance condition number
- minimum covariance eigenvalue

### Layer 5: PID Callable

Verify that the pipeline receives the PID function from outside and calls it through a standardized interface:

```python
pid_func(target, source_1, source_2, *, estimator_context)
```

Expected output is a dictionary-like object containing:

- `red`
- `unq1`
- `unq2`
- `syn`
- `method`
- `extra`

Require the pipeline to depend only on `red`, `unq1`, `unq2`, and `syn`. Store all method-specific information under `extra`; the rest of the pipeline must not require it.

Flag fixed internal PID definitions. Do not accept MMI as the default PID definition. Do not assume Idep is the active scientific choice. Require PID-agnostic design that can support Gaussian BROJA, flow-based BROJA, delta-PID, Idep, or another definition.

### Layer 6: Results And Diagnostics

Verify that results, metadata, sanity checks, uncertainty estimates, and diagnostic flags are saved in a structured format suitable for plotting and later analysis.

Prefer long-format output, or output that can be easily converted into long-format rows.

Recommended diagnostics include PID identity errors, negative-component flags, singular-covariance flags, bootstrap intervals if available, permutation baselines if available, predictive performance metrics if available, condition number, minimum eigenvalue, `n_samples`, `dim_T`, `dim_X1`, `dim_X2`, `n_samples / total_dimension`, and convergence status for optimization-based PID.

## Required PID Identity Checks

For every run, verify the bivariate PID identities:

```text
I(T;X1,X2) = red + unq1 + unq2 + syn
I(T;X1)    = red + unq1
I(T;X2)    = red + unq2
```

Flag any run where an identity does not close within tolerance. Also flag PID components that are negative beyond numerical tolerance.

Check that `unq1` always means the unique information of `X1` about `T` relative to `X2`, and that `unq2` always means the unique information of `X2` about `T` relative to `X1`. Be especially careful about accidental swapping of `X1` and `X2`.

## Minimal Result Schema

Verify that each analysis produces one structured row or dictionary with these conceptual groups:

- `metadata`: subject, session, area, model_1, model_2, layer_1, layer_2, analysis_mode, preprocessing choices, transform choices, estimator choices, PID method, and random seed when relevant.
- `pid`: red, unq1, unq2, syn.
- `mi`: I_T_X1, I_T_X2, I_T_X1X2, I_X1_X2, and co_info when relevant.
- `normalized`: red_frac, unq1_frac, unq2_frac, syn_frac when meaningful.
- `diagnostics`: pid_identity_error_joint, pid_identity_error_x1, pid_identity_error_x2, min_eigenvalue_cov, condition_number_cov, n_samples, dim_T, dim_X1, dim_X2, flags, and warnings.

## Dummy Examples

For every file in the pipeline folder, create or propose a small deterministic dummy example that the user can run at any time.

Keep examples minimal, fast, and independent of real neural data. Use synthetic arrays with simple known shapes. Use fixed random seeds for stochastic examples.

When possible, include a dummy external PID function that follows the standard interface and returns a dictionary with `red`, `unq1`, `unq2`, `syn`, `method`, and `extra`.

## Report Format

Lead with findings, ordered by severity. Include file and line references when available.

Use this concise structure:

1. Findings: bugs, plan violations, leakage risks, schema gaps, identity-check failures, and unclear contracts.
2. Pass/fail/unclear checklist: one status per major pipeline layer and cross-cutting concern.
3. Dummy examples: commands or snippets the user can run, plus expected behavior.
4. Suggested fix plan: describe changes but do not apply them unless explicitly requested.

If no issues are found, say so clearly and mention remaining test gaps or residual risk.
