# Codex Implementation Instructions: Lorenz Gaussian gPID Bias Correction

## Objective

Implement the finite-sample bias-correction procedure for the **two-source Gaussian BROJA/gPID** described by Lorenz et al. (2026), *Sampling bias corrections for discrete and Gaussian partial information decompositions*.

The implementation must:

1. Preserve the existing Gaussian BROJA/gPID estimator.
2. Add the Lorenz correction as a separate wrapper/module.
3. Correct:
   - \(I(T;X_1)\)
   - \(I(T;X_2)\)
   - \(I(T;X_1,X_2)\)
   - synergy
4. Reconstruct redundancy and both unique-information terms from the corrected quantities.
5. Use:
   - exact Gaussian mutual-information bias correction;
   - Gaussian parametric resampling;
   - target-shuffle subtraction;
   - an equally weighted merged correction.
6. Keep the original Venkatesh correction available for comparison, but **do not use it inside the Lorenz implementation**.

Do not replace BROJA/gPID with MMI or Idep.

---

## Scientific basis

For two sources, the PID identities are

\[
I(T;X_1,X_2)=R+U_1+U_2+S,
\]

\[
I(T;X_1)=R+U_1,
\]

\[
I(T;X_2)=R+U_2.
\]

Lorenz et al. correct synergy, the joint mutual information, and the two single-source mutual informations. They then reconstruct \(R,U_1,U_2\) from the PID identities so that the corrected decomposition closes exactly.

For the Gaussian case, Lorenz et al. use:

- \(W=20\) parametric-resampling iterations;
- \(V=20\) shuffle iterations;
- equal weighting of the two bias estimates:

\[
b_{\mathrm{merged}}
=
\frac12 b_{\mathrm{resampling}}
+
\frac12 b_{\mathrm{shuffle}}.
\]

The implementation must expose \(W\), \(V\), and the merge weight as configurable arguments, but default to these values.

---

## Scope of the first implementation

Implement the correction for sample-level arrays

\[
T\in\mathbb{R}^{N\times d_T},
\qquad
X_1\in\mathbb{R}^{N\times d_1},
\qquad
X_2\in\mathbb{R}^{N\times d_2}.
\]

Version 1 should support:

- two sources only;
- jointly Gaussian estimation through the sample covariance;
- covariance computed with divisor \(N-1\);
- information returned in bits;
- NumPy arrays and, optionally, PyTorch tensors converted safely to NumPy;
- a user-supplied gPID callable that accepts a covariance matrix.

Version 1 should not silently support shrinkage covariance estimators. The exact Wishart/Goodman mutual-information correction is not exact after Ledoit-Wolf, OAS, ridge shrinkage, or other nonlinear covariance regularization.

If shrinkage is requested, raise `NotImplementedError` or emit a clear error explaining that the Lorenz/Goodman implementation currently assumes the ordinary sample covariance.

---

## Required public API

Create a module such as:

```text
pid_bias/lorenz_gaussian.py
```

Implement the following public functions.

```python
def goodman_logdet_bias(
    dimension: int,
    n_samples: int,
    *,
    covariance_ddof: int = 1,
) -> float:
    """Expected plugin log-determinant error in natural-log units."""
```

```python
def gaussian_mi_bias_bits(
    dim_a: int,
    dim_b: int,
    n_samples: int,
    *,
    covariance_ddof: int = 1,
) -> float:
    """Finite-sample bias of Gaussian plugin MI in bits."""
```

```python
def gaussian_mi_from_cov(
    covariance: np.ndarray,
    indices_a: np.ndarray | slice | list[int],
    indices_b: np.ndarray | slice | list[int],
    *,
    base: float = 2.0,
) -> float:
    """Gaussian mutual information calculated from covariance blocks."""
```

```python
def estimate_resampling_synergy_bias(
    covariance_hat: np.ndarray,
    *,
    n_samples: int,
    dims: tuple[int, int, int],
    gpid_from_cov: Callable,
    n_resamples: int = 20,
    random_state: int | np.random.Generator | None = None,
    gpid_kwargs: dict | None = None,
) -> dict:
    """Estimate synergy bias by Gaussian parametric resampling."""
```

```python
def estimate_shuffle_synergy_bias(
    target: np.ndarray,
    source_1: np.ndarray,
    source_2: np.ndarray,
    *,
    gpid_from_cov: Callable,
    n_shuffles: int = 20,
    random_state: int | np.random.Generator | None = None,
    gpid_kwargs: dict | None = None,
) -> dict:
    """Estimate synergy bias by shuffling target rows."""
```

```python
def lorenz_gaussian_bias_corrected_pid(
    target: np.ndarray,
    source_1: np.ndarray,
    source_2: np.ndarray,
    *,
    gpid_from_cov: Callable,
    n_resamples: int = 20,
    n_shuffles: int = 20,
    merge_weight: float = 0.5,
    covariance_ddof: int = 1,
    random_state: int | np.random.Generator | None = None,
    gpid_kwargs: dict | None = None,
    negative_tolerance: float = 1e-8,
) -> dict:
    """Return plugin, bias, corrected PID, and diagnostics."""
```

The ordering convention must be explicit and consistent:

```text
Z = [T, X1, X2]
dims = (dT, d1, d2)
```

---

## Input validation

Before any computation:

1. Convert inputs to `float64`.
2. Require all arrays to be two-dimensional.
3. Require equal sample counts.
4. Reject NaN and infinite values.
5. Require \(N\ge 2\).
6. Require each variable to have at least one feature.
7. Check that the total sample covariance is positive definite within a numerical tolerance.
8. Record:
   - minimum eigenvalue;
   - maximum eigenvalue;
   - condition number;
   - matrix rank.
9. Reject or clearly flag cases in which the covariance is singular.

For the exact log-determinant correction, require enough degrees of freedom for every covariance block whose determinant is used. With mean subtraction and `ddof=1`, use

\[
\nu=N-1.
\]

Require

\[
\nu \ge p
\]

for each relevant \(p\)-dimensional covariance block. At minimum, validate the full dimension

\[
p_{\mathrm{total}}=d_T+d_1+d_2.
\]

Do not add jitter silently. If an optional jitter mechanism is later added, it must default to zero and its use must be reported in the output diagnostics.

---

## Step 1: Estimate the original covariance and plugin quantities

Concatenate the observations:

```python
Z = np.concatenate([target, source_1, source_2], axis=1)
cov_hat = np.cov(Z, rowvar=False, ddof=1)
```

Symmetrize only to remove floating-point asymmetry:

```python
cov_hat = 0.5 * (cov_hat + cov_hat.T)
```

Do not alter the covariance otherwise.

Run the existing Gaussian BROJA/gPID estimator:

```python
pid_plugin = gpid_from_cov(
    cov_hat,
    dims=(dT, d1, d2),
    **gpid_kwargs,
)
```

The wrapper must accept either of these key conventions and normalize internally:

```text
syn / synergy
red / redundancy
unq1 / unique_1
unq2 / unique_2
```

Require the plugin result to contain synergy. Prefer also storing the plugin values for all four atoms.

Calculate the three Gaussian plugin mutual informations:

\[
\widehat I_1=I_{\widehat\Sigma}(T;X_1),
\]

\[
\widehat I_2=I_{\widehat\Sigma}(T;X_2),
\]

\[
\widehat I_{12}=I_{\widehat\Sigma}(T;[X_1,X_2]).
\]

Use stable log-determinants through `numpy.linalg.slogdet`. Reject covariance blocks whose determinant sign is not positive.

For two Gaussian vectors \(A,B\),

\[
I(A;B)
=
\frac12
\log_2
\frac{|\Sigma_A||\Sigma_B|}{|\Sigma_{AB}|}.
\]

---

## Step 2: Implement the exact Gaussian MI bias correction

Assume the centered sample covariance is computed with denominator \(N-1\). Define

\[
\nu=N-1.
\]

For a \(p\)-dimensional covariance block, define

\[
c(p,\nu)
=
\sum_{j=1}^{p}
\psi\left(\frac{\nu+1-j}{2}\right)
+
p\log 2
-
p\log\nu,
\]

where \(\psi\) is the digamma function.

Implement with `scipy.special.digamma`.

The plugin MI bias in bits is

\[
b_{\mathrm{MI}}(d_A,d_B,N)
=
\frac{
c(d_A,\nu)
+
c(d_B,\nu)
-
c(d_A+d_B,\nu)
}{
2\log 2
}.
\]

Correct the three MI quantities:

\[
I_{1,c}
=
\widehat I_1
-
b_{\mathrm{MI}}(d_T,d_1,N),
\]

\[
I_{2,c}
=
\widehat I_2
-
b_{\mathrm{MI}}(d_T,d_2,N),
\]

\[
I_{12,c}
=
\widehat I_{12}
-
b_{\mathrm{MI}}(d_T,d_1+d_2,N).
\]

Return each bias term separately.

Do not apply the Venkatesh proportional correction to union information.

---

## Step 3: Gaussian parametric-resampling correction for synergy

The purpose of this procedure is to estimate finite-sample bias near the observed covariance and information level.

### Reference value

Calculate the reference synergy by applying gPID directly to the fitted covariance:

\[
S_{\mathrm{ref}}=S(\widehat\Sigma).
\]

Normally this is the same as the original plugin synergy. Store both and report their absolute difference as an optimizer-reproducibility diagnostic.

### Surrogate loop

For \(w=1,\ldots,W\):

1. Draw \(N\) observations from

   \[
   \mathcal N(0,\widehat\Sigma).
   \]

2. Estimate the surrogate sample covariance with `ddof=1`.
3. Apply the **same** gPID estimator and optimizer settings.
4. Store surrogate synergy and optimizer diagnostics.

Estimate

\[
\widehat b_{S,\mathrm{resampling}}
=
\frac1W\sum_{w=1}^{W}\widehat S^{(w)}
-
S_{\mathrm{ref}}.
\]

Return at least:

```python
{
    "bias": float,
    "reference_synergy": float,
    "surrogate_synergies": np.ndarray,
    "mean_surrogate_synergy": float,
    "std_surrogate_synergy": float,
    "sem_surrogate_synergy": float,
    "optimizer_diagnostics": list[dict],
}
```

Use independent random-number substreams for every iteration.

---

## Step 4: Target-shuffle correction for synergy

The shuffle must remove target-source information while preserving the empirical relationship between the two sources.

For each shuffle \(v=1,\ldots,V\):

1. Draw a random permutation \(\pi_v\) of the \(N\) row indices.
2. Construct

   \[
   Z_v=[T_{\pi_v},X_1,X_2].
   \]

3. Do **not** permute \(X_1\) relative to \(X_2\).
4. Estimate the covariance with `ddof=1`.
5. Apply the same gPID estimator.
6. Store the surrogate synergy and diagnostics.

Because the population synergy under this null is zero, estimate

\[
\widehat b_{S,\mathrm{shuffle}}
=
\frac1V\sum_{v=1}^{V}\widehat S_{\mathrm{shuffle}}^{(v)}.
\]

Return the same style of diagnostic structure as for resampling.

Add a diagnostic confirming that the \(X_1-X_2\) covariance block is unchanged by each shuffle up to floating-point tolerance.

---

## Step 5: Merge the synergy bias estimates

Validate

\[
0\le w\le1.
\]

Default to

\[
w=0.5.
\]

Calculate

\[
\widehat b_{S,\mathrm{merged}}
=
w\widehat b_{S,\mathrm{resampling}}
+
(1-w)\widehat b_{S,\mathrm{shuffle}}.
\]

Then

\[
S_c
=
\widehat S_{\mathrm{plugin}}
-
\widehat b_{S,\mathrm{merged}}.
\]

Do not clip \(S_c\) to zero by default.

Record whether the corrected value is below `-negative_tolerance`.

---

## Step 6: Reconstruct corrected redundancy and unique information

Use the corrected MI quantities and corrected synergy:

\[
R_c
=
I_{1,c}
+
I_{2,c}
-
I_{12,c}
+
S_c,
\]

\[
U_{1,c}
=
I_{1,c}
-
R_c,
\]

\[
U_{2,c}
=
I_{2,c}
-
R_c.
\]

Equivalent formulas that may be used as cross-checks are

\[
U_{1,c}
=
I_{12,c}
-
I_{2,c}
-
S_c,
\]

\[
U_{2,c}
=
I_{12,c}
-
I_{1,c}
-
S_c.
\]

Calculate both versions in tests and verify agreement.

Do not independently subtract null biases from \(R,U_1,U_2\). They must be reconstructed from the corrected synergy and corrected mutual informations.

---

## PID identity checks

Calculate:

\[
e_{\mathrm{joint}}
=
I_{12,c}
-
(R_c+U_{1,c}+U_{2,c}+S_c),
\]

\[
e_1
=
I_{1,c}
-
(R_c+U_{1,c}),
\]

\[
e_2
=
I_{2,c}
-
(R_c+U_{2,c}).
\]

Require these errors to be below a strict numerical tolerance, such as \(10^{-10}\) for `float64`, unless the existing gPID code uses lower precision.

The returned result must include all three errors.

---

## Required output schema

Return a nested dictionary similar to:

```python
{
    "method": "lorenz_gaussian_merged",
    "configuration": {
        "n_samples": N,
        "dim_target": dT,
        "dim_source_1": d1,
        "dim_source_2": d2,
        "n_resamples": W,
        "n_shuffles": V,
        "merge_weight": w,
        "covariance_ddof": 1,
        "information_unit": "bits",
    },
    "plugin": {
        "mi_target_source_1": I1_plugin,
        "mi_target_source_2": I2_plugin,
        "mi_target_joint_sources": I12_plugin,
        "red": red_plugin,
        "unq1": unq1_plugin,
        "unq2": unq2_plugin,
        "syn": syn_plugin,
    },
    "bias": {
        "mi_target_source_1_goodman": b_I1,
        "mi_target_source_2_goodman": b_I2,
        "mi_target_joint_sources_goodman": b_I12,
        "syn_resampling": b_syn_resampling,
        "syn_shuffle": b_syn_shuffle,
        "syn_merged": b_syn_merged,
    },
    "corrected": {
        "mi_target_source_1": I1_corrected,
        "mi_target_source_2": I2_corrected,
        "mi_target_joint_sources": I12_corrected,
        "red": red_corrected,
        "unq1": unq1_corrected,
        "unq2": unq2_corrected,
        "syn": syn_corrected,
    },
    "surrogates": {
        "resampling_synergies": [...],
        "shuffle_synergies": [...],
    },
    "diagnostics": {
        "pid_identity_error_joint": ...,
        "pid_identity_error_source_1": ...,
        "pid_identity_error_source_2": ...,
        "covariance_min_eigenvalue": ...,
        "covariance_max_eigenvalue": ...,
        "covariance_condition_number": ...,
        "covariance_rank": ...,
        "plugin_reference_synergy_difference": ...,
        "negative_component_flags": {
            "red": bool,
            "unq1": bool,
            "unq2": bool,
            "syn": bool,
        },
        "resampling_failures": int,
        "shuffle_failures": int,
        "optimizer": {...},
    },
}
```

Store surrogate arrays optionally through a flag if memory becomes an issue, but retain their means, standard deviations, SEMs, and failure counts.

---

## Randomness and reproducibility

Use `numpy.random.SeedSequence` to produce independent deterministic seeds for:

- the original gPID call, if stochastic;
- each resampling draw;
- each resampling gPID optimization;
- each target permutation;
- each shuffle gPID optimization.

Do not reuse the same seed for all optimizer calls.

Given the same root seed and same inputs, the complete result must be reproducible.

---

## Optimizer requirements

Every plugin, resampling, and shuffle gPID call must use the same:

- number of initializations;
- initialization rule;
- convergence tolerance;
- maximum iterations;
- PSD projection;
- numerical precision;
- objective convention;
- information units.

If the optimizer supports multiple restarts, return the best feasible objective according to the gPID implementation.

Store, when available:

```python
{
    "converged": bool,
    "n_iterations": int,
    "objective": float,
    "constraint_violation": float,
    "minimum_eigenvalue": float,
    "restart_objectives": list[float],
}
```

Do not interpret optimizer instability as sampling bias. Flag surrogate calls whose optimizer fails.

Default behavior should raise an error if any surrogate gPID call fails. A later optional mode may allow failed iterations to be excluded, but it must report the number excluded and must not silently continue.

---

## Computational cost

With the paper defaults, each corrected result requires approximately

\[
1+20+20=41
\]

gPID evaluations, excluding optimizer restarts.

Implement `n_jobs` or another optional parallel-execution argument if practical. Parallel execution must remain deterministic.

Do not parallelize before the serial implementation and tests are correct.

---

## Treatment of PCA and ridge predictions

Apply this correction to the final sample-level variables used for PID:

- target neural PCs \(T\);
- cross-validated prediction from model 1, \(X_1\);
- cross-validated prediction from model 2, \(X_2\).

Treat previously fitted PCA and ridge models as fixed.

Do not refit PCA or ridge inside each Lorenz resampling or shuffle iteration unless a separate analysis explicitly asks for uncertainty across the entire preprocessing pipeline.

Use the retained dimensions after PCA in all bias formulas and diagnostics.

---

## Negative corrected components

Do not silently clip negative corrected components to zero.

Negative corrected atoms can indicate:

- residual finite-sample error;
- numerical instability;
- optimizer failure;
- violation of the Gaussian assumption;
- insufficient samples relative to dimension.

Return the raw corrected values and diagnostic flags.

Optionally provide a separate presentation helper that clips for plotting, but never overwrite the raw scientific result.

---

## Sufficient-sampling diagnostic

For equal dimensions

\[
d_T=d_1=d_2=d,
\]

Lorenz et al. use the rule of thumb

\[
N>12d.
\]

Implement a diagnostic only:

```python
sufficient_sampling_equal_dims = (
    dT == d1 == d2 and N > 12 * dT
)
```

For unequal dimensions, optionally report the heuristic ratio

\[
\frac{N}{4(d_T+d_1+d_2)},
\]

but label this clearly as a heuristic extension rather than a directly validated theorem.

Do not block computation solely because the sufficient-sampling rule is not met.

---

## Tests

Create unit and integration tests.

### 1. Gaussian MI formula

Generate a positive-definite covariance with known block structure. Verify that `gaussian_mi_from_cov` agrees with a direct conditional-covariance calculation.

### 2. Goodman bias formula

For several combinations of \(N,d_A,d_B\):

1. Generate many Gaussian datasets from a fixed covariance.
2. Calculate plugin MI.
3. Verify that the Monte Carlo plugin bias approximately matches `gaussian_mi_bias_bits`.
4. Verify that subtracting the analytical term substantially reduces the mean bias.

Use low dimensions and enough Monte Carlo replications for a stable automated test, or mark the expensive version as a slow test.

### 3. Zero-information null

Generate mutually independent Gaussian \(T,X_1,X_2\). Across many datasets:

- plugin synergy should be upward biased;
- the merged correction should reduce the average synergy bias;
- corrected atoms should be close to zero on average.

Do not require every individual corrected atom to be nonnegative.

### 4. Source-pair preservation under target shuffle

Verify exactly that target shuffling leaves:

```text
Cov(X1, X1)
Cov(X2, X2)
Cov(X1, X2)
```

unchanged up to floating-point tolerance.

### 5. PID identities

For every test case, verify the three corrected PID identities.

### 6. Reproducibility

Two calls with the same root seed must return equal results and surrogate arrays.

### 7. Seed independence

Verify that surrogate datasets and permutations are not duplicated because of seed reuse.

### 8. Invalid covariance

Verify that singular or non-positive-definite covariance inputs fail with an informative error.

### 9. Dimension/sample check

Verify that the exact log-determinant correction rejects dimensions that exceed the available Wishart degrees of freedom.

### 10. No Venkatesh correction

Add a regression test confirming that the Lorenz wrapper never calls the old proportional union-information correction.

### 11. Merge calculation

Verify exactly:

\[
b_{\mathrm{merged}}
=
0.5b_{\mathrm{resampling}}
+
0.5b_{\mathrm{shuffle}}
\]

under default settings.

### 12. Corrected unique-information formulas

Verify agreement between:

```python
unq1 = I1_corr - red_corr
unq1_alt = I12_corr - I2_corr - syn_corr
```

and similarly for source 2.

---

## Recommended simulation validation outside unit tests

Add a standalone script:

```text
scripts/validate_lorenz_gaussian_bias.py
```

It should compare:

1. plugin gPID;
2. original Venkatesh correction;
3. Lorenz resampling correction;
4. Lorenz shuffle correction;
5. Lorenz merged correction.

For each population covariance and sample size:

1. Calculate population ground truth from the true covariance.
2. Draw many independent datasets.
3. Apply all estimators.
4. Report for every PID atom:
   - mean estimate;
   - absolute bias;
   - relative bias;
   - standard deviation;
   - RMSE;
   - fraction of negative corrected estimates;
   - optimizer failure rate.

At minimum, test:

- zero-information system;
- mainly redundant system;
- mainly unique system;
- mainly synergistic system;
- bit-of-all system;
- several sample sizes;
- several dimensions;
- a configuration matching the real analysis, such as \(N=1000\) and \(d_T=d_1=d_2=98\), when computationally feasible.

Save results as CSV or Parquet and generate diagnostic plots.

---

## Logging

Use structured logging rather than print statements.

At the start of a run, log:

```text
N, dT, d1, d2, W, V, merge weight, root seed
```

After completion, log:

```text
plugin synergy
resampling bias
shuffle bias
merged bias
corrected synergy
corrected redundancy
corrected unique information
optimizer failure counts
PID identity errors
```

---

## Documentation

Add docstrings that state:

1. This implements the Lorenz et al. Gaussian merged bias correction.
2. The correction applies to Gaussian BROJA/gPID.
3. Classical MIs use the exact Wishart/Goodman correction.
4. Synergy bias is estimated through parametric resampling and target shuffling.
5. Redundancy and unique information are reconstructed from PID identities.
6. The implementation assumes the ordinary sample covariance with `ddof=1`.
7. Shrinkage covariance is not supported by the exact analytical MI correction in version 1.
8. Corrected components are not clipped.

Add a usage example:

```python
result = lorenz_gaussian_bias_corrected_pid(
    target=T,
    source_1=X1,
    source_2=X2,
    gpid_from_cov=existing_gpid_function,
    n_resamples=20,
    n_shuffles=20,
    merge_weight=0.5,
    random_state=88,
    gpid_kwargs={
        "max_iterations": 10_000,
        "n_restarts": 5,
        "tolerance": 1e-8,
    },
)

print(result["corrected"])
print(result["bias"])
print(result["diagnostics"])
```

---

## Acceptance criteria

The implementation is complete when:

- [ ] It accepts \(T,X_1,X_2\) and an existing gPID callable.
- [ ] It computes plugin MI and gPID quantities.
- [ ] It applies exact Gaussian MI corrections.
- [ ] It estimates resampling synergy bias with \(W=20\) by default.
- [ ] It estimates target-shuffle synergy bias with \(V=20\) by default.
- [ ] It merges the two synergy biases with weight \(0.5\) by default.
- [ ] It reconstructs redundancy and unique information from the corrected identities.
- [ ] It returns plugin, bias, corrected, surrogate, and diagnostic outputs.
- [ ] It is deterministic under a fixed seed.
- [ ] It does not call the Venkatesh proportional union correction.
- [ ] It does not clip negative components.
- [ ] All unit tests pass.
- [ ] A simulation script shows that the merged correction reduces finite-sample bias relative to the plugin and Venkatesh estimators in representative Gaussian systems.

---

## Primary reference

Gabriel Matías Lorenz, Nicola Marie Engel, Loren Koçillari, et al. (2026).  
*Sampling bias corrections for discrete and Gaussian partial information decompositions*.  
Patterns 7, 101619.  
DOI: 10.1016/j.patter.2026.101619

The implementation should follow the Gaussian resampling, shuffle-subtraction, and merged-correction procedures in the paper's Methods section. When this document and the paper disagree, follow the paper and document the discrepancy in the code review.
