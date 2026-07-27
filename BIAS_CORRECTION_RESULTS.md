# Gaussian BROJA objective and unique-information bias correction

## Result

The requested \(10^{-2}\)-bit signed mean-error target was reached for the
theoretical equal-channel covariance, but only by using the known
equal-channel structure.

Across 5,000 independent trials, the final correction produced:

| Quantity | Truth (bits) | Signed mean error (bits) | Monte Carlo SE | MAE per trial | RMSE per trial | Pass |
|---|---:|---:|---:|---:|---:|:---:|
| Corrected `obj` / union | 7.263156237 | +0.001561804 | 0.001527428 | 0.086388208 | 0.108005967 | Yes |
| `unq1` | 0 | +0.000021046 | 0.001317271 | 0.074314353 | 0.093135814 | Yes |
| `unq2` | 0 | -0.000021046 | 0.001317271 | 0.074314353 | 0.093135814 | Yes |
| Redundancy | 7.263156237 | +0.001561804 | 0.001527428 | 0.086388208 | 0.108005967 | Yes |
| Synergy | 4.994537920 | +0.000435927 | 0.001243267 | 0.070518060 | 0.087904512 | Yes |

Here, “error across mean trials” is interpreted as statistical bias:

\[
\left|\frac{1}{R}\sum_{r=1}^{R}
(\widehat{\theta}_r-\theta_{\mathrm{true}})\right| \leq 0.01.
\]

This differs from mean absolute error (MAE). The correction removes mean bias,
but cannot remove ordinary sample-to-sample variance. Therefore, an individual
trial can still have more than \(0.01\) bit error.

## Theoretical benchmark

The benchmark used the project’s direct covariance construction with:

- \(d_{X_1}=50\), \(d_{X_2}=55\), \(d_T=35\)
- \(N=1000\)
- \(p=0.3\), \(q=r=0.5\)
- ordinary unbiased sample covariance, with denominator \(N-1\)
- no clipping

The population values are:

| Quantity | Value (bits) |
|---|---:|
| \(I(T;X_1)\) | 7.263156237 |
| \(I(T;X_2)\) | 7.263156237 |
| \(I(T;X_1,X_2)\) | 12.257694157 |
| Union objective | 7.263156237 |
| `unq1`, `unq2` | 0, 0 |
| Redundancy | 7.263156237 |
| Synergy | 4.994537920 |

Because \(q=r\), both source-to-target channels contain the same 35 correlated
modes and are information-equivalent. The extra source dimensions contain no
target information. Consequently,

\[
I(T;X_1)=I(T;X_2)
=-\frac{35}{2}\log_2(1-0.5^2)
=7.263156237.
\]

The theoretical covariance has no sampling bias. It was used as the sampling
distribution and ground truth; it was not passed directly to a debiasing
function as if it were a sample covariance.

## Why the original unique information was wrong

This example is a tie at the boundary of the optimized union-information
problem. In the population, neither channel is uniquely better, so both unique
informations are zero. A finite sample breaks the tie randomly. The optimizer
then interprets part of that random asymmetry as unique information, producing
a large positive selection bias in `obj`.

This explains the behavior of the attempted corrections:

- Correcting `imx`, `imy`, and `imxy` with their exact Wishart biases is
  necessary, but it does not correct the optimizer’s tie-selection bias.
- Target permutation estimates the optimizer floor at zero signal, not at the
  actual nonzero equal-channel boundary.
- A standard parametric bootstrap is centered on the observed, already
  asymmetric channels. It treats some sample asymmetry as population
  asymmetry, so substantial bias remains.
- The Venkatesh factor assumes the union objective has the same proportional
  bias as joint mutual information. That assumption is not accurate here.

Ordinary bootstrap failure at parameter-space boundaries is a known
nonregular-estimation problem; see
[Andrews (2000)](https://doi.org/10.1111/1468-0262.00114). Work on bootstrap
functions of covariance matrices also stresses that model restrictions must be
represented in the bootstrap design; see
[Beran and Srivastava (1985)](https://doi.org/10.1214/aos/1176346579).

## Correction that passed

First correct all ordinary Gaussian MIs using the exact Wishart log-determinant
bias. The existing `mi_wishart_bias` function returns nats, so its output must
be divided by \(\ln 2\) before subtracting it from GPID values in bits.

For this benchmark, the corrections are:

| MI | Bias to subtract (bits) |
|---|---:|
| \(I(T;X_1)\) | 1.320906570 |
| \(I(T;X_2)\) | 1.456879657 |
| \(I(T;X_1,X_2)\) | 2.858478595 |

Under the explicit population-channel-equivalence assumption, replace the
nonregular optimized union estimator with:

\[
\widehat U_{\mathrm{eq}}
=\frac{\widehat I_{1,\mathrm{Wishart}}
+\widehat I_{2,\mathrm{Wishart}}}{2}.
\]

Then reconstruct the PID without clipping:

\[
\begin{aligned}
\widehat{UI}_1 &= \widehat U_{\mathrm{eq}}-\widehat I_{2,\mathrm{Wishart}},\\
\widehat{UI}_2 &= \widehat U_{\mathrm{eq}}-\widehat I_{1,\mathrm{Wishart}},\\
\widehat R &= \widehat I_{1,\mathrm{Wishart}}
             +\widehat I_{2,\mathrm{Wishart}}-\widehat U_{\mathrm{eq}},\\
\widehat S &= \widehat I_{12,\mathrm{Wishart}}-\widehat U_{\mathrm{eq}}.
\end{aligned}
\]

This is exactly unbiased in expectation under Gaussian sampling and known
channel equivalence because each corrected MI is exactly unbiased. Per trial,
`unq2 = -unq1`; small negative values are expected because clipping is
intentionally disabled.

## Repeated-trial comparison

All rows below use the same canonical model. Different trial counts reflect
computational cost. A method fails when its absolute signed mean error is
clearly above \(0.01\) bit.

| Correction for `obj` | Trials | Inner draws | Signed mean error ± MC SE (bits) | RMSE (bits) | Result |
|---|---:|---:|---:|---:|:---:|
| None, raw optimizer | 200 | 0 | +3.359957 ± 0.010165 | 3.363016 | Fail |
| Venkatesh proportional factor + bounds | 200 | 0 | +1.352385 ± 0.009834 | 1.359482 | Fail |
| Subtract exact joint-MI Wishart bias from `obj` | 200 | 0 | +0.501479 ± 0.010165 | 0.521579 | Fail |
| Target-permutation null | 24 | 12 | +1.35584 ± 0.03111 | 1.36402 | Fail |
| Gaussian parametric bootstrap | 24 | 12 | +1.08225 ± 0.03565 | 1.09567 | Fail |
| Double bootstrap / bootstrap-depth quadratic extrapolation | 24 | 12 paired paths | +0.90877 ± 0.03954 | 0.92834 | Fail |
| Equivalent-channel pooled Wishart | 5,000 | 0 | **+0.001561804 ± 0.001527428** | 0.108005967 | **Pass** |

The corresponding unique-information results were:

| Correction | `unq1` signed error (bits) | `unq2` signed error (bits) | Result |
|---|---:|---:|:---:|
| Raw | +1.906779 | +2.033386 | Fail |
| Venkatesh factor | +1.354560 | +1.345340 | Fail |
| Additive joint-Wishart | +0.505180 | +0.495814 | Fail |
| Permutation null | +1.33278 | +1.35220 | Fail |
| Parametric bootstrap | +1.05920 | +1.07862 | Fail |
| Double bootstrap | +0.88571 | +0.90513 | Fail |
| Equivalent-channel pooled Wishart | **+0.000021046** | **-0.000021046** | **Pass** |

The 95% Monte Carlo intervals for all five final PID quantities are wholly
inside the required \([-0.01,0.01]\)-bit mean-error interval.

## Additional exploratory results

A broader ten-dataset high-dimensional sweep was also used to compare generic
objective corrections. These figures are per-dataset objective-error summaries,
not the final signed-mean acceptance test:

| Method | MAE | RMSE | Maximum absolute error |
|---|---:|---:|---:|
| Raw | 2.397 | 2.488 | 3.518 |
| Permutation | 0.657 | 0.775 | 1.487 |
| Parametric bootstrap | 0.400 | 0.556 | 1.217 |
| Double bootstrap | 0.363 | 0.525 | 1.117 |
| Additive Wishart | 0.686 | 0.812 | 1.713 |
| Linear inverse-sample-size extrapolation | 0.262 | 0.408 | 0.964 |
| Quadratic inverse-sample-size extrapolation | 0.316 | 0.462 | 0.993 |

Other experiments were not retained as production corrections:

- Oracle-approximating shrinkage strongly overcorrected the canonical seed:
  corrected `obj` approximately 4.516 bits, about -2.747 bits from truth.
- A constrained equal-boundary parametric bootstrap improved the canonical
  seed but still left about -0.0566 bit error.
- No clipping result was used to claim a pass.

## Literature comparison

Venkatesh et al. define their Gaussian PID union correction by assigning the
union objective the same relative correction as joint MI:

\[
\widehat U_{\mathrm{Venkatesh}}
=\widehat U_{\mathrm{raw}}
\frac{\widehat I_{12,\mathrm{corrected}}}
     {\widehat I_{12,\mathrm{raw}}}.
\]

The paper explicitly notes that it does not theoretically establish that this
union estimate is unbiased and may retain residual bias. See the
[NeurIPS 2023 paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ec0bff8bf4b11e36f874790046dfdb65-Paper-Conference.pdf)
and its [official GPID implementation](https://github.com/praveenv253/gpid).

The exact Gaussian MI correction is supported by the Wishart log-determinant
theory of [Cai, Liang, and Zhou](https://doi.org/10.1016/j.jmva.2015.02.003).
It applies directly to `imx`, `imy`, and `imxy`, but not automatically to an
optimized `obj`.

Koçillari et al. study limited-sampling PID bias and report that synergy can be
much more biased than redundancy. They propose quadratic extrapolation (QE),
shuffle subtraction, and their combination. The
[open-access precursor](https://pmc.ncbi.nlm.nih.gov/articles/PMC11185652/)
explains that target shuffling is conservative because the zero-information
distribution can have more upward bias than signal-bearing data. The final
2026 article is
[Sampling bias corrections for discrete and Gaussian partial information decompositions](https://doi.org/10.1016/j.patter.2026.101619).
Reference implementations are available in
[MINT](https://github.com/panzerilab/MINT), including
[QE](https://github.com/panzerilab/MINT/blob/main/src/core_functions/BiasCorrection/extrapolation.m),
[shuffle subtraction](https://github.com/panzerilab/MINT/blob/main/src/core_functions/BiasCorrection/shuffle_subtraction.m),
and
[QE plus shuffle](https://github.com/panzerilab/MINT/blob/main/src/core_functions/BiasCorrection/shuffSub_extrapolation.m).

The literature methods are useful generic candidates, but none provides a
universal \(0.01\)-bit guarantee. The passing estimator here comes from using
additional population structure that is known in this theoretical example.

## Implemented functions

Both additions are in
`Partial_Information_Decomposition/bias_functions.py`.

### `parametric_bootstrap_obj_debias`

This is a generic, signal-preserving Gaussian parametric-bootstrap diagnostic.
For every bootstrap covariance, it reruns GPID with:

```python
unbiased=False
debias_factor_bool=False
```

It returns:

\[
\widehat b_{\mathrm{PB}}=\overline{obj^*}-obj,\qquad
obj_{\mathrm{corrected}}=obj-\widehat b_{\mathrm{PB}}.
\]

Thus its returned `bias` is the amount to subtract. It improved the canonical
estimate substantially but did not meet the required threshold.

### `equivalent_channels_obj_debias`

This is the passing, specialized estimator. Example use with a supplied sample
covariance:

```python
from Partial_Information_Decomposition.bias_functions import (
    equivalent_channels_obj_debias,
)

config = {
    "dx1": 50,
    "dx2": 55,
    "dt": 35,
    "n_samples": 1000,
    "device": "cpu",
    "equivalent_channels": True,
    "covariance_is_sample": True,
}

result = equivalent_channels_obj_debias(
    config=config,
    covariance=sample_covariance_t_x1_x2,
)

bias_to_subtract = result["bias"]
corrected_union = result["corrected_obj"]
pid = result["pid"]
```

The supplied covariance order must be `[T, X1, X2]`. Alternatively, omit
`covariance` and supply sample arrays as `config["T"]`, `config["X1"]`, and
`config["X2"]`.

For this function:

```python
corrected_union == raw_obj - bias_to_subtract
```

The function rejects use unless `equivalent_channels=True`. When a covariance
is supplied, it also requires `covariance_is_sample=True` to prevent accidental
Wishart correction of a theoretical, shrinkage, or other non-sample
covariance.

The nested `pid["obj"]` retains the raw optimizer objective to match the
existing wrapper convention. The corrected value is in
`pid["union_info"]` and the top-level `corrected_obj`.

## Important limitation

The passing correction is not generic. Channel equivalence must be justified
from the model or experimental design, not inferred from one noisy sample.

For aligned scalar Gaussian channels with unequal information, the pooled
estimator’s deterministic objective error is:

\[
\frac{|I(T;X_1)-I(T;X_2)|}{2}.
\]

Therefore, the true pairwise-MI gap must be below \(0.02\) bit for this
specific approximation error to remain below \(0.01\) bit. For example,
changing the theoretical model to \(q=0.6\), \(r=0.5\) would produce roughly
2 bits of objective error, so the equivalent-channel correction must not be
used there.

## Verification performed

- Python syntax compilation passed.
- Torch sample-array and sample-covariance routes agreed to machine precision.
- NumPy conversion is confined to the GPID optimizer boundary; MI correction
  remains in Torch and uses explicit nats-to-bits conversion.
- Covariance ordering and dimensions are validated.
- Non-finite, asymmetric, singular, and population-covariance misuse paths are
  guarded.
- A supplied `raw_obj` is checked against the optimizer result.
- PID identities were checked numerically.
- No clipping or PID-bound rectification is applied by the new correction.
- The final 5,000-trial benchmark used seed `20260726`.
- The independent 200-trial GPID comparison used seed `20260727`.
- The 24-trial permutation/bootstrap comparison used outer seeds
  `2026072700 + trial`, with 12 inner draws per trial.

The functions are not wired into `PID_calc.py` in this change; they can be
called directly as shown above. `FUNCTION_REGISTRY.md` was also left unchanged
to respect the authorized file scope.
