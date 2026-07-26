# Route-Specific Ground Truth for the PCA/Ridge Simulations

## 1. Scope and notation

This document proves how the population ground truth is calculated for the
`RAW`, `PCA`, and `RIDGE CV` routes in:

- `all_above_zero_pca_ridge.py`
- `con_all_above_zero_pca_ridge.py`
- `equal_unique_pca_ridge.py`
- the Sonic experiment in `pid_feature_middleware.py`

All population covariances are constructed in the observable order

\[
Z=
\begin{bmatrix}
X_1\\
X_2\\
T
\end{bmatrix},
\qquad
\Sigma_{\mathrm{RAW}}
=\operatorname{Cov}(Z)
=
\begin{bmatrix}
\Sigma_{11}&\Sigma_{12}&\Sigma_{1T}\\
\Sigma_{21}&\Sigma_{22}&\Sigma_{2T}\\
\Sigma_{T1}&\Sigma_{T2}&\Sigma_{TT}
\end{bmatrix}.
\]

`calculate_theoretical_pid` accepts this `[X1, X2, T]` order. Immediately
before calling a PID implementation, it changes the order to the
`[T, X1, X2]` convention required by that implementation.

Let

\[
\mathcal P_m(\Sigma;d_1,d_2,d_T)
\]

denote all PID and mutual-information components returned by PID method \(m\)
from covariance \(\Sigma\) and block dimensions
\([d_1,d_2,d_T]\).

Every generator considered here is jointly Gaussian. Consequently, its mean
and covariance determine its complete observable distribution, and applying
\(\mathcal P_m\) to its population covariance gives the covariance-based
population ground truth.

## 2. Proof of the full RAW covariance construction

### 2.1 Independent-coordinate theorem

Suppose \(C\in\mathbb R^{q\times q}\) is the covariance of one scalar
coordinate from each latent/observable block and there are \(p\) independent,
identically distributed coordinates. The generated arrays group all \(p\)
coordinates of the first scalar block, then all coordinates of the second
block, and so on.

For scalar blocks \(a,b\) and coordinate indices \(j,\ell\),

\[
\operatorname{Cov}(Z_{a,j},Z_{b,\ell})
=C_{ab}\mathbf 1_{\{j=\ell\}}.
\]

The \((a,b)\) covariance block is therefore \(C_{ab}I_p\). By the definition
of the Kronecker product,

\[
\boxed{\Sigma_{\mathrm{RAW}}=C\otimes I_p.}
\]

This ordering matters. \(I_p\otimes C\) is coordinate-major and does not match
the grouped columns returned by the generators.

The code constructs the full matrix with
`torch.kron(coordinate_covariance, torch.eye(p))`. RAW PID is calculated
directly from this full covariance and its full dimensions. There is no
scalar-PID route multiplier.

### 2.2 All-above-zero covariance

Write \(a_1,a_2,r,s,\sigma\) for the two unique weights, redundant weight,
shared-noise weight, and noise standard deviation. One coordinate is

\[
\begin{aligned}
X_1&=a_1U_1+rR+\sigma N_1+s\sigma N_s,\\
X_2&=a_2U_2+rR+\sigma N_2+s\sigma N_s,\\
T&=a_1U_1+a_2U_2+rR+\sigma N_T,
\end{aligned}
\]

where the displayed latent variables are independent standard Gaussians.
Expanding each covariance gives

\[
C_{\mathrm{all}}=
\begin{bmatrix}
a_1^2+r^2+\sigma^2(1+s^2)&r^2+s^2\sigma^2&a_1^2+r^2\\
r^2+s^2\sigma^2&a_2^2+r^2+\sigma^2(1+s^2)&a_2^2+r^2\\
a_1^2+r^2&a_2^2+r^2&a_1^2+a_2^2+r^2+\sigma^2
\end{bmatrix}.
\]

Thus

\[
\boxed{\Sigma_{\mathrm{RAW}}=C_{\mathrm{all}}\otimes I_p},
\qquad
[d_1,d_2,d_T]=[p,p,p].
\]

### 2.3 Equal-unique covariance

For one coordinate,

\[
X_1=U_1+\sigma N_1,\qquad
X_2=U_2+\sigma N_2,\qquad
T=U_1+U_2+\sigma N_T.
\]

Independence gives

\[
C_{\mathrm{equal}}=
\begin{bmatrix}
1+\sigma^2&0&1\\
0&1+\sigma^2&1\\
1&1&2+\sigma^2
\end{bmatrix}.
\]

Therefore

\[
\boxed{\Sigma_{\mathrm{RAW}}=C_{\mathrm{equal}}\otimes I_p},
\qquad
[d_1,d_2,d_T]=[p,p,p].
\]

### 2.4 Concatenated covariance

Each observable random variable contains `[U1, U2, R]` blocks. Let

\[
q=r^2\left(1+s^2\sigma^2\right).
\]

For one coordinate from each of the nine scalar blocks,

\[
C_{11}=\operatorname{diag}(1+\sigma^2,\sigma^2,q+\sigma^2),
\]

\[
C_{22}=\operatorname{diag}(\sigma^2,1+\sigma^2,q+\sigma^2),
\qquad
C_{TT}=(1+\sigma^2)I_3,
\]

\[
C_{12}=\operatorname{diag}(0,0,q),
\]

\[
C_{1T}=\operatorname{diag}(1,0,r),
\qquad
C_{2T}=\operatorname{diag}(0,1,r).
\]

Hence

\[
C_{\mathrm{concat}}=
\begin{bmatrix}
C_{11}&C_{12}&C_{1T}\\
C_{12}^{\mathsf T}&C_{22}&C_{2T}\\
C_{1T}^{\mathsf T}&C_{2T}^{\mathsf T}&C_{TT}
\end{bmatrix}.
\]

The row/column order of this \(9\times9\) matrix is

```text
[X1_U1, X1_U2, X1_R, X2_U1, X2_U2, X2_R, T_U1, T_U2, T_R].
```

The generator uses the same block grouping through `np.hstack`, so

\[
\boxed{\Sigma_{\mathrm{RAW}}=C_{\mathrm{concat}}\otimes I_p},
\qquad
[d_1,d_2,d_T]=[3p,3p,3p].
\]

This construction does not assume that PCA retains one third of its
components from each latent block.

### 2.5 Sonic/Shadow Evil Twin covariance

For Sonic, summing the independent latent variances and shared covariances
gives

\[
C_0=
\begin{bmatrix}
5.5&3&3\\
3&3.5&1\\
3&1&4.5
\end{bmatrix}.
\]

The generator scales each coordinate so that its latent variance is the
specified total variance divided by \(p\). Consequently,

\[
\boxed{
\Sigma_{\mathrm{RAW}}
=\left(\frac{C_0}{p}\right)\otimes I_p
},
\qquad
[d_1,d_2,d_T]=[p,p,p].
\]

The Shadow construction gives exactly the same observable covariance.
Because both constructions are Gaussian, they have the same observable
distribution and therefore the same covariance-based PID ground truth.

## 3. Affine covariance propagation theorem

### Theorem

For a random vector \(Z\) with covariance \(\Sigma\), a deterministic matrix
\(A\), and a deterministic offset \(b\), let

\[
Y=AZ+b.
\]

Then

\[
\boxed{\operatorname{Cov}(Y)=A\Sigma A^{\mathsf T}.}
\]

### Proof

Because

\[
\mathbb E[Y]=A\mathbb E[Z]+b,
\]

\[
Y-\mathbb E[Y]=A(Z-\mathbb E[Z]).
\]

It follows that

\[
\begin{aligned}
\operatorname{Cov}(Y)
&=\mathbb E[(Y-\mathbb E[Y])(Y-\mathbb E[Y])^{\mathsf T}]\\
&=A\,\mathbb E[(Z-\mathbb E[Z])(Z-\mathbb E[Z])^{\mathsf T}]A^{\mathsf T}\\
&=A\Sigma A^{\mathsf T}.
\end{aligned}
\]

Thus PCA means, scaler means, and Ridge intercepts do not affect covariance.
The fitted linear matrices do affect it.

The propagated matrix also remains positive semidefinite because, for every
vector \(v\),

\[
v^{\mathsf T}A\Sigma A^{\mathsf T}v
=(A^{\mathsf T}v)^{\mathsf T}\Sigma(A^{\mathsf T}v)\ge0.
\]

## 4. PCA-route ground truth

For trial \(s\), sklearn fits three PCA component matrices using only the
training rows:

\[
W_{1,s}\in\mathbb R^{k\times d_1},\qquad
W_{2,s}\in\mathbb R^{k\times d_2},\qquad
W_{T,s}\in\mathbb R^{k\times d_T}.
\]

Its transforms are affine:

\[
X'_{1,s}=W_{1,s}(X_1-\mu_{1,s}),
\]

\[
X'_{2,s}=W_{2,s}(X_2-\mu_{2,s}),
\qquad
T'_s=W_{T,s}(T-\mu_{T,s}).
\]

Define

\[
A_{\mathrm{PCA},s}
=\operatorname{blockdiag}(W_{1,s},W_{2,s},W_{T,s}).
\]

By the affine covariance theorem,

\[
\boxed{
\Sigma_{\mathrm{PCA},s}
=A_{\mathrm{PCA},s}
\Sigma_{\mathrm{RAW}}
A_{\mathrm{PCA},s}^{\mathsf T}
}.
\]

The exact trial-specific PCA ground truth is

\[
\boxed{
\theta_{\mathrm{PCA},s}^{*}
=\mathcal P_m(\Sigma_{\mathrm{PCA},s};k,k,k).
}
\]

For independent scalar replicas, a cross-covariance block has the form
\(\Sigma_{ij}=c_{ij}I_p\). After PCA it becomes

\[
W_{i,s}\Sigma_{ij}W_{j,s}^{\mathsf T}
=c_{ij}W_{i,s}W_{j,s}^{\mathsf T}.
\]

Although \(W_{i,s}W_{i,s}^{\mathsf T}=I_k\), separately fitted PCA bases do
not generally satisfy \(W_{i,s}W_{j,s}^{\mathsf T}=I_k\). This proves why
`k * one_coordinate_PID` is not generally the PCA-route ground truth.

For the concatenated example, the marginal variances also differ between
the U1, U2, and R blocks. PCA can preferentially retain higher-variance block
directions, providing another reason that the old `k / 3` multiplier was not
the covariance of the fitted PCA route.

## 5. Ridge-CV-route ground truth

The Ridge route predicts the target PCA representation separately from each
RAW source. For source \(i\) in trial \(s\), the selected sklearn pipeline
contains:

- a `StandardScaler` mean \(m_{i,s}\) and scale vector \(q_{i,s}\);
- a fitted Ridge coefficient matrix \(C_{i,s}\);
- a fitted Ridge intercept \(c_{i,s}\).

Let

\[
D_{i,s}=\operatorname{diag}(q_{i,s}).
\]

The prediction is

\[
\widehat T_{i,s}
=C_{i,s}D_{i,s}^{-1}(X_i-m_{i,s})+c_{i,s}.
\]

Therefore its effective RAW-input linear map is

\[
\boxed{B_{i,s}=C_{i,s}D_{i,s}^{-1}.}
\]

In the NumPy/sklearn orientation used by the code,

```python
effective_map = ridge.coef_ / scaler.scale_[None, :]
```

after converting `ridge.coef_` to shape `(k, source_dimension)`.

Define

\[
A_{\mathrm{Ridge},s}
=\operatorname{blockdiag}(B_{1,s},B_{2,s},W_{T,s}).
\]

The route covariance is

\[
\boxed{
\Sigma_{\mathrm{Ridge},s}
=A_{\mathrm{Ridge},s}
\Sigma_{\mathrm{RAW}}
A_{\mathrm{Ridge},s}^{\mathsf T}
}.
\]

Equivalently,

\[
\Sigma_{\mathrm{Ridge},s}
=
\begin{bmatrix}
B_1\Sigma_{11}B_1^{\mathsf T}&
B_1\Sigma_{12}B_2^{\mathsf T}&
B_1\Sigma_{1T}W_T^{\mathsf T}\\
B_2\Sigma_{21}B_1^{\mathsf T}&
B_2\Sigma_{22}B_2^{\mathsf T}&
B_2\Sigma_{2T}W_T^{\mathsf T}\\
W_T\Sigma_{T1}B_1^{\mathsf T}&
W_T\Sigma_{T2}B_2^{\mathsf T}&
W_T\Sigma_{TT}W_T^{\mathsf T}
\end{bmatrix}.
\]

The exact trial-specific Ridge-CV ground truth is

\[
\boxed{
\theta_{\mathrm{Ridge},s}^{*}
=\mathcal P_m(\Sigma_{\mathrm{Ridge},s};k,k,k).
}
\]

The existing shared Ridge helper returns predictions and the selected alpha,
but not its fitted pipeline. The simulation preserves those returned
predictions. It then refits the same deterministic
`StandardScaler -> Ridge` pipeline, using the selected alpha and exactly the
same complete training rows, only to recover \(B_{i,s}\).

This refit is the same operation performed by `GridSearchCV(refit=True)` after
alpha selection. The code additionally asserts that the recovered pipeline
reproduces the original held-out predictions to numerical precision before
using its matrix for ground truth.

## 6. Why PCA and Ridge ground truth is calculated per trial

Let \(\mathcal D_s^{\mathrm{train}}\) be trial \(s\)'s training data. The PCA
matrices, scaler parameters, selected Ridge alpha, and Ridge coefficients are
functions of this training set.

The held-out rows are independent of the training rows. Conditional on
\(\mathcal D_s^{\mathrm{train}}\), all fitted maps are fixed and

\[
Z_{\mathrm{test}}\mid\mathcal D_s^{\mathrm{train}}
\sim\mathcal N(\mu,\Sigma_{\mathrm{RAW}}).
\]

For route \(r\),

\[
Y_{r,s}=A_{r,s}Z_{\mathrm{test}}+b_{r,s},
\]

so

\[
\boxed{
\operatorname{Cov}
(Y_{r,s}\mid\mathcal D_s^{\mathrm{train}})
=A_{r,s}\Sigma_{\mathrm{RAW}}A_{r,s}^{\mathsf T}.
}
\]

This proves that the propagated covariance is the exact conditional
population covariance for the route actually fitted in that trial. It uses
the known generator covariance, not the empirical held-out covariance.

RAW uses the identity map, so its ground truth is constant across trials.
PCA and Ridge-CV receive a separately calculated theoretical PID for every
trial.

The transformed covariances must not be averaged before calculating PID.
After integrating over random fitted maps, the transformed law is generally
a Gaussian mixture, and PID is nonlinear. The code calculates PID first for
each fitted route covariance and only then averages the resulting values.

## 7. Proof of the reported bias, variance, and MSE

For route \(r\), trial \(s\), and PID/MI component \(j\), let

\[
\widehat\theta_{r,s,j}
\]

be the held-out estimate and let

\[
\theta^*_{r,s,j}
\]

be the route-specific population ground truth. Define the paired error

\[
e_{r,s,j}=\widehat\theta_{r,s,j}-\theta^*_{r,s,j}.
\]

The displayed theoretical and sampled means are

\[
\overline{\theta^*}_{r,j}
=\frac1S\sum_{s=1}^S\theta^*_{r,s,j},
\qquad
\overline{\widehat\theta}_{r,j}
=\frac1S\sum_{s=1}^S\widehat\theta_{r,s,j}.
\]

The Monte Carlo bias is

\[
\boxed{
\operatorname{Bias}_{r,j}
=\frac1S\sum_{s=1}^Se_{r,s,j}
=\overline{\widehat\theta}_{r,j}
-\overline{\theta^*}_{r,j}.
}
\]

The code reports the paired-error variance

\[
\boxed{
\operatorname{Variance}_{r,j}
=\frac1S\sum_{s=1}^S
\left(e_{r,s,j}-\operatorname{Bias}_{r,j}\right)^2
}
\]

and the paired MSE

\[
\boxed{
\operatorname{MSE}_{r,j}
=\frac1S\sum_{s=1}^Se_{r,s,j}^2.
}
\]

Expanding the square around the mean error proves

\[
\boxed{
\operatorname{MSE}_{r,j}
=\operatorname{Bias}_{r,j}^2
+\operatorname{Variance}_{r,j}.
}
\]

Using only the variance of sampled estimates would not satisfy this identity
when the route-specific truth changes between trials.

## 8. Numerical contract

The implementation follows these rules:

1. Construct the full RAW population covariance in `[X1, X2, T]` order.
2. Fit every PCA and Ridge-CV transformation using training rows only.
3. Preserve the existing held-out transformed arrays and Ridge predictions.
4. Propagate the known population covariance through the fitted linear maps.
5. Calculate theoretical PID with the same PID method as the sampled route
   and with finite-sample bias correction disabled.
6. Pair every held-out estimate with its own trial-specific ground truth.
7. Add no empirical-covariance substitution and no unexplained covariance
   jitter.

If a learned map is rank deficient, its propagated covariance may be
singular. A PID solver that requires positive definiteness should report that
failure rather than silently changing the stated population covariance.

The equal-unique simulation uses the existing Thin-PID numerical optimizer.
That optimizer can emit `invalid value` and maximum-iteration warnings,
including on the original scalar example, and the full \(p=70\) RAW problem
can be slow. These are numerical convergence limitations of the configured
solver, not changes to the population covariance proof. The code passes the
full covariance without replacing it by a scaled scalar result; a warning
therefore remains visible instead of being silently hidden.
