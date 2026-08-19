# Function Registry

Purpose: search this document before creating or changing a function. Reuse an existing callable only after verifying its data order, shapes, estimator convention, and side effects.

Scope: project Python files only. Excludes `external/`, `uni_tests/`, `.git/`, tool metadata, generated caches, and ignored trash code.

Generated from the current AST. Paths, signatures, and line numbers reflect the working tree. Existing useful descriptions were preserved when the same callable still exists at the same path.

## Convention checks before reuse

- Verify random-variable order: `[X1, X2, T]`, `[T, X1, X2]`, or another documented order.
- Verify samples versus covariance input, raw versus whitened covariance, NumPy versus torch, device/dtype, biased versus unbiased estimation, log base, and PID definition.
- Entries marked task-specific or private should not be promoted to general helpers without checking their callers.

## Folder Overview

- `Eigen_PID_Simulations/`: 2 Python files.
- `Lorenz_bias_corr/`: 2 Python files.
- `MultivariateStatistics/`: 7 Python files.
- `Partial_Information_Decomposition/`: 8 Python files.
- `Partial_Information_Decomposition/Idep/`: 7 Python files.
- `Partial_Information_Decomposition/Idep/non_parametric_bias_corr/`: 3 Python files.
- `Simulations/Encoder_simulation/`: 3 Python files.
- `Simulations/PCA_Ridge/`: 3 Python files.
- `Simulations/PCA_rank/`: 5 Python files.
- `Simulations/Theoretical_Examples/Covariance/`: 5 Python files.
- `Simulations/Theoretical_Examples/Covariance/results_covariance/Sweeps_comp2/`: 1 Python file.
- `Simulations/Theoretical_Examples/RVs_Story/`: 4 Python files.
- `Simulations/Theoretical_Examples/RVs_Story/Non-gaussian/`: 1 Python file.
- `Simulations/Theoretical_Examples/RVs_Story/regular_examples/`: 4 Python files.
- `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/`: 4 Python files.
- `Simulations/evil_twin/`: 5 Python files.
- `data/`: 3 Python files.
- `encoding_model/`: 10 Python files.
- `library_wrappers/`: 11 Python files.
- `pipeline/`: 3 Python files.
- `pipeline/analysis/`: 1 Python file.
- `pipeline/analysis/pca_analysis/all_models_pairwise/`: 3 Python files.
- `pipeline/analysis/pca_analysis/function_as_pc/`: 2 Python files.
- `pipeline/analysis/pca_analysis/permuation_analysis/`: 1 Python file.
- `pipeline/analysis/pca_analysis/unique_search_outputs/`: 3 Python files.
- `pipeline/analysis/ridge_analysis/`: 2 Python files.
- `pipeline/full_OTC/`: 2 Python files.
- `pipeline/pipeline_phases/`: 6 Python files.
- `pipeline/plotting/`: 1 Python file.
- `pipeline/ridge_find_alpha/`: 1 Python file.
- `pipeline/subj_PCs/`: 3 Python files.
- `pipeline/toy_examples/`: 2 Python files.
- `pipeline/voxel_experiments/`: 2 Python files.
- `repository root/`: 2 Python files.
- `source_conwell_code/`: 1 Python file.
- `source_conwell_code/pressures/`: 3 Python files.
- `source_conwell_code/pressures/brain_data/`: 3 Python files.
- `supression_effect/`: 4 Python files.
- `toy_examples/`: 7 Python files.

## Eigen_PID_Simulations

### `Eigen_PID_Simulations/gammaSTAR_as_inpunt.py`

File description: Compare native and Eigen-PID Gamma-star optimizer initializations.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `load_optimizer_modules (line 71)` | No inputs | dict[str, tuple[str, Any]] | Load the GPID and Thin-PID modules used by the replay experiment. | Scope: reusable/public. Related: load_exact_gauss_thin_pid. |
| `generate_gaussian_system (line 86)` | random_generator: np.random.Generator, dimension: int | tuple[np.ndarray, np.ndarray, np.ndarray] | Generate one balanced Gaussian target/source covariance system. | Scope: reusable/public. |
| `construct_native_couplings (line 128)` | channel_x: np.ndarray, channel_y: np.ndarray, thin_pid_module: Any | dict[str, np.ndarray] | Construct the native feasible optimizer coupling for each PID method. | Scope: reusable/public. |
| `run_experiment (line 156)` | No inputs | list[dict[str, Any]] | Run all cases while printing repeat and dimension-sweep durations. | Scope: reusable/public. Related: load_optimizer_modules, add_paired_comparisons, generate_gaussian_system, construct_native_couplings. |
| `main (line 265)` | No inputs | Path | Run the experiment and save its CSV, plots, and hyperparameters. | Scope: entry point. Related: run_experiment, save_results_csv, plot_iteration_comparison, plot_pid_comparison. |

### `Eigen_PID_Simulations/gamma_star_reporting.py`

File description: CSV and figure helpers for the Gamma-star initialization experiment.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `create_result_row (line 21)` | experiment_metadata: Mapping[str, Any], run_metadata: Mapping[str, Any], eigen_result: Any, eigen_coupling: Any, convergence: Mapping[str, Any] | ResultRow | Create one CSV-ready optimizer-versus-Eigen result row. | Scope: reusable/public. |
| `add_paired_comparisons (line 116)` | result_rows: Sequence[Mapping[str, Any]] | list[ResultRow] | Add native-minus-Gamma update differences to paired result rows. | Scope: reusable/public. |
| `save_results_csv (line 156)` | result_rows: Sequence[Mapping[str, Any]], path: str \| Path | Path | Write all detailed experiment rows to one CSV file. | Scope: reusable/public. |
| `save_hyperparameters_yaml (line 179)` | hyperparameters: Mapping[str, Any], path: str \| Path | Path | Write experiment hyperparameters to a reproducibility YAML file. | Scope: reusable/public. |
| `print_experiment_summary (line 199)` | result_rows: Sequence[Mapping[str, Any]], csv_path: str \| Path | None | Print convergence counts and method-versus-Eigen error summaries. | Scope: reusable/public. |
| `_result_matrix (line 270)` | result_rows: Sequence[Mapping[str, Any]], dimensions: Sequence[int], repeats: int, method: str, initialization: str, field: str | np.ndarray | Arrange one scalar result field into a dimension-by-repeat matrix. | Scope: private helper. |
| `_masked_summary (line 303)` | values: np.ndarray, valid_mask: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray] | Compute median, minimum, and maximum over valid repeats. | Scope: private helper. Conventions: float64. |
| `plot_iteration_comparison (line 325)` | result_rows: Sequence[Mapping[str, Any]], dimensions: Sequence[int], repeats: int, methods: Sequence[str], initializations: Sequence[str] | Figure | Plot raw optimizer update counts for both initializations. | Scope: reusable/public. Related: _result_matrix, _masked_summary. |
| `plot_pid_comparison (line 405)` | result_rows: Sequence[Mapping[str, Any]], dimensions: Sequence[int], repeats: int, methods: Sequence[str], initializations: Sequence[str] | Figure | Plot raw PID atoms and native-minus-Gamma-star atom differences. | Scope: reusable/public. Related: _result_matrix, _masked_summary. |
| `save_figure (line 591)` | figure: Figure, path: str \| Path, dpi: int=300 | Path | Save one Matplotlib figure to disk. | Scope: reusable/public. |
| `show_figures (line 615)` | No inputs | None | Display all currently open Matplotlib figures and return ''None''. | Scope: reusable/public. |


## Lorenz_bias_corr

### `Lorenz_bias_corr/figure5B_recreatation.py`

File description: Recreate Lorenz Figure 5B and add a Lorenz-corrected Eigen-PID curve.

No functions, methods, or classes defined in this file.

### `Lorenz_bias_corr/plot_figure5b_runtime.py`

File description: Plot Figure 5B method runtimes as a function of simulation dimension.

No functions, methods, or classes defined in this file.


## MultivariateStatistics

### `MultivariateStatistics/Q1_part1.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.

### `MultivariateStatistics/Q1_part2.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.

### `MultivariateStatistics/Q2_part1.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `classify (line 15)` | x, sig1, sig2 | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |

### `MultivariateStatistics/Q2_part2.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `classify (line 15)` | x, sig1, sig2 | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |

### `MultivariateStatistics/Q3.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `log_likelihood (line 20)` | lamb, psi, n | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `objective_lamb (line 36)` | lamb, psi, n | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: log_likelihood. |
| `objective_psi (line 39)` | psi, lamb, n | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: log_likelihood. |

### `MultivariateStatistics/Q4_b.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `negative_log_likelihood (line 7)` | theta, uc_sample, c_samples | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `optimize_theta (line 19)` | uc_sample, c_samples | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: negative_log_likelihood. |
| `optimize_theta.objective (line 22)` | theta | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. Related: negative_log_likelihood. |
| `main (line 35)` | No inputs | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: optimize_theta. |

### `MultivariateStatistics/Q4_f.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `expected_censord_func (line 7)` | theta_old, c_samples | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `m_step (line 11)` | expected_censored, uc_sample, c_samples | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `em_algorithm (line 16)` | theta_initial, uc_sample, c_samples, tolerance=1e-08, max_iterations=1000 | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: expected_censord_func, m_step. |


## Partial_Information_Decomposition/Idep

### `Partial_Information_Decomposition/Idep/Idep_multivariate_gauss.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `Idep_multivariate_gauss (line 29)` | class | Idep_multivariate_gauss instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. Related: dependency_matrix, compute_Idep, pid_values, create_cov_matrix. Conventions: bias-correction aware. |
| `Idep_multivariate_gauss.__init__ (line 30)` | self, config, rng=None, sources: Optional[list]=None, targets: Optional[list]=None, cov_matrix: Optional[torch.tensor]=None, base_e: bool=True, bias_correction: bool=False | No explicit return; likely None / side effects | Initialize the Idep multivariate gaussian class | Scope: method/nested helper. Related: create_cov_matrix, whiten_block, bias_func. Conventions: bias-correction aware. |
| `Idep_multivariate_gauss.create_model_M (line 126)` | self, block1: Optional[torch.tensor]=None, block2: Optional[torch.tensor]=None, block3: Optional[torch.tensor]=None | torch.tensor | This function will create the dependency matrix for the given blocks | Scope: method/nested helper. |
| `Idep_multivariate_gauss.dependency_matrix (line 150)` | self, constraints: list, cov_matrix: Optional[torch.tensor]=None, cov_dict: Optional[dict]=None | dict | This function will create the dependency matrix for the given constraint | Scope: method/nested helper. Related: create_model_M. |
| `Idep_multivariate_gauss.compute_Idep (line 214)` | self | dict | This function calcualtes the mutual information for a given covariance matrix - U models in the lattice | Scope: method/nested helper. |
| `Idep_multivariate_gauss.pid_values (line 261)` | self, unique_1, unique_2 | Unannotated return value; inspect implementation before reuse | This function will compute the PID values using the I_dep values input: unique_0, unique_1 are the unique informations for source 0 and source 1 output: a dictionary with the PID values keys: 'red', 'unq1', 'unq2', 'syn' | Scope: method/nested helper. Related: logdet_wishart_bias. Conventions: bias-correction aware. |
| `Idep_multivariate_gauss.idep (line 309)` | self, cov_matrix: Optional[torch.tensor]=None | dict | This function will compute the full Idep PID decomposition | Scope: method/nested helper. Related: dependency_matrix, compute_Idep, pid_values. |

### `Partial_Information_Decomposition/Idep/Idep_simulations.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `pid_simulation (line 25)` | config, rng, cov, pid_ver, true_values=None | Unannotated return value; inspect implementation before reuse | Run PID simulation with know ground truth PID values from the covariance matrix sample for the true covariance matrix, calculate the PID using the specified method, and return the results along with the ground truth PID values calculated from the covariance matrix. | Scope: reusable/public. Related: pid_calc, sample_data_from_cov. |
| `trials_simulation (line 107)` | config, title | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: make_random_true_cov, calculate_mi_raw, already_exists_in_csv, pid_simulation. |
| `main (line 164)` | config, single=True, multi=False, exp_name=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: make_random_true_cov, pid_simulation, trials_simulation. |

### `Partial_Information_Decomposition/Idep/Idep_univariabe_gauss.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `Idep_univariate_gauss (line 18)` | class | Idep_univariate_gauss instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. Related: dependency_matrix, compute_Idep, pid_values, create_cov_matrix. |
| `Idep_univariate_gauss.__init__ (line 19)` | self, sources: Optional[list]=None, targets: Optional[list]=None, cov_matrix: Optional[torch.tensor]=None | No explicit return; likely None / side effects | Initialize the Idep univariate gaussian class | Scope: method/nested helper. Related: create_cov_matrix. |
| `Idep_univariate_gauss.dependency_matrix (line 46)` | self, constraints: list, cov_matrix: Optional[torch.tensor]=None, cov_dict: Optional[dict]=None | dict | This function will create the dependency matrix for the given constraint | Scope: method/nested helper. |
| `Idep_univariate_gauss.compute_Idep (line 116)` | self, unique: list=[0, 1] | dict | This function calcualtes the mutual information for a given covariance matrix - U models in the lattice | Scope: method/nested helper. |
| `Idep_univariate_gauss.pid_values (line 170)` | self, unique_0, unique_1 | Unannotated return value; inspect implementation before reuse | This function will compute the PID values using the I_dep values input: unique_0, unique_1 are the unique informations for source 0 and source 1 output: a dictionary with the PID values keys: 'red', 'unq0', 'unq1', 'syn' | Scope: method/nested helper. |
| `Idep_univariate_gauss.idep (line 197)` | self, cov_matrix: Optional[torch.tensor]=None | dict | This function will compute the full Idep PID decomposition | Scope: method/nested helper. Related: dependency_matrix, compute_Idep, pid_values. |
| `test_idep_gauss_q0_example (line 221)` | p=0.3, r=0.5, tol=1e-08 | No explicit return; likely None / side effects | Test Example 1 from the paper: q = corr(X0, Y) = 0 p = corr(X0, X1) != 0 r = corr(X1, Y) != 0 | Scope: reusable/public. Related: idep. |
| `check_idep_gauss_r0_example (line 271)` | p=0.3, q=0.5, tol=1e-08 | No explicit return; likely None / side effects | Example 2 from the paper: r = corr(X1, Y) = 0 p = corr(X0, X1) != 0 q = corr(X0, Y) != 0 | Scope: reusable/public. Related: idep. |
| `check_idep_gauss_p0_example (line 323)` | q=0.3, r=0.5, tol=1e-08 | No explicit return; likely None / side effects | Example 3 from the paper: p = corr(X0, X1) = 0 q = corr(X0, Y) != 0 r = corr(X1, Y) != 0 | Scope: reusable/public. Related: idep. |
| `tests (line 375)` | No inputs | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: dependency_matrix, compute_Idep. |

### `Partial_Information_Decomposition/Idep/covariance_shrinkage.py`

File description: Optional covariance-shrinkage operations used by resampling corrections.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `ledoit_wolf_cov (line 10)` | samples: np.ndarray | np.ndarray | Estimate a covariance matrix with Ledoit–Wolf shrinkage. | Scope: reusable/public. Related: fit. |
| `oracle_shrinkage_cov (line 23)` | samples: np.ndarray, assume_centered: bool=False, return_shrinkage: bool=False | np.ndarray \| tuple[np.ndarray, float] | Estimate covariance with Oracle Approximating Shrinkage. | Scope: reusable/public. Related: fit. |
| `shrunk_cov (line 46)` | samples: np.ndarray, alpha: float=0.1 | np.ndarray | Estimate covariance with a fixed shrinkage coefficient. | Scope: reusable/public. Related: fit. |
| `on_covariance (line 60)` | config: dict, covariance: torch.Tensor | dict[str, torch.Tensor] | Apply the configured shrinkage method to one or more covariance matrices. | Scope: reusable/public. Related: ledoit_wolf_cov, oracle_shrinkage_cov, shrunk_cov. |

### `Partial_Information_Decomposition/Idep/covariance_utils.py`

File description: Reusable covariance construction and shrinkage helpers for Idep code.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `build_m8_terms (line 14)` | config: dict, covariance_blocks: dict, whiten: str='whiten_ver', para: bool=False | dict[str, torch.Tensor] | Construct M8 covariance terms from covariance blocks. | Scope: reusable/public. Conventions: whitening-sensitive. |
| `build_m7_terms (line 98)` | config: dict, covariance_blocks: dict, whiten: str='whiten_ver', para: bool=False | dict[str, torch.Tensor] | Construct M7 covariance terms from covariance blocks. | Scope: reusable/public. Related: para_whiten_block. Conventions: whitening-sensitive. |
| `create_cov_m8 (line 170)` | config: dict, p_block: torch.Tensor, q_block: torch.Tensor, r_block: torch.Tensor | torch.Tensor | Create an M8 covariance from P, Q, and R cross-blocks. | Scope: reusable/public. |
| `create_m7_cov (line 204)` | config: dict, cov_m8: torch.Tensor, whitening_normalize: bool=True | torch.Tensor | Construct the corresponding M7 covariance from an M8 covariance. | Scope: reusable/public. Related: create_cov_matrix, whiten_block. Conventions: order [X1, X2, T], whitening-sensitive. |
| `make_random_true_cov (line 241)` | config: dict, rng: torch.Generator \| None=None | tuple[torch.Tensor, torch.Tensor] | Generate a positive-definite M8 covariance and corresponding M7 model. | Scope: reusable/public. Related: create_cov_m8, create_m7_cov, calcualte_mi. |


## Partial_Information_Decomposition/Idep/non_parametric_bias_corr

### `Partial_Information_Decomposition/Idep/non_parametric_bias_corr/bootstrap.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `_get_bootstrap_count (line 19)` | config: dict | int | Infer the number of bootstrap replicates from config. | Scope: private helper. |
| `_to_tensor_list (line 35)` | rvs_list: list, device: str \| torch.device | list[torch.Tensor] | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. Conventions: float64. |
| `_estimate_fitted_model_cov (line 45)` | config: dict | torch.Tensor | Return the fitted covariance used by the parametric bootstrap. | Scope: private helper. Related: _to_tensor_list, para_create_cov_matrix, bootstrap_whiten. Conventions: whitening-sensitive. |
| `bootstrap_func (line 76)` | config: dict, cov_bootstrap: torch.Tensor, calculate_statistic_func: callable | Unannotated return value; inspect implementation before reuse | Estimate parametric-bootstrap bias for a statistic. | Scope: reusable/public. Related: _get_bootstrap_count. |
| `bootstrap_resample (line 129)` | config: dict | list | Generate parametric-bootstrap covariance estimates. | Scope: reusable/public. Related: _get_bootstrap_count, _estimate_fitted_model_cov, para_create_cov_matrix, bootstrap_whiten. Conventions: whitening-sensitive. |
| `bootstrap_whiten (line 168)` | config: dict, cov_dict: dict | torch.Tensor | Project batched covariance estimates onto the M7/M8 whitened model space. | Scope: reusable/public. Related: build_m8_terms, build_m7_terms. Conventions: whitening-sensitive. |

### `Partial_Information_Decomposition/Idep/non_parametric_bias_corr/jackknife.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `jackkinfe_func (line 9)` | config, cov_loo, calculate_statistic_func | Unannotated return value; inspect implementation before reuse | Calculate the jackknife bias correction for a given statistic calculated on leave-one-out covariance matrices. | Scope: reusable/public. Conventions: bias-correction aware. |
| `jackknife_resample (line 29)` | config: dict | list | Compute the full covnarice matrix across smaples and the covariance matrix of the left out ovbesrvation. Using the formula for covariance matrix Σ(-j)=N-2/(S(-j)-(1/N)*s(-j)s(-j)T) Where S(-j)=S-ZjZjT and s(-j)=s-Zj | Scope: reusable/public. Related: para_create_cov_matrix, jackknife_whiten. |
| `jackknife_whiten (line 92)` | config, m7_cov_dict | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: para_whiten_block. Conventions: whitening-sensitive. |

### `Partial_Information_Decomposition/Idep/non_parametric_bias_corr/resampling_wrapper.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `resampleing (line 14)` | resample_inputs: dict, rng | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `calculate_statistic (line 28)` | config: dict, calc_func: callable, population: dict, rng: np.random.Generator | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `calculate_bias (line 35)` | config: dict, statistic_dict: dict, bias_func: callable | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: bias_func. |
| `bias_resampling (line 44)` | config: dict, calc_func: callable=None | dict | This function will calculate the statistics value and it's and will return a dictionary with the following keys: | Scope: reusable/public. Related: bias_func. |


## Partial_Information_Decomposition/Idep

### `Partial_Information_Decomposition/Idep/parallel_Idep_multivariate_gauss.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `para_Idep_multivariate_gauss (line 23)` | class | para_Idep_multivariate_gauss instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. Related: compute_Idep, pid_values, para_create_cov_matrix, whiten_block. Conventions: bias-correction aware. |
| `para_Idep_multivariate_gauss.__init__ (line 24)` | self, N=None, df=None, device='cuda', sources: Optional[list]=None, targets: Optional[list]=None, cov_matrix: Optional[torch.tensor]=None, dims: Optional[list]=None, bias_correction: bool=False | No explicit return; likely None / side effects | Initialize the Idep multivariate gaussian class | Scope: method/nested helper. Related: para_create_cov_matrix, whiten_block. Conventions: bias-correction aware. |
| `para_Idep_multivariate_gauss.compute_Idep (line 110)` | self | dict | This function calcualtes the mutual information for a given covariance matrix - U models in the lattice | Scope: method/nested helper. |
| `para_Idep_multivariate_gauss.pid_values (line 167)` | self, unique_1, unique_2 | Unannotated return value; inspect implementation before reuse | This function will compute the PID values using the I_dep values input: unique_0, unique_1 are the unique informations for source 0 and source 1 output: a dictionary with the PID values keys: 'red', 'unq1', 'unq2', 'syn' | Scope: method/nested helper. Conventions: whitening-sensitive. |
| `para_Idep_multivariate_gauss.idep (line 195)` | self, cov_matrix: Optional[torch.tensor]=None | dict | This function will compute the full Idep PID decomposition | Scope: method/nested helper. Related: compute_Idep, pid_values. |

### `Partial_Information_Decomposition/Idep/simulation_utils.py`

File description: Small reusable data and result helpers retained from legacy Idep simulations.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `to_python_scalar (line 15)` | value: Any | Any | Convert torch/NumPy scalar-like values to serializable Python values. | Scope: reusable/public. |
| `flatten_pid_results (line 35)` | pid_results: dict | dict[str, Any] | Flatten one level of nested PID result dictionaries. | Scope: reusable/public. Related: to_python_scalar. |
| `get_pid_ver_csv_path (line 55)` | output_folder: str \| Path, pid_ver: str, csv_title: str='pid_results' | Path | Build the result CSV path for one PID definition. | Scope: reusable/public. Related: safe_filename. |
| `append_row_to_csv (line 76)` | row: dict, output_folder: str \| Path, csv_title: str='pid_results' | Path | Append one simulation row to its PID-specific CSV. | Scope: reusable/public. Related: get_pid_ver_csv_path. |
| `already_exists_in_csv (line 104)` | output_folder: str \| Path, n_samples: int, dimensions: tuple[int, int, int] \| list[int], pid_ver: str, seed: int, csv_title: str='pid_results' | bool | Check whether one exact simulation setting is already recorded. | Scope: reusable/public. Related: get_pid_ver_csv_path. |
| `sample_data_from_cov (line 144)` | config: dict, true_cov: torch.Tensor, rng: np.random.Generator \| torch.Generator \| None=None | tuple[torch.Tensor, list[torch.Tensor]] | Sample Gaussian RVs and their unbiased empirical covariance. | Scope: reusable/public. Conventions: order [X1, X2, T]. |
| `make_pre_config (line 177)` | exp: str, mi_config: dict, mi0_config: dict, above0_m7_mi_config: dict, above0_m8_mi_config: dict, n_p_config: dict, unknown_config: dict, de_config: dict \| None=None | dict | Merge the configuration fragments for one Idep simulation regime. | Scope: reusable/public. |


## Partial_Information_Decomposition

### `Partial_Information_Decomposition/PID_calc.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `mi_wishart_bias (line 38)` | dims: list, n_samples: int | Unannotated return value; inspect implementation before reuse | Calculate Gaussian MI Wishart biases without loading simulation dependencies. | Scope: reusable/public. |
| `pid_calc (line 71)` | config=None, sources=None, target=None, rng=torch.Generator().manual_seed(56), method=None, on_rvs: callable=None, covariance: torch.Tensor=None, param_bias=False | Unannotated return value; inspect implementation before reuse | Dispatch standard PID inputs to Idep, Tilde, Delta, Thin, Flow, or Gaussian Eigen-PID and return '(pid, mi)'. Eigen-PID is selected with 'method="eigen"' or 'method="eigen_pid"'. | Scope: reusable/public. Related: pid_idep_wrapper, pid_tilde_wrapper, delta_wrapper, thin_pid_wrapper. |
| `pid_idep_wrapper (line 140)` | config, sources=None, target=None, covariance=None, rng=None, on_rvs=None | Unannotated return value; inspect implementation before reuse | This function is a wrapper to PID calculated by Idep_multivariate_gauss class, which implements the Idep PID calculation for multivariate Gaussian variables. This wrapper allows us to use the same input format for both Idep and BROJA implementations, and also allows us to apply transformations to the random variables before PID calculation if needed. if covariance is provided, it will used to calculate the PID directly from the covariance matrix without sampling. If covariance is not provided, the PID will be calculated from the sampled data covariance. | Scope: reusable/public. Related: idep. |
| `pid_tilde_wrapper (line 168)` | config: dict, sources: list, target: list, covariance: torch.Tensor, rng: torch.random.Generator, on_rvs: callable=None, param_bias=False | Unannotated return value; inspect implementation before reuse | Calculate Gaussian BROJA/Tilde PID with an optional objective correction. | Scope: reusable/public. Related: create_cov_matrix, lorenz_gaussian_obj_debias, permutation_null_debias. Conventions: order [T, X1, X2]. |
| `delta_wrapper (line 331)` | config, sources, target, covariance, rng, on_rvs | Unannotated return value; inspect implementation before reuse | This function is a wrapper to PID calculated by BROJA and implemented by Venkatesh et al. 2023 Because Idep and BROJA have different input format, this wrapper converts the input format to fit the BROJA implementation and then calls the PID calculation function. and calculates the PID using BROJA definition and calculation from Venkateh et al. 2023. | Scope: reusable/public. Related: create_cov_matrix, calculate_mi_raw, mi_wishart_bias. |
| `_to_numpy_samples (line 387)` | data | Unannotated return value; inspect implementation before reuse | Convert torch/numpy samples to the numpy format expected by flow-pid. | Scope: private helper. |
| `thin_pid_wrapper (line 397)` | config: dict, sources: list, target: list, covariance: torch.Tensor, rng: torch.random.Generator, on_rvs: callable=None | Unannotated return value; inspect implementation before reuse | Calculate Thin-PID from samples or covariance using the standard PID inputs. | Scope: reusable/public. Related: load_exact_gauss_thin_pid, create_cov_matrix. |
| `eigen_pid_wrapper (line 428)` | config: dict, sources: list, target: list, covariance: torch.Tensor, rng: torch.Generator, on_rvs: callable=None | Unannotated return value; inspect implementation before reuse | Calculate Gaussian Eigen-PID using the standard PID_calc inputs. | Scope: reusable/public. Related: create_cov_matrix. Conventions: order [T, X1, X2], bias-correction aware. |
| `flow_pid_wrapper (line 539)` | config: dict, sources: list, target: list, covariance: torch.Tensor, rng: torch.random.Generator, on_rvs: callable=None | Unannotated return value; inspect implementation before reuse | Wrapper for flow-pid. | Scope: reusable/public. Related: load_exact_gauss_thin_pid, load_flow_pid, _to_numpy_samples. Conventions: bias-correction aware. |

### `Partial_Information_Decomposition/PID_util.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `LinearRegression_fit (line 11)` | X, y | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: fit. |
| `compute_ridge_cv_r2 (line 25)` | X, y, alphas=None | Unannotated return value; inspect implementation before reuse | Compute cross-validated R² using RidgeCV with efficient LOO cross-validation. | Scope: reusable/public. Related: fit. |
| `cond_cov (line 61)` | sigma_1, sigma_2, sigma12, sigma21 | Unannotated return value; inspect implementation before reuse | This function will compute the conditional covariance matrix of two Gaussian variables Sigma_1\|2 = Sigma_1 - Sigma12*inv(Sigma_2)*Sigma21 | Scope: reusable/public. |
| `ledoit_wolf_cov_torch (line 75)` | X: torch.Tensor, assume_centered: bool=False | torch.Tensor | Fit Ledoit-Wolf on X (N,פ) and return covariance as torch.Tensor on same device/dtype. | Scope: reusable/public. Related: fit. |
| `create_cov_matrix (line 87)` | rvs: list=[], verbose=False, Sigma=None, dims: list=None, device='cpu', check_singular=True | Unannotated return value; inspect implementation before reuse | This function will create the covariance matrix for the three variables M1,M2,T input: M1,M2,T are torch tensors of shape (N,p) rvs is a list of the three variables [M1,M2,T] N is the number of observations, p is the dimension of each observation. | Scope: reusable/public. Related: eigvenvalue_summary, block_singularity_check. |
| `reorder_cov_blocks (line 152)` | Sigma: torch.Tensor, dims: dict[str, int], old_order: list[str], new_order: list[str] | torch.Tensor | Reorder covariance matrix blocks according to variable names. | Scope: reusable/public. Conventions: order [X1, X2, T], order [T, X1, X2]. |
| `para_create_cov_matrix (line 193)` | dims, Sigmas=None, verbose=False | Unannotated return value; inspect implementation before reuse | This function will create the covariance matrix for the three variables M1,M2,T | Scope: reusable/public. |
| `old_para_create_cov_matrix (line 241)` | dims, Sigmas=None, verbose=False | Unannotated return value; inspect implementation before reuse | This function will create the covariance matrix for the three variables M1,M2,T input: M1,M2,T are torch tensors of shape (N,p) rvs is a list of the three variables [M1,M2,T] N is the number of observations, p is the dimension of each observation. | Scope: reusable/public. |
| `whiten_block (line 290)` | Sigma_xx: torch.Tensor, Sigma_xy: torch.Tensor, Sigma_yy: torch.Tensor | torch.Tensor | return Ux^{-T} @ Sigma_xy @ Uy^{-1} where Sigma_xx = Ux^T Ux, Sigma_yy = Uy^T Uy, and Ux,Uy are upper triangular. | Scope: reusable/public. Conventions: whitening-sensitive. |
| `para_whiten_block (line 307)` | Sigma_xx: torch.Tensor, Sigma_xy: torch.Tensor, Sigma_yy: torch.Tensor | torch.Tensor | Computes: Ux^{-T} @ Sigma_xy @ Uy^{-1} where Sigma_xx = Ux^T Ux, Sigma_yy = Uy^T Uy, and Ux,Uy are upper triangular. Supports batched inputs of shape (N, d, d). | Scope: reusable/public. Related: whiten_block. Conventions: whitening-sensitive. |
| `plot_cov_blocks (line 332)` | cov_dict, x0_dim, x1_dim, x2_dim, *, title='Covariance (block view)', cmap='Blues', vmin=None, vmax=None, fine_grid=False, show_colorbar=True | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `standardize (line 373)` | X: torch.Tensor, eps: float=1e-12 | torch.Tensor | Standardize columns of X to zero mean and unit variance. | Scope: reusable/public. |
| `assert_full_rank (line 385)` | X: torch.Tensor, jitter=0 | None | Assert that the input matrix X is full rank. | Scope: reusable/public. |
| `correlation_matrix (line 409)` | X | Unannotated return value; inspect implementation before reuse | Compute the correlation matrix of the columns of X. | Scope: reusable/public. |
| `block_singularity_check (line 424)` | X, tol=1e-10 | Unannotated return value; inspect implementation before reuse | Check if a block is singular or ill-conditioned. | Scope: reusable/public. |
| `singularity_report (line 445)` | X_M1, X_M2, y_real, tol=1e-10, return_printing_required=False | Unannotated return value; inspect implementation before reuse | Return min eigenvalue and singularity flag for blocks and combinations. | Scope: reusable/public. Related: block_singularity_check, correlation_matrix. |
| `diagnostic_plots (line 475)` | X_M1, X_M2, y_real, method, mixing_dimension | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: cross_correlation. |
| `diagnostic_plots.cross_correlation (line 476)` | X, Y | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `vif_summary (line 510)` | X | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `std_scaling_summary (line 542)` | X | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `eigvenvalue_summary (line 561)` | X | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `_get_first (line 569)` | mapping, *keys | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `compare_results (line 576)` | vp_results, pid_results, mi_results=None | No explicit return; likely None / side effects | Compare Variance Partitioning and Partial Information Decomposition results. | Scope: reusable/public. Related: _get_first. |
| `pid_comparison_table (line 622)` | results: dict, decimals: int=4, print_table: bool=True | Unannotated return value; inspect implementation before reuse | Print and return a compact table comparing PID and MI outputs. | Scope: reusable/public. Related: _get_first. |
| `save_pid_comparison_table (line 651)` | results: dict, save_path: str, decimals: int=4, title: str='PID Method Comparison', config: dict=None | Unannotated return value; inspect implementation before reuse | Save the PID comparison table as a clean matplotlib image with compact title, run-metadata placement, and slightly reduced table text for longer method labels. | Scope: reusable/public. Related: pid_comparison_table. |
| `commonality_comparison_table (line 695)` | results: dict, decimals: int=4, print_table: bool=True | list[dict] | Normalize key variants and optionally print commonality-analysis values without displaying the internal example-label column. | Scope: reusable/public rendering helper. Related: _get_first, save_commonality_comparison_table, commanility_analysis. Conventions: returned rows retain example metadata and accept R²/unique/common key capitalization variants. |
| `save_commonality_comparison_table (line 773)` | results: dict, save_path: str, decimals: int=4, title: str='Commonality Analysis Comparison', config: dict \| None=None | str | Save normalized commonality values without an Example column as a Matplotlib table with run metadata. | Scope: reusable/public rendering helper. Related: commonality_comparison_table. |
| `commanility_analysis (line 870)` | results: dict, decimals: int=4, print_table: bool=True | list[dict] | Preserve the legacy misspelled commonality-table entry point. | Scope: backward-compatible public wrapper. Related: commonality_comparison_table. |
| `plot_mi_heatmap (line 893)` | csv_path, value_col, *, n_col='N', p_col='p', figsize=(7, 5), title=None, save_path=None, annotate=True, fmt='.3f', cmap='viridis' | No explicit return; likely None / side effects | Plot a block heatmap from an averaged CSV. | Scope: reusable/public. |
| `plot_all_mi_heatmaps (line 1013)` | csv_path, title='Mutual Information Heatmaps', *, n_col='N', p_col='p', figsize=(16, 5), save_path=None, annotate=True, mean_fmt='.2f', std_fmt='.2f', log_scale=False, cmap='viridis', annotation_mode='pm', fontsize=9, aggfunc='mean' | No explicit return; likely None / side effects | Plot theoretical, naive, and bias-corrected MI heatmaps in one figure. | Scope: reusable/public. |
| `plot_block_heatmap (line 1180)` | csv_path, save_path=None | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `anchored_oas_shrinkage (line 1240)` | Sigma_full: torch.Tensor, cov_loo_all: torch.Tensor, n_samples: int | Unannotated return value; inspect implementation before reuse | Calculates OAS parameters ONCE on the full matrix, and applies the EXACT SAME linear shrinkage to all LOO matrices. | Scope: reusable/public. |
| `oas_cov_torch (line 1281)` | S: torch.Tensor, N: int | torch.Tensor | Apply Oracle Approximating Shrinkage (OAS) to a covariance matrix. Requires ONLY the sample covariance matrix S and sample size N. | Scope: reusable/public. |
| `residual_rvs (line 1312)` | rv_list: list, predictor_index=0 | Unannotated return value; inspect implementation before reuse | Given a list of random variables (Torch.Tensors), returns a list where we predict the second rv using the first rv and return the residuls. | Scope: reusable/public. Related: compute_ridge_cv_r2. |

### `Partial_Information_Decomposition/__init__.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.

### `Partial_Information_Decomposition/bias_functions.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `logdet_wishart_bias (line 22)` | df: int, d: int | float | Exact finite-sample bias for log\|S\| when S is the unbiased sample covariance from Gaussian data and (df) * S ~ Wishart_d(Sigma, df). | Scope: reusable/public. Conventions: float64. |
| `mi_wishart_bias (line 42)` | dims: list, n_samples: int | Unannotated return value; inspect implementation before reuse | Bias correction for Gaussian mutual information estimates from unbiased sample covariance. | Scope: reusable/public. Related: logdet_wishart_bias. Conventions: order [X1, X2, T]. |
| `permuteation_debiased (line 101)` | config, term='nume' | Unannotated return value; inspect implementation before reuse | Evaluate an M7 MI term from permuted '[X1, X2, T]' samples, reusing the already-whitened M7 covariance blocks directly. | Scope: reusable/public. Related: create_cov_matrix, create_m7_cov, calcualte_mi. Conventions: whitening-sensitive. |
| `broja_venkatesh_bias (line 124)` | config | Unannotated return value; inspect implementation before reuse | Calculate one raw Gaussian BROJA statistic for permutation debiasing. | Scope: reusable/public. Related: create_cov_matrix. Conventions: bias-correction aware. |
| `permutation_null_debias (line 168)` | config, func | Unannotated return value; inspect implementation before reuse | Debias an MI-like estimator by subtracting its permutation null floor. | Scope: reusable/public. |
| `unique_bias (line 241)` | config, functions_dict: dict=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `bias_func (line 261)` | config, model | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: logdet_wishart_bias, permutation_null_debias. |
| `parametric_bootstrap_obj_debias (line 291)` | config: dict, covariance=None, raw_obj: float \| None=None, statistic: str='obj' | dict | Estimate Gaussian BROJA objective or synergy bias by parametric bootstrap. | Scope: reusable/public. Related: create_cov_matrix. Conventions: order [T, X1, X2]. |
| `lorenz_gaussian_obj_debias (line 634)` | config: dict, covariance=None, raw_obj: float \| None=None | dict | Apply the Lorenz et al. Gaussian merged correction to raw BROJA ''obj''. | Scope: reusable/public. Related: parametric_bootstrap_obj_debias, permutation_null_debias, mi_wishart_bias, create_cov_matrix. Conventions: order [T, X1, X2]. |
| `equal_direct_wishart_control_obj_debias (line 965)` | config: dict, covariance, raw_obj: float \| None=None | dict | Estimate bias of the raw Venkatesh objective under a known direct tie. | Scope: reusable/public. Related: mi_wishart_bias. Conventions: order [T, X1, X2]. |
| `equivalent_channels_obj_debias (line 1341)` | config: dict, covariance=None, raw_obj: float \| None=None | dict | Debias Gaussian BROJA PID under a known equivalent-channel constraint. | Scope: reusable/public. Related: mi_wishart_bias, create_cov_matrix. Conventions: order [T, X1, X2]. |

### `Partial_Information_Decomposition/heatmap_plot.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `title_case_words (line 101)` | value | Unannotated return value; inspect implementation before reuse | Convert names like: 'idep_gaussian' -> 'Idep Gaussian' 'm7_pid' -> 'M7 PID' | Scope: reusable/public. |
| `make_full_title (line 132)` | base_title, pid_ver_name, component_name | Unannotated return value; inspect implementation before reuse | Build the full plot title. | Scope: reusable/public. |
| `find_column (line 149)` | df, component_key, stat_name | Unannotated return value; inspect implementation before reuse | Find the correct column for a component and statistic. | Scope: reusable/public. |
| `make_p_column (line 202)` | df, p_col='p' | Unannotated return value; inspect implementation before reuse | Create a p-like column from dx1, dx2, dt if p does not already exist. | Scope: reusable/public. |
| `display_p_label (line 230)` | v | Unannotated return value; inspect implementation before reuse | Pretty display for p values on the y-axis. | Scope: reusable/public. |
| `sort_p_index (line 244)` | values | Unannotated return value; inspect implementation before reuse | Sort p values numerically when they are tuples like: (dx1, dx2, dt) | Scope: reusable/public. |
| `sort_p_index.key (line 250)` | v | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `optional_pivot_table (line 258)` | df, *, index, columns, values, aggfunc, reference_index, reference_columns | Unannotated return value; inspect implementation before reuse | Build a pivot table for an optional statistic and align it with the mean matrix. | Scope: reusable/public. |
| `plot_single_component_heatmap (line 287)` | df, *, pid_ver, component_key, base_title=None, x_col='N', y_col='p', aggfunc='last', cmap='viridis', figsize=(9, 7), save_dir=None, show=True, mean_fmt='.3f', std_fmt='.3f' | Unannotated return value; inspect implementation before reuse | Plot one heatmap for one PID version and one component. | Scope: reusable/public. Related: title_case_words, find_column, optional_pivot_table, make_full_title. |
| `plot_pid_and_mi_heatmaps_from_csv (line 508)` | csv_path, *, base_title=None, save_dir=None, pid_versions=None, components=('red', 'unq1', 'unq2', 'syn', 'mi_x1_t', 'mi_x2_t', 'mi_x1x2_t', 'mi_m7', 'mi_m8'), x_col='N', y_col='p', seed=None, aggfunc='last', cmap='viridis', figsize=(9, 7), show=True | Unannotated return value; inspect implementation before reuse | Read a checkpoint CSV and create heatmaps for PID components and mutual information values. | Scope: reusable/public. Related: make_p_column, plot_single_component_heatmap. |

### `Partial_Information_Decomposition/mi_functions.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `mi_calculation_not_whiten (line 16)` | config | float | Compute MI from covariance matrices using the formula: MI = 0.5 * (log\|deno_matrix\| - log\|nume_matrix\|) | Scope: reusable/public. Related: para_create_cov_matrix, safe_logdet. Conventions: whitening-sensitive. |
| `safe_logdet (line 73)` | A: torch.Tensor | float | Compute log determinant and raise if matrix is not positive definite. | Scope: reusable/public. |
| `np_safe_logdet (line 91)` | A, eps=1e-08 | Unannotated return value; inspect implementation before reuse | Stable logdet for covariance matrices. | Scope: reusable/public. |
| `calcualte_mi (line 103)` | config, sigma_dict, term='full' | Unannotated return value; inspect implementation before reuse | This function calculates the tri-variate mutual information using the covariance matrices and the formula MI = 0.5 * (log\|deno_matrix\| - log\|nume_matrix\|) | Scope: reusable/public. Related: safe_logdet. |
| `calculate_mi_raw (line 129)` | device: torch.device, sigma: torch.Tensor, dims: list | Unannotated return value; inspect implementation before reuse | This function calculates the tri-variate or bi-variate mutual information using the covariance matrices without any whitening - in raw mode (: | Scope: reusable/public. Related: create_cov_matrix, safe_logdet. Conventions: whitening-sensitive. |
| `para_calcualte_mi (line 184)` | config, sigma_dict, term='full', assumed_whitened=True | Unannotated return value; inspect implementation before reuse | This function calculates the tri-variate mutual information using the for multiple covariances matrices and the formula MI = 0.5 * (log\|deno_matrix\| - log\|nume_matrix\|) | Scope: reusable/public. Related: safe_logdet. Conventions: whitening-sensitive. |
| `calculate_mi_lr (line 215)` | config, sigma_dict | Unannotated return value; inspect implementation before reuse | This function calculates the trivarite (X1;X2,T) mutual information using the covaraince matrix especially for functions that use linear regression. The function above uses matrices that ill-conditioned using linear regression. Therefore we use the next equations: [logdetΣX-logdet(Σ1\|T)-logdet(Σ2\|T)] where X = joint_cov_x1_x2 | Scope: reusable/public. Related: safe_logdet. |
| `mi_wrapper (line 252)` | config, sigma_dict, whiten_terms_dict, tri_variate=True | Unannotated return value; inspect implementation before reuse | This function is a wrapper for the mutual information calculation functions. It takes in the config and sigma_dict and calls the appropriate function based on the mi_type argument. | Scope: reusable/public. Related: calcualte_mi, calculate_mi_lr. Conventions: whitening-sensitive. |
| `pid_components (line 277)` | pid_config, print_results=False | Unannotated return value; inspect implementation before reuse | Calculate PID components with the known components. | Scope: reusable/public. |

### `Partial_Information_Decomposition/output_utils.py`

File description: Small output/path helpers shared by PID plotting and simulation code.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `safe_filename (line 4)` | name | Unannotated return value; inspect implementation before reuse | Return the filename exactly as given. Nothing is deleted or replaced. | Scope: reusable/public. |

### `Partial_Information_Decomposition/tilde_ohad.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `calculate_tilde_union_info (line 22)` | hx, hy, reg=1e-07, max_iters=20000, verbose=False | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |


## Simulations/Encoder_simulation

### `Simulations/Encoder_simulation/Both_Unique_Encodrs.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `get_run_config (line 20)` | No inputs | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `creature_featurs (line 46)` | rng, snr, unique_ratio, features, signal, redundant_dim=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `run_single_seed (line 76)` | seed: int, config: dict, features: torch.Tensor, fmri_dict: dict | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. Related: prepare_inputs, create_predictions, creature_featurs, commonality_analysis. |
| `main (line 105)` | No inputs | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: get_run_config, load_model_and_fmri, run_configured_multiseed, run_single_seed. |

### `Simulations/Encoder_simulation/both_unique.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `get_run_config (line 20)` | No inputs | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `half_permute (line 36)` | rng, features, snr=10 | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `orthogonal_vectors (line 57)` | rng, n, p, features, noise=None, singal=None, unique_ratio=None, function=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `feature_creation (line 98)` | rng, unique_ratio, unique_method='orthogonal', n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', show_diagnostic_plots=False | Unannotated return value; inspect implementation before reuse | Creates dummy predictors and a target | Scope: task-specific. Related: half_permute, orthogonal_vectors. |
| `test_both_unique (line 139)` | rng, unique_ratio, n=1024, p=100, snr=10.0, method='standard' | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. Related: feature_creation, commonality_analysis, idep. Conventions: bias-correction aware. |
| `run_single_seed (line 150)` | seed: int, config: dict | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. Related: test_both_unique, extract_all_components. |
| `main (line 164)` | No inputs | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: get_run_config, run_configured_multiseed. |

### `Simulations/Encoder_simulation/turned_off_unqiue.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `get_run_config (line 28)` | No inputs | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `feature_creation (line 48)` | rng, r_str, u1_str, u2_str, unique_method='orthogonal', n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', show_diagnostic_plots=False | Unannotated return value; inspect implementation before reuse | Creates dummy predictors and a target | Scope: task-specific. |
| `test (line 86)` | rng, r_str, u1_str, u2_str, n=1024, p=100, snr=10.0, method='standard' | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. Related: feature_creation, standardize_np, commonality_analysis, extract_betas. |
| `extract_betas (line 101)` | ca_results | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `run_single_seed (line 120)` | seed: int, config: dict | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. Related: test, extract_all_components. |
| `test_regularization_term (line 140)` | seed: int, config: dict | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. Related: feature_creation, standardize_np, commonality_analysis, idep. |
| `save_term_results_csv (line 169)` | x_axis: str, term_results: dict, output_csv_path: str \| Path | Path | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. Related: extract_all_components. |
| `test_u2str (line 195)` | seed: int, config: dict, final_ratio: float | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. Related: test. |
| `plot_keys_vs_alpha (line 226)` | csv_path: Union[str, Path], keys: Sequence[str], *, x_col: str='alpha', sort_alpha: bool=False, logx: bool=False, figsize: tuple[float, float]=(8, 4.5), marker: Optional[str]=None, save_path: Optional[Union[str, Path]]=None | None | Plot selected columns (keys) vs x_col from a CSV file. | Scope: task-specific. |
| `main (line 290)` | No inputs | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: get_run_config, run_configured_multiseed, run_single_seed. |


## Simulations/PCA_Ridge

### `Simulations/PCA_Ridge/pid_feature_middleware.py`

File description: Shared PCA, Ridge-CV, theoretical-PID, and trial-loop helpers.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `pca_target (line 33)` | target: Any, shared_mask: np.ndarray, n_components_target: int, random_state: int=0 | tuple[np.ndarray, np.ndarray] | Fit target PCA and return its scores and population linear map. | Scope: task-specific. Related: fit. |
| `pca_sources (line 63)` | source_1: Any, source_2: Any, target: Any, shared_mask: np.ndarray, n_components_source_1: int, n_components_source_2: int, random_state: int=0 | tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | Fit source PCAs and return held-out arrays and population linear maps. | Scope: task-specific. Related: fit. |
| `_ridge_prediction_and_map (line 108)` | source_train: np.ndarray, target_train: np.ndarray, source_test: np.ndarray, alphas: np.ndarray | tuple[np.ndarray, np.ndarray] | Return native per-target-alpha RidgeCV predictions and RAW-input map. | Scope: private helper. Related: fit. |
| `ridge_sources_on_target (line 157)` | source_1: Any, source_2: Any, target: Any, shared_mask: np.ndarray, alphas: np.ndarray | tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | Fit per-target-PC Ridge-CV models and return arrays and linear maps. | Scope: task-specific. Related: _ridge_prediction_and_map. |
| `expand_independent_covariance (line 195)` | coordinate_covariance: torch.Tensor, n_replicas: int | torch.Tensor | Expand independent coordinate covariance into grouped variable order. | Scope: task-specific. Conventions: order [X1, X2, T]. |
| `transform_population_covariance (line 219)` | covariance: torch.Tensor, linear_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | torch.Tensor | Propagate [X1, X2, T] covariance through three fitted linear maps. | Scope: task-specific. Conventions: order [X1, X2, T], float64. |
| `standardize_covariance (line 246)` | covariance: torch.Tensor | torch.Tensor | Convert a covariance to a correlation matrix without changing PID. | Scope: task-specific. Conventions: float64. |
| `calculate_theoretical_pid (line 271)` | covariance: torch.Tensor, dims: list[int], method: str | dict[str, float] | Calculate exact Gaussian PID and MI values from a population covariance. | Scope: task-specific. Related: standardize_covariance, calculate_mi_raw, pid_calc, change_covariance_order. Conventions: order [X1, X2, T]. |
| `prepare_pid_routes (line 315)` | source_1: Any, source_2: Any, target: Any, shared_mask: np.ndarray, n_components: int, seed: int, population_covariance: torch.Tensor, population_dims: list[int] | dict[str, tuple[tuple[np.ndarray, np.ndarray, np.ndarray], torch.Tensor]] | Prepare held-out PID arrays and exact route population covariances. | Scope: task-specific. Related: pca_target, pca_sources, ridge_sources_on_target, transform_population_covariance. Conventions: order [X1, X2, T]. |
| `run_pid_feature_comparison (line 375)` | sample_for_seed: Callable[[int], tuple[Any, Any, Any]], population_covariance: torch.Tensor, population_dims: list[int], *, n_samples: int, n_train: int, n_components: int, n_trials: int, base_seed: int, pid_method: str, bias_correction: bool, experiment_name: str, plot_path: str \| Path, plot_title: str, metadata: dict[str, Any] \| None=None | dict[str, dict[str, dict[str, float]]] | Run a seeded RAW/PCA/Ridge-CV PID comparison and save its table plot. | Scope: task-specific. Related: calculate_theoretical_pid, save_sample_simulation_results_table, prepare_pid_routes, print_pid_mi. Conventions: order [X1, X2, T], bias-correction aware. |
| `build_sonic_covariance (line 507)` | p: int | torch.Tensor | Build the full Sonic covariance in grouped [X1, X2, T] order. | Scope: task-specific. Related: expand_independent_covariance. Conventions: order [X1, X2, T], float64. |

### `Simulations/PCA_Ridge/run_pca_ridge.py`

File description: Run any registered PCA–Ridge PID comparison through one command-line entry point.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `parse_args (line 18)` | No inputs | argparse.Namespace | Parse common PCA–Ridge simulation arguments. | Scope: entry point. |
| `run_scenario (line 47)` | scenario_name: str, *, n_samples: int=10000, n_train: int=9000, p: int=70, noise_std: float=1.0, n_components: int \| None=None, n_trials: int=2, base_seed: int=0, pid_method: str \| None=None, bias_correction: bool=True, output_dir: Path \| str=PROJECT_ROOT / 'Simulations/PCA_Ridge/results' | dict[str, Any] | Run one registered PCA–Ridge scenario with shared orchestration. | Scope: task-specific. Related: get_scenario, run_pid_feature_comparison. Conventions: bias-correction aware. |
| `main (line 127)` | No inputs | dict[str, Any] | Run the scenario selected on the command line. | Scope: entry point. Related: parse_args, run_scenario. |

### `Simulations/PCA_Ridge/scenarios.py`

File description: Scenario definitions shared by the consolidated PCA–Ridge runner.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `PcaRidgeScenario (line 32)` | class | PcaRidgeScenario instance | Describe one PCA–Ridge simulation without duplicating runner code. | Scope: class. |
| `concatenated_equal_unique (line 63)` | rng: np.random.Generator, n: int, p: int, noise_std: float | tuple[np.ndarray, np.ndarray, np.ndarray] | Generate equal unique information in separate concatenated target blocks. | Scope: task-specific. |
| `build_all_above_zero_covariance (line 94)` | p: int, noise_std: float, unique1_weight: float, unique2_weight: float, redundant_weight: float, shared_noise_weight: float | torch.Tensor | Build the all-above-zero covariance in grouped ''[X1, X2, T]'' order. | Scope: task-specific. Related: expand_independent_covariance. Conventions: order [X1, X2, T], float64. |
| `build_concatenated_all_above_zero_covariance (line 143)` | p: int, noise_std: float, redundant_weight: float, shared_noise_weight: float | torch.Tensor | Build the concatenated all-above-zero covariance. | Scope: task-specific. Related: expand_independent_covariance. Conventions: float64. |
| `build_concatenated_equal_unique_covariance (line 180)` | p: int, noise_std: float | torch.Tensor | Build the concatenated equal-unique covariance. | Scope: task-specific. Related: expand_independent_covariance. Conventions: float64. |
| `build_equal_unique_covariance (line 210)` | p: int, noise_std: float | torch.Tensor | Build the equal-unique covariance. | Scope: task-specific. Related: expand_independent_covariance. Conventions: float64. |
| `build_full_suppresion_covariance (line 232)` | p: int, noise_std: float | torch.Tensor | Build the full-suppression covariance. | Scope: task-specific. Related: expand_independent_covariance. Conventions: float64. |
| `build_unq2_zero_covariance (line 255)` | p: int, noise_std: float | torch.Tensor | Build the zero-source-2-unique covariance. | Scope: task-specific. Related: expand_independent_covariance. Conventions: float64. |
| `get_scenario (line 364)` | name: str | PcaRidgeScenario | Return one registered PCA–Ridge scenario. | Scope: task-specific. |


## Simulations/PCA_rank

### `Simulations/PCA_rank/eigenvector_pca.py`

File description: Numerical eigenvector PCA cross-validation and loading estimators. Checkpoint persistence is delegated to `eigenvector_pca_wrapper.py` so existing callers retain the optional `checkpoint_csv_path` interface without carrying CSV implementation here.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `EigenvectorPCACVResult (line 41)` | class | EigenvectorPCACVResult instance | Store the selected PCA rank, PRESS/MSEP vectors, data dimensions, elapsed time, and evaluated maximum rank. | Scope: task-specific result class. |
| `fit_pca_loadings_svd (line 51)` | X_train: np.ndarray, n_components: int | np.ndarray | Fit PCA loadings with NumPy SVD. | Scope: reusable within PCA-rank selection. Convention: samples by features input; features by components output. |
| `_eigenvector_pca_cv_sample_press (line 65)` | X: np.ndarray, sample_index: int, max_components: int, pca_fit_fn: PCAFitFunction, center: bool, scale: bool, include_zero_components: bool, method_pca: str \| None, eps: float | np.ndarray | Calculate one held-out sample's PRESS contribution, using the fast leverage formula for orthonormal loadings and a direct pseudoinverse fallback otherwise. | Scope: private numerical helper. |
| `eigenvector_pca_cv (line 149)` | X: np.ndarray, max_components: int \| None=None, pca_fit_fn: PCAFitFunction \| None=None, center: bool=True, scale: bool=False, include_zero_components: bool=True, method_pca: str \| None=None, eps: float=1e-12, checkpoint_csv_path: str \| Path \| None=None | EigenvectorPCACVResult | Run eigenvector cross-validation for PCA component selection. | Scope: task-specific public entry point. Related: _eigenvector_pca_cv_sample_press and the checkpoint wrapper functions. Convention: samples by features input. |
| `fit_pca_loadings_sklearn (line 265)` | X_train: np.ndarray, n_components: int | np.ndarray | Fit PCA loadings with sklearn's full SVD solver. | Scope: reusable within PCA-rank selection. Related: fit. Convention: features by components output. |
| `regular_PCA (line 285)` | X: np.ndarray, variance_threshold: float | np.ndarray | Fit sklearn PCA using a variance threshold. | Scope: task-specific. Related: fit. Convention: features by selected components output. |

### `Simulations/PCA_rank/eigenvector_pca_wrapper.py`

File description: Checkpoint hashing, schema validation, CSV resume loading, and atomic CSV persistence for eigenvector PCA cross-validation.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `_checkpoint_fields (line 31)` | max_components: int | tuple[list[str], list[str]] | Build the fixed metadata, sample-index, and PRESS columns for a checkpoint CSV. | Scope: private persistence helper. |
| `build_checkpoint_metadata (line 45)` | X: np.ndarray, max_components: int, pca_fit_fn: PCAFitFunction, center: bool, scale: bool, include_zero_components: bool, method_pca: str \| None, eps: float | dict[str, str] | Hash the complete input data and serialize strict run settings used to reject incompatible checkpoint resumes. | Scope: checkpoint wrapper helper. Related: eigenvector_pca_cv. |
| `load_eigenvector_pca_checkpoint (line 99)` | checkpoint_csv_path: str \| Path, metadata: dict[str, str], max_components: int | dict[int, np.ndarray] | Validate and load completed per-sample PRESS vectors from a checkpoint CSV. | Scope: checkpoint wrapper helper. Related: _checkpoint_fields. Convention: each PRESS vector has shape `(max_components + 1,)`. |
| `write_eigenvector_pca_checkpoint (line 161)` | checkpoint_csv_path: str \| Path, metadata: dict[str, str], completed_press: dict[int, np.ndarray], max_components: int | None | Atomically replace the checkpoint CSV with all completed per-sample PRESS rows. | Scope: checkpoint wrapper helper. Related: _checkpoint_fields. |

### `Simulations/PCA_rank/pca_simulation.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `create_T_and_P (line 30)` | n_samples: int, n_features: int, rank: int, loading_corr: float=0.0, component_strengths: np.ndarray \| list[float] \| None=None, random_state: int \| None=None | tuple[np.ndarray, np.ndarray] | Create scores and correlated loadings. Inputs define sizes, rank, correlation, strengths, and seed; outputs are T and P arrays. | Scope: task-specific. |
| `generate_rank_simulation_data (line 69)` | n_samples: int, n_features: int, rank: int, loading_corr: float, noise_std: float, random_state: int, component_strengths: list[float] \| np.ndarray \| None=None, center_columns: bool=True | dict | Generate noisy known-rank data. Inputs define the condition; output contains X, T, P, normalized signal, and metadata. | Scope: task-specific. Related: create_T_and_P. |
| `run_rank_simulation (line 95)` | grid: dict[str, list], output_dir: str \| Path=OUTPUT_DIR, nbsim: int=NBSIM | tuple[pd.DataFrame, pd.DataFrame] | Run EM/K-fold rank selection. Inputs are grid, output path, and K-fold count; outputs are raw/summary tables and saved CSV/heatmap files. | Scope: task-specific. Related: estimate_ncp_pca, generate_rank_simulation_data. |

### `Simulations/PCA_rank/rowwise_PCA.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `RowwiseLOOVariancePCAResult (line 10)` | class | RowwiseLOOVariancePCAResult instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. |
| `rowwise_loo_pca_variance_threshold (line 23)` | X: np.ndarray, variance_threshold: float=0.99 | RowwiseLOOVariancePCAResult | Row-wise leave-one-out PCA where the number of PCs is chosen by explained variance threshold inside each fold. | Scope: task-specific. Related: fit. |

### `Simulations/PCA_rank/run_simulation.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `run_simulation (line 13)` | n_samples: int, n_features: int, rank: int, loading_corr: float, noise_std: float, random_state: int | dict | Run PCA simulations with eigenvector and row-wise LOOCV methods. | Scope: task-specific. Related: generate_rank_simulation_data, estimate_ncp_pca, eigenvector_pca_cv, rowwise_loo_pca_variance_threshold. |


## Simulations/Theoretical_Examples/Covariance

### `Simulations/Theoretical_Examples/Covariance/compare_multidim_gaussian_pid.py`

File description: Compare population Gaussian PID methods on a flexible channel model.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `save_run_constants (line 93)` | yaml_path: str \| Path | Path | Save every uppercase script constant as a YAML run snapshot. | Scope: task-specific. |
| `validate_parameters (line 140)` | target_dim: int, source1_dim: int, source2_dim: int, source1_gain: float, source2_gain: float, source1_noise_variance: float, source2_noise_variance: float | None | Validate dimensions, gains, and source-noise variances. | Scope: task-specific. |
| `build_channel_matrix (line 192)` | source_dim: int, target_dim: int, special_gain: float, special_coordinate: int | torch.Tensor | Build a rectangular channel by repeating a two-coordinate gain block. | Scope: task-specific. Related: rectangular_identity. Conventions: float64. |
| `build_population_covariance (line 226)` | target_dim: int, source1_dim: int, source2_dim: int, source1_gain: float, source2_gain: float, source1_noise_variance: float, source2_noise_variance: float | torch.Tensor | Construct the theoretical Gaussian covariance in ''[T, X1, X2]'' order. | Scope: task-specific. Related: validate_parameters, build_channel_matrix. Conventions: order [T, X1, X2], float64. |
| `run_pid_methods (line 292)` | covariance: torch.Tensor, target_dim: int, source1_dim: int, source2_dim: int | dict[str, dict[str, float]] | Run GPID Tilde, Thin-PID, and Eigen-PID on one population covariance. | Scope: task-specific. Related: pid_calc. Conventions: order [T, X1, X2]. |
| `run_dimension_sweep (line 348)` | dimension_to_sweep: str, dimension_values: list[int], target_dim: int, source1_dim: int, source2_dim: int, source1_gain: float, source2_gain: float, source1_noise_variance: float, source2_noise_variance: float, csv_path: str \| Path | list[dict[str, float \| int \| str]] | Run and checkpoint PID methods while changing one variable dimension. | Scope: task-specific. Related: write_rows_to_csv, build_population_covariance, run_pid_methods. |
| `plot_dimension_sweep_csv (line 441)` | csv_path: str \| Path, plot_path: str \| Path, speedup_plot_path: str \| Path | tuple[Path, Path] | Plot result curves and direct Eigen-PID speedups from the sweep CSV. | Scope: task-specific. |
| `main (line 616)` | No inputs | tuple[Path, Path, Path, Path] | Save constants, run the sweep, and save result and speedup plots. | Scope: entry point. Related: save_run_constants, run_dimension_sweep, plot_dimension_sweep_csv. |

### `Simulations/Theoretical_Examples/Covariance/cov_functions.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `make_random_true_cov (line 20)` | config: dict, rng: torch.Generator \| None=None | np.ndarray | Construct a generic positive-definite Gaussian covariance. | Scope: task-specific. |
| `rectangular_identity (line 85)` | d_a: int, d_b: int, dtype: torch.dtype=torch.float64, device: str \| torch.device='cpu' | torch.Tensor | Create a rectangular identity-like matrix of shape (d_a, d_b). | Scope: task-specific. Conventions: float64. |
| `make_direct_true_cov_from_config (line 109)` | config: dict, dtype: torch.dtype=torch.float64, eps: float=1e-10 | torch.Tensor | Create an interpretable covariance matrix for [X1, X2, T] directly from a merged config dictionary. | Scope: task-specific. Related: rectangular_identity. Conventions: order [X1, X2, T], float64. |
| `make_both_unique_true_cov_from_config (line 183)` | config: dict, rng: torch.Generator \| None=None | torch.Tensor | Create the paper's Gaussian both-unique covariance in [X1, X2, T]. | Scope: task-specific. Conventions: order [X1, X2, T], float64. |
| `sample_from_cov (line 223)` | config, true_cov: torch.Tensor, n_samples: int, rng: torch.Generator | torch.Tensor | Sample from a Gaussian distribution with the given covariance. | Scope: task-specific. |
| `change_covariance_order (line 252)` | cov: torch.Tensor, new_order: list[int], dims: list[int] | torch.Tensor | Permute the covariance matrix to change the order of variables. | Scope: task-specific. Related: create_cov_matrix. |


## Simulations/Theoretical_Examples/Covariance/results_covariance/Sweeps_comp2

### `Simulations/Theoretical_Examples/Covariance/results_covariance/Sweeps_comp2/run_sweeps_comp2.py`

File description: Run the second theoretical PID timing sweep in its own tmux session.

No functions, methods, or classes defined in this file.


## Simulations/Theoretical_Examples/Covariance

### `Simulations/Theoretical_Examples/Covariance/run_gaussian_pid_examples.py`

File description: Benchmark Gaussian PID methods repeatedly on theoretical covariances.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `one_dimensional_target_example (line 135)` | No inputs | ExampleConfig | Return the example whose target is one-dimensional. | Scope: task-specific. |
| `run_method (line 157)` | covariance: torch.Tensor, example: ExampleConfig, runtime_repeat: int, label: str, method: str | dict[str, object] | Time one PID method call on a theoretical covariance. | Scope: task-specific. Related: pid_calc. |
| `check_theoretical_example (line 230)` | example: ExampleConfig, rows: list[dict[str, object]] | None | Verify the promised theoretical behavior before saving results. | Scope: task-specific. |
| `summarize_results (line 268)` | rows: list[dict[str, object]], examples: list[ExampleConfig] | list[dict[str, object]] | Calculate PID and runtime statistics across timing repetitions. | Scope: task-specific. |
| `plot_results (line 309)` | rows: list[dict[str, object]], examples: list[ExampleConfig], path: str \| Path | Path | Plot theoretical PID values and runtime mean ± sample SD. | Scope: task-specific. |
| `save_hyperparameters (line 377)` | path: str \| Path, examples: list[ExampleConfig], outputs: dict[str, str] | Path | Save all non-path settings and output paths as YAML. | Scope: task-specific. |
| `main (line 404)` | results_csv_path: str \| Path=RESULTS_CSV, summary_csv_path: str \| Path=SUMMARY_CSV, plot_path: str \| Path=PLOT_PATH, yaml_path: str \| Path=YAML_PATH | tuple[Path, Path, Path, Path] | Benchmark every theoretical covariance over repeated method calls. | Scope: entry point. Related: write_rows_to_csv, summarize_results, plot_results, save_hyperparameters. |

### `Simulations/Theoretical_Examples/Covariance/sample_simulation.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `calculate_sample_whitened_wishart_mi_bits (line 67)` | sources: list[torch.Tensor], target: list[torch.Tensor] | dict[str, float] | Calculate raw and Wishart-corrected sample Gaussian MIs in bits. | Scope: task-specific. Related: create_cov_matrix, whiten_block, calcualte_mi, mi_wishart_bias. Conventions: whitening-sensitive. |
| `load_config (line 196)` | config_path: str \| Path=DEFAULT_CONFIG_PATH | dict | Load configuration from YAML file. | Scope: task-specific. |
| `csv_save (line 202)` | config: dict, experiment_name: str, method: str, theoretical_values: tuple, sampled_values: dict | Path | Save theoretical and sampled PID/MI component values to one method CSV. | Scope: task-specific. |
| `simulation (line 238)` | config: dict, methods: list, experiment_name: str \| None=None | dict | Run theoretical-covariance PID calculations and sampled trial summaries. | Scope: task-specific. Related: change_covariance_order, calculate_mi_raw, make_both_unique_true_cov_from_config, make_direct_true_cov_from_config. |

### `Simulations/Theoretical_Examples/Covariance/save_results.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `save_sample_simulation_results_table (line 15)` | results: dict, config: dict, save_path: str \| Path, decimals: int=4, title: str='PID Method Comparison', dpi: int=200 | Path | Save sample-simulation PID/MI summaries as a styled table image. | Scope: task-specific. Conventions: bias-correction aware. |


## Simulations/Theoretical_Examples/RVs_Story/Non-gaussian

### `Simulations/Theoretical_Examples/RVs_Story/Non-gaussian/t-dist_rvs.py`

File description: Student-t non-Gaussian RV examples.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `standardized_t (line 15)` | rng, df, size | Unannotated return value; inspect implementation before reuse | Sample a variance-one Student-t random variable. | Scope: task-specific. |
| `unq2_zero_t (line 31)` | rng, n, p, noise_std, df=5 | Unannotated return value; inspect implementation before reuse | Generate a Student-t version of the zero-source-2-unique example. | Scope: task-specific. Related: standardized_t. |
| `unq12_zero (line 55)` | rng, n, p, noise_std, df=5 | Unannotated return value; inspect implementation before reuse | Generate a Student-t version of the zero-both-unique example. | Scope: task-specific. Related: standardized_t. |


## Simulations/Theoretical_Examples/RVs_Story

### `Simulations/Theoretical_Examples/RVs_Story/flow_pid_grid_search.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `load_example_and_truth (line 67)` | example_name | Unannotated return value; inspect implementation before reuse | Return the registered sample generator and analytical truth function. | Scope: task-specific. |
| `grid_items (line 81)` | grid | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `standardize_train_val (line 87)` | train, val | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `make_folds (line 96)` | n, k_folds, seed | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `true_synergy (line 104)` | truth_func, x1, x2, t | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `write_csv (line 111)` | path, rows, fieldnames | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `run_grid_search (line 119)` | config, example_name, k_folds, grid, results_dir, device | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. Related: load_example_and_truth, make_folds, standardize_train_val, flow_pid_wrapper. |
| `parse_args (line 241)` | No inputs | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. |
| `main (line 256)` | No inputs | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: parse_args, run_grid_search. |


## Simulations/Theoretical_Examples/RVs_Story/regular_examples

### `Simulations/Theoretical_Examples/RVs_Story/regular_examples/All_above_zero.py`

File description: Regular examples with balanced unique information.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `all_above_zero_weighted (line 15)` | rng, n, p, noise_std, unique1_weight=5.0, unique2_weight=5.0, redundant_weight=1.0, shared_noise_weight=1.0 | Unannotated return value; inspect implementation before reuse | Gaussian example where all PID components should be above zero, but unique information is emphasized. | Scope: task-specific. |
| `con_all_above_zero_weighted (line 71)` | rng, n, p, noise_std, unique1_weight=5.0, unique2_weight=5.0, redundant_weight=1.0, shared_noise_weight=1.0 | Unannotated return value; inspect implementation before reuse | Concatenated Gaussian example where all PID components should be above zero, but unique information is emphasized. | Scope: task-specific. |

### `Simulations/Theoretical_Examples/RVs_Story/regular_examples/all_reg_examples.py`

File description: Run all regular RVs_Story examples across seeds.

No functions, methods, or classes defined in this file.

### `Simulations/Theoretical_Examples/RVs_Story/regular_examples/equal_unique.py`

File description: Regular examples with balanced unique information.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `equal_unique (line 15)` | rng, n, p, noise_std | Unannotated return value; inspect implementation before reuse | Generate a Gaussian example with equal unique information in both sources. | Scope: task-specific. |
| `equal_unique2 (line 39)` | rng, n, p, noise_std, snr=1 | Unannotated return value; inspect implementation before reuse | Generate a higher-dimensional equal-unique example from real features. | Scope: task-specific. |

### `Simulations/Theoretical_Examples/RVs_Story/regular_examples/no_mi.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `zero_MI (line 10)` | rng, n, p, noise_std | Unannotated return value; inspect implementation before reuse | Generate a Gaussian example with equal unique information in both sources. | Scope: task-specific. |


## Simulations/Theoretical_Examples/RVs_Story

### `Simulations/Theoretical_Examples/RVs_Story/story_batch_utils.py`

File description: Seed-loop and CSV helpers for RVs_Story example batches.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `_as_float (line 53)` | value | float | Convert scalar numeric values from tensors or Python numbers to float. | Scope: private helper. |
| `csv_path (line 65)` | results_dir: Path, example: str, method: str | Path | Build the per-example, per-method PID seed CSV path. | Scope: task-specific. |
| `commonality_csv_path (line 81)` | results_dir: Path, example: str=COMMONALITY_KEY | Path | Build the shared per-seed commonality CSV path. | Scope: task-specific persistence helper. |
| `csv_has_seed (line 95)` | path: Path, seed: int, example: str \| None=None | bool | Check whether a seed, optionally for one named example, is present in a CSV. | Scope: task-specific resume helper. |
| `seed_is_done (line 116)` | seed: int, example_names: list[str], results_dir: Path, methods=PID_METHODS, require_commonality: bool=False | bool | Check whether all expected PID rows and, when enabled, commonality rows exist for a seed. | Scope: task-specific. Related: csv_has_seed, csv_path, commonality_csv_path. |
| `loop_examples (line 148)` | config: dict, functions_to_run: list[Callable], example_names: list[str], main_func: Callable, save_image: bool=True | dict | Run each RV example, preserve its PID results, and optionally calculate commonality from the same generated RVs. | Scope: task-specific. Related: commonality_analysis, save_pid_comparison_table. Conventions: RV order [X1, X2, T]. |
| `save_seed_csvs (line 198)` | seed: int, all_results: dict, results_dir: Path, save_commonality_csv: bool=False | None | Upsert one seed of PID results and optional example-labelled commonality results into CSV checkpoints. | Scope: task-specific. Related: pid_comparison_table, csv_path, commonality_csv_path, _as_float. |
| `mean_results_from_csvs (line 272)` | results_dir: Path, example: str, seeds: list[int] | dict | Average saved True Values, Tilde, Delta, Analytical BROJA, and Flow seed CSVs back into a PID result dictionary. | Scope: task-specific. Related: csv_path, pid_method_display_name. |
| `mean_commonality_from_csvs (line 300)` | results_dir: Path, example: str, seeds: list[int] | dict | Average one example's commonality rows across selected seeds using canonical result keys. | Scope: task-specific. Related: commonality_csv_path. |
| `loop_examples_over_seeds (line 337)` | config: dict, functions_to_run: list[Callable], example_names: list[str], main_func: Callable, num_seeds: int \| None=None, seeds: list[int] \| None=None | dict | Run examples over seeds, checkpoint PID/commonality rows, and save averaged figures with concise titles and canonical PID display labels. | Scope: task-specific. Related: seed_is_done, loop_examples, save_seed_csvs, mean_results_from_csvs, mean_commonality_from_csvs, pid_method_display_name. |

### `Simulations/Theoretical_Examples/RVs_Story/story_math_utils.py`

File description: Small adapters around shared MI and bias helpers for RVs_Story truth rows.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `calculate_story_mi_values (line 19)` | sources: list[torch.Tensor], target: list[torch.Tensor] | tuple[dict, dict] | Calculate raw Gaussian MI values and legacy Wishart bias values. | Scope: task-specific. Related: create_cov_matrix, calculate_mi_raw, mi_wishart_bias. |

### `Simulations/Theoretical_Examples/RVs_Story/story_pid_utils.py`

File description: Shared PID execution helpers for the RVs_Story examples.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `pid_method_display_name (line 32)` | method: str | str | Convert PID dispatch identifiers to stable result-table and CSV labels, including `eigen`/`eigen_pid` as `Analytical BROJA`. | Scope: reusable RVs_Story display helper. Related: run_pid_story, loop_examples_over_seeds. |
| `truth_pid_suppression (line 52)` | sources: list[torch.Tensor], target: list[torch.Tensor], covariance=None | tuple[dict, dict] | Compute the Gaussian truth row for suppression-style examples. | Scope: task-specific. Related: calculate_story_mi_values. |
| `truth_pid_equal_unique (line 75)` | sources: list[torch.Tensor], target: list[torch.Tensor], covariance=None | tuple[dict, dict] | Compute the Gaussian truth row for equal-unique regular examples. | Scope: task-specific. Related: calculate_story_mi_values. |
| `run_pid_story (line 103)` | config: dict, function_to_run: Callable, truth_func: Callable \| None=None, methods: tuple[str, ...]=('tilde', 'delta', 'flow') | tuple[dict, list[np.ndarray]] | Run one RVs_Story generator through selected PID methods, apply canonical display labels, and return the generated RVs for downstream commonality analysis. | Scope: task-specific. Related: pid_calc, pid_method_display_name, loop_examples. Conventions: returned RV order [X1, X2, T], samples by features. |
| `load_story_config (line 158)` | config_path: str \| Path \| None=None | dict | Load the RVs_Story YAML configuration. | Scope: task-specific. |
| `save_single_example (line 172)` | config: dict, function_to_run: Callable, output_name: str, truth_func: Callable \| None=None | dict | Run one example, unpack the shared runner result, and save its PID comparison figure. | Scope: task-specific. Related: run_pid_story, save_pid_comparison_table. |


## Simulations/Theoretical_Examples/RVs_Story/suppresion_examples

### `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/all_supp_examples.py`

File description: Run all suppression RVs_Story examples across seeds.

No functions, methods, or classes defined in this file.

### `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/full_suppresion.py`

File description: Full suppression Gaussian RV example.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `full_suppresion (line 15)` | rng, n, p, noise_std | Unannotated return value; inspect implementation before reuse | Generate the full suppression example. | Scope: task-specific. |

### `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/unq12_zero.py`

File description: Suppression example where both sources have zero unique information.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `unq12_zero (line 15)` | rng, n, p, noise_std | Unannotated return value; inspect implementation before reuse | Generate an example with zero source-1 and source-2 unique information. | Scope: task-specific. |

### `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/unq2_zero.py`

File description: Suppression example where source 2 has zero unique information.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `unq2_zero (line 15)` | rng, n, p, noise_std | Unannotated return value; inspect implementation before reuse | Generate an example with zero source-2 unique information. | Scope: task-specific. |


## Simulations/evil_twin

### `Simulations/evil_twin/__init__.py`

File description: Sonic/Shadow evil-twin covariance examples.

No functions, methods, or classes defined in this file.

### `Simulations/evil_twin/covariance_example.py`

File description: Torch Sonic/Shadow evil-twin covariance example.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `empirical_covariance_matrix_torch (line 6)` | X1, X2, T, correction=1 | Unannotated return value; inspect implementation before reuse | Compute the empirical covariance matrix of (X1, X2, T). | Scope: task-specific. Conventions: order [X1, X2, T]. |
| `check_evil_twin_covariances_torch (line 24)` | data, atol=0.05, rtol=0.05, verbose=True | Unannotated return value; inspect implementation before reuse | Compare the empirical covariance matrices of Sonic and Shadow. | Scope: task-specific. Related: empirical_covariance_matrix_torch. |
| `evil_twin_example_torch (line 70)` | generator, n, p, device='cpu', dtype=torch.float64 | Unannotated return value; inspect implementation before reuse | Generate the Sonic and Shadow evil-twin Gaussian examples. | Scope: task-specific. Related: randn_scaled. Conventions: float64. |
| `evil_twin_example_torch.randn_scaled (line 88)` | var_total | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `run_covariance_comparison (line 126)` | n=1000, p=300, seed=0, device='cpu', dtype=torch.float64, atol=0.05, rtol=0.05, verbose=True | Unannotated return value; inspect implementation before reuse | Generate Sonic/Shadow samples and compare their empirical covariances. | Scope: task-specific. Related: evil_twin_example_torch, check_evil_twin_covariances_torch. Conventions: float64. |

### `Simulations/evil_twin/evil_twin_pid_batch_utils.py`

File description: Seed-loop and CSV helpers for evil-twin PID_calc sweeps.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `summary_csv_path (line 29)` | output_dir: Path, prefix: str='evil_twin_pid' | Path | Build the output CSV path for the mean summary table. | Scope: task-specific. |
| `method_csv_path (line 43)` | output_dir: Path, method: str, prefix: str='evil_twin_pid' | Path | Build the output CSV path for one PID method. | Scope: task-specific. |
| `write_rows_to_csv (line 58)` | path: Path, rows: list[dict], fieldnames: list[str] | Path | Write a complete CSV table, replacing any existing file. | Scope: task-specific. |
| `make_pid_config (line 77)` | n: int, p: int, device: str, flow_epochs: int, flow_verbose: bool | dict | Create the config dictionary expected by PID_calc wrappers. | Scope: task-specific. |
| `result_row (line 102)` | seed: int, twin: str, method: str, n: int, p: int, pid: dict, mi: dict | dict | Create a successful result row for one twin and PID method. | Scope: task-specific. |
| `error_row (line 135)` | seed: int, twin: str, method: str, n: int, p: int, error: Exception | dict | Create an error row for one failed twin and PID method. | Scope: task-specific. |
| `summary_fieldnames (line 164)` | twins: tuple[str, ...]=SUMMARY_TWINS | list[str] | Build ordered column names for the mean summary table. | Scope: task-specific. |
| `mean_summary_rows (line 180)` | seed_results: dict, methods: tuple[str, ...], n_samples: int, dimension: int, bias_correction: bool, twins: tuple[str, ...]=SUMMARY_TWINS | list[dict] | Calculate mean PID and MI values across seeds for each method. | Scope: task-specific. Conventions: bias-correction aware. |
| `save_summary_csv (line 237)` | output_dir: Path, prefix: str, rows: list[dict] | Path | Save the mean summary table to a CSV file. | Scope: task-specific. Related: write_rows_to_csv, summary_csv_path, summary_fieldnames. |
| `summary_image_path (line 251)` | output_dir: Path, prefix: str, twin: str | Path | Build the output image path for one twin's mean summary table. | Scope: task-specific. |
| `summary_rows_to_pid_results (line 266)` | rows: list[dict], twin: str | dict | Convert mean summary rows to the PID result shape used by RVs_Story tables. | Scope: task-specific. |
| `save_summary_table_images (line 295)` | rows: list[dict], output_dir: Path, prefix: str, config: dict, twins: tuple[str, ...]=SUMMARY_TWINS | list[Path] | Save RVs_Story-style PID comparison images for the mean summary. | Scope: task-specific. Related: summary_rows_to_pid_results, summary_image_path, save_pid_comparison_table. |
| `format_summary_value (line 340)` | value, decimals: int | str | Format one summary table cell for terminal output. | Scope: task-specific. |
| `format_summary_table (line 357)` | rows: list[dict], decimals: int=6 | str | Format summary rows as an aligned plain-text table. | Scope: task-specific. Related: summary_fieldnames, format_summary_value. |

### `Simulations/evil_twin/plot_summary_csv.py`

File description: Render an evil-twin summary CSV in the shared simulation table style.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `load_summary_rows (line 37)` | summary_csv: Path \| str | list[dict] | Load and validate rows from an evil-twin mean-summary CSV. | Scope: task-specific. Related: summary_fieldnames. |
| `plot_summary_csv (line 103)` | summary_csv: Path \| str, output_dir: Path \| str \| None=None, prefix: str \| None=None, n_samples: int \| None=None, dimension: int \| None=None, bias_correction: bool \| None=None | list[Path] | Create Sonic and Shadow plots from an existing evil-twin summary CSV. | Scope: task-specific. Related: load_summary_rows, save_summary_table_images. Conventions: bias-correction aware. |
| `main (line 174)` | config: dict \| None=None | list[Path] | Render one configured evil-twin summary CSV and print output paths. | Scope: entry point. Related: plot_summary_csv. Conventions: bias-correction aware. |

### `Simulations/evil_twin/run_from_config.py`

File description: Run the evil-twin PID experiment from an editable Python configuration.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `validate_config (line 51)` | config: dict | None | Validate the editable evil-twin experiment configuration. | Scope: task-specific. Conventions: bias-correction aware. |
| `uncorrected_delta_pid (line 119)` | config: dict, sources: list[torch.Tensor], target: list[torch.Tensor] | tuple[dict, dict] | Calculate raw Delta PID without applying Wishart bias correction. | Scope: task-specific. Related: create_cov_matrix. |
| `calculate_pid (line 167)` | config: dict, sources: list[torch.Tensor], target: list[torch.Tensor], generator: torch.Generator, method: str | tuple[dict, dict] | Run one configured PID method on one evil-twin sample set. | Scope: task-specific. Related: pid_calc, uncorrected_delta_pid. |
| `run_from_config (line 197)` | config: dict | dict | Run the configured evil-twin experiment and save method and summary CSVs. | Scope: task-specific. Related: validate_config, make_pid_config, mean_summary_rows, save_summary_csv. |
| `parse_args (line 299)` | No inputs | argparse.Namespace | Parse optional overrides for the consolidated evil-twin runner. | Scope: entry point. |
| `config_from_args (line 336)` | args: argparse.Namespace | dict | Apply explicitly supplied command-line values to ''CONFIG''. | Scope: task-specific. Conventions: bias-correction aware. |
| `main (line 366)` | No inputs | dict | Run the evil-twin experiment with CONFIG and command-line overrides. | Scope: entry point. Related: run_from_config, config_from_args, parse_args. |


## repository root

### `__init__.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.


## data

### `data/FBA-1.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.

### `data/OTC.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `load_OTC (line 8)` | subject_id: int, path_to_data: str | dict | Load OTC fMRI data for a given subject. (This function assumes data files are zarr files stored in the specified path.) | Scope: reusable/public. |
| `main (line 30)` | No inputs | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: load_OTC. |

### `data/V1.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `load_roi_data (line 20)` | args, roi_name='V1v' | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: get_specific_roi_fmri. |
| `roi_encoding_model (line 38)` | train_data_loader, val_data_loader, lh_roi_fmri_train, rh_roi_fmri_train, lh_roi_fmri_val, rh_roi_fmri_val, layer_name='features.2', model=None, features=None, batch_size=500, ncomponents=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: run_model. |
| `main (line 67)` | No inputs | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: train_save_or_load, grid_search_suppression_analysis. |


## encoding_model

### `encoding_model/__init__.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.

### `encoding_model/algoanut_data.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `argObj (line 26)` | class | argObj instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. |
| `argObj.__init__ (line 27)` | self, data_dir, parent_submission_dir, subj | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `load_data_algonauts (line 41)` | paths_dict, args=None, subj=1, plot_fmri=False | Unannotated return value; inspect implementation before reuse | Load fMRI data and image file lists for a given subject. | Scope: reusable/public. Related: plot_fmri. |

### `encoding_model/commonality.py`

File description: Shared commonality analysis utilities.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `_ensure_2d (line 13)` | features | Unannotated return value; inspect implementation before reuse | Raise value error if features are not 2D: (n_samples, n_features). If features are 1D, raise an error with instructions to reshape. | Scope: private helper. |
| `_score_only (line 22)` | score_result | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `commonality_analysis (line 28)` | features_X1, features_X2, target, method='standard', alphas=None, scale_by_target_variance=False | Unannotated return value; inspect implementation before reuse | Decompose predictive power into unique, common, and unexplained components. | Scope: reusable/public. Related: _ensure_2d, _score_only, compute_ridge_cv_r2. |

### `encoding_model/encoding_utils.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `ImageDataset (line 23)` | class | ImageDataset instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. |
| `ImageDataset.__init__ (line 24)` | self, imgs_paths, idxs, transform | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `ImageDataset.__len__ (line 31)` | self | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `ImageDataset.__getitem__ (line 34)` | self, idx | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `plot_fmri (line 44)` | path, args, hemi, title='' | No explicit return; likely None / side effects | Plot fMRI data on a brain surface and save the figure. | Scope: reusable/public. |
| `fmri_response_image (line 76)` | path, args, hemisphere, img_idx, train_img_dir, train_img_list, lh_fmri, rh_fmri | No explicit return; likely None / side effects | This function outputs the fmri response that matches the image shown. accoring to the NSD dataset structure. | Scope: reusable/public. |
| `split_dataset (line 128)` | train_img_list, test_img_list, rand_seed=5, train_p=90 | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `fmri_data_loader (line 153)` | lh_fmri, rh_fmri, train_img_list, test_img_list, train_img_dir, test_img_dir, batch_size=500, train_p=90 | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: split_dataset. |
| `map_correlation_to_rois (line 203)` | args, lh_correlation, rh_correlation, hemisphere | No explicit return; likely None / side effects | Map correlation values to ROIs. | Scope: reusable/public. |
| `roi_fmri_data (line 240)` | args, lh_fmri, rh_fmri | Unannotated return value; inspect implementation before reuse | Map fMRI data to ROIs. | Scope: reusable/public. |
| `get_specific_roi_fmri (line 294)` | args, lh_fmri, rh_fmri, roi_name | Unannotated return value; inspect implementation before reuse | Get fMRI data for a specific ROI. | Scope: reusable/public. Related: roi_fmri_data. |
| `visualize_encdoing_accuaracy (line 312)` | args, lh_correlation, rh_correlation, correlation_path, plot=True | Unannotated return value; inspect implementation before reuse | Visualize encoding accuracy with a bar graph and return ROI correlation values and ROI names for left and right hemispheres for a given subject. | Scope: reusable/public. Related: check_file_exists. |
| `save_corellation (line 396)` | roi_names, lh_correlation, rh_correlation, correlation_path, experiment_name | No explicit return; likely None / side effects | Save correlation values to .npy files. | Scope: reusable/public. Related: check_file_exists. |
| `save_model (line 425)` | folder_path, model_name, save_dict, reg_lh: Optional[ndarray]=None, reg_rh: Optional[ndarray]=None, features_val_pred_lh: Optional[List]=None, features_val_pred_rh: Optional[List]=None, features_train: Optional[ndarray]=None, features_val_trained: Optional[ndarray]=None, predict_array: Optional[ndarray]=None, roi_names: Optional[List]=None, lh_correlation: Optional[ndarray]=None, rh_correlation: Optional[ndarray]=None | Unannotated return value; inspect implementation before reuse | Save the trained encoding model. with its corellation values and roi names and figs' | Scope: reusable/public. Related: check_folder_exists, save_corellation. |

### `encoding_model/fmri_model.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `encoding_model (line 14)` | class | encoding_model instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. Related: fit, extract_features, train, fit_pca. |
| `encoding_model.__init__ (line 15)` | self, device, model: str='alexnet', model_layer: str='features.2', model_path: str='pytorch/vision:v0.10.0', features: Optional[np.ndarray]=None, n_features=None | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `encoding_model.fit_pca (line 40)` | self, dataloader, batch_size=100, ncomponents=100 | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `encoding_model.extract_features (line 51)` | self, dataloader, pca | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `encoding_model.train (line 61)` | self, train_data_loader, lh_fmri_train, rh_fmri_train, features_train: Optional[np.ndarray]=None, alphas: Optional[np.ndarray]=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. Related: fit. |
| `encoding_model.validate (line 82)` | self, reg_lh, reg_rh, lh_fmri_val, rh_fmri_val, features_val | Unannotated return value; inspect implementation before reuse | "This funciton validates the encoding model on the validation set and returns the correlation scores for each hemisphere. | Scope: method/nested helper. |
| `encoding_model.run_model (line 110)` | self, train_imgs_dataloader, val_imgs_dataloader, lh_fmri_train, rh_fmri_train, lh_fmri_val, rh_fmri_val, batch_size=100, ncomponents=None | Unannotated return value; inspect implementation before reuse | This function runs the entire encoding model pipeline: feature extraction, training, validation without testing. | Scope: method/nested helper. Related: extract_features, train, fit_pca, validate. |

### `encoding_model/grid_search.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `grid_search_ols (line 51)` | method, suppresions_strengths_list, snr_list, mixing_dimensions_list, signal | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: create_supression_model, commonality_analysis. |

### `encoding_model/pred_pipeline.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `pipeline (line 20)` | data_dir, parent_submission_dir, subj, args, layer_name='features.2', model=None, features=None, only_validate=False, train_p=80, data_fmri=None, data_imgs=None | Unannotated return value; inspect implementation before reuse | Main pipeline to run the encoding model on Algonauts data for a given subject. | Scope: reusable/public. Related: run_model, load_data_algonauts, split_dataset, fmri_data_loader. |
| `trained_model (line 99)` | layer_name, model, model_name, train_p | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: pipeline, save_model, map_correlation_to_rois, visualize_encdoing_accuaracy. |
| `just_validate (line 128)` | layer_name, model | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: pipeline, map_correlation_to_rois, save_model, visualize_encdoing_accuaracy. |

### `encoding_model/regression_metrics.py`

File description: Shared regression scoring helpers for encoding and toy examples.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `compute_ols_cv_r2 (line 6)` | X, y, return_model=False | Unannotated return value; inspect implementation before reuse | Compute cross-validated R2 using leave-one-out cross-validation. | Scope: reusable/public. Related: fit. |
| `compute_ridge_cv_r2 (line 32)` | X, y, alphas=None, return_model=False | Unannotated return value; inspect implementation before reuse | Compute cross-validated R2 using RidgeCV with efficient LOO cross-validation. | Scope: reusable/public. Related: fit. |
| `compute_r2 (line 64)` | X, y, return_model=False | Unannotated return value; inspect implementation before reuse | Compute in-sample R2 for OLS regression. | Scope: reusable/public. Related: fit. |
| `compute_lasso_cv_r2 (line 87)` | X, y | Unannotated return value; inspect implementation before reuse | Compute in-sample R2 after fitting multi-output LassoCV. | Scope: reusable/public. Related: fit. |

### `encoding_model/suppresion_model.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `real_model_func (line 52)` | model, layer_name, model_path, batch_size, ncomponents, train_data_loader | Unannotated return value; inspect implementation before reuse | Create and train the real encoding model using fMRI data and image features. | Scope: reusable/public. Related: fit_pca, extract_features, train. |
| `train_save_or_load (line 81)` | folder_path=None, model_name=None, path_to_load=None | Unannotated return value; inspect implementation before reuse | Load a trained encoding model from disk. | Scope: reusable/public. Related: real_model_func, save_model, check_file_exists. |
| `main (line 100)` | dict, suppression_strength=0.5, rng_seed=0 | No explicit return; likely None / side effects | Run the 2x3 factorial experiment design. | Scope: entry point. Related: run_all_methods. |
| `test_run (line 164)` | run_name, save_dir, features, fmri_dict, rng_seeds, suppression_method, suppression_strength=[0.5], n_samples=[1000], n_features=[100], snr=[1.0], mixing_dimension=[None] | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: meta_exists, create_encoder, create_predictions, run_all_methods. |

### `encoding_model/suppression_core.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `create_predictions (line 12)` | reg_lh, reg_rh, features | Unannotated return value; inspect implementation before reuse | Create fMRI predictions using trained regression models. | Scope: reusable/public. |
| `create_encoder (line 28)` | rng, features, target, n_features | Unannotated return value; inspect implementation before reuse | Create and train a linear regression encoder. mostly usable when the number of model features is larger than the number of samples. there for we randomly select a subset of features to use for training. | Scope: reusable/public. Related: fit. |
| `permutate_models (line 59)` | rng, features, suppression_strength | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `noise_component (line 80)` | rng, features, suppression_strength, permutation | Unannotated return value; inspect implementation before reuse | Create suppresion model using only noise component. | Scope: reusable/public. Related: permutate_models. |
| `create_supression_model (line 106)` | rng, signal, suppresion_method, features, suppression_strength=0.5, snr=1.0, mixing_dimension=None | Unannotated return value; inspect implementation before reuse | Create suppression model features X_M1 and X_M2 based on the given parameters. | Scope: reusable/public. Related: permutate_models. |
| `run_all_methods (line 146)` | rng_seed, suppresion_method, mixing_dimension, snr, suppression_strength, models_and_features_dict=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: create_supression_model, commonality_analysis. |
| `suppression_analysis_pipeline (line 164)` | features, reg_lh=None, reg_rh=None, hemisphere='both', suppression_strength=0.5, snr=1.0, mixing_dimension=None, suppresion_method='permutate', analysis_methods=['standard', 'ols_cv', 'ridge_cv'], rng_seed=None, alphas=None | Unannotated return value; inspect implementation before reuse | Complete pipeline that takes model features, creates predictions via regression, generates suppression models, and performs commonality analysis. | Scope: reusable/public. Related: create_supression_model, commonality_analysis. |
| `grid_search_suppression_analysis (line 306)` | features, reg_lh=None, reg_rh=None, suppression_strength_list=None, snr_list=None, mixing_dimension_list=None, rng_seed_list=None, hemisphere='both', suppresion_method='permutate', output_dir='./grid_search_results', grid_name='NoName', verbose=True | Unannotated return value; inspect implementation before reuse | Perform a grid search over suppression analysis parameters using ridge regression. | Scope: reusable/public. Related: suppression_analysis_pipeline. |


## library_wrappers

### `library_wrappers/Delta_PID.py`

File description: Wrapper for the gpid Gaussian delta PID definition.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `parse_args (line 38)` | No inputs | argparse.Namespace | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. |
| `simple_example_args (line 54)` | No inputs | argparse.Namespace | Small debug example: source1 and source2 are noisy copies of one target. | Scope: reusable/public. |
| `main (line 65)` | No inputs | int | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: simple_example_args, parse_args, covariance_example_context, run_covariance_pid_wrapper. |

### `library_wrappers/Flow_PID.py`

File description: Wrapper for warrenzha/flow-pid's normalizing-flow PID estimator.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `parse_args (line 36)` | No inputs | argparse.Namespace | Parse CLI arguments for Flow-PID on raw sample matrices. | Scope: entry point. |
| `simple_example_args (line 77)` | No inputs | argparse.Namespace | Small debug example: train Flow-PID on samples from the shared Gaussian case. | Scope: reusable/public. |
| `load_flow_pid (line 99)` | No inputs | Unannotated return value; inspect implementation before reuse | Load and return flow-pid's original ''flow_pid'' function. | Scope: reusable/public. Related: load_module. |
| `read_samples (line 131)` | path: Path, expected_columns: int \| None=None | np.ndarray | Read a two-dimensional sample CSV with rows as observations. | Scope: reusable/public. |
| `split_combined_samples (line 145)` | samples: np.ndarray, sizes: tuple[int, int, int] | tuple[np.ndarray, np.ndarray, np.ndarray] | Split [source1, source2, target] samples into flow-pid's (target, source1, source2). | Scope: reusable/public. |
| `load_input_arrays (line 157)` | args: argparse.Namespace | tuple[np.ndarray, np.ndarray, np.ndarray] | Load target/M, source1/X, and source2/Y sample arrays. | Scope: reusable/public. Related: read_samples, simple_gaussian_samples, split_combined_samples. |
| `validate_training_args (line 183)` | args: argparse.Namespace | None | Validate Flow-PID training hyperparameters. | Scope: reusable/public. |
| `main (line 195)` | No inputs | int | Load raw samples, train flow-pid's estimator, and save PID components. | Scope: entry point. Related: validate_training_args, load_input_arrays, load_flow_pid, pid_result_row. |

### `library_wrappers/IG_R.py`

File description: Small Python wrapper for JWKay/PID/IGFuns.R.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `parse_correlation (line 149)` | value: str | float | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `parse_args (line 159)` | No inputs | argparse.Namespace | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. |
| `simple_example_args (line 182)` | No inputs | argparse.Namespace | Small debug example: run IG on the shared 1D Gaussian covariance. | Scope: reusable/public. |
| `apply_example_defaults (line 203)` | args: argparse.Namespace | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: set_evil_twin_inputs. |
| `set_evil_twin_inputs (line 224)` | args: argparse.Namespace | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `set_simple_gaussian_inputs (line 237)` | args: argparse.Namespace, matrix_csv: Path | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: write_simple_gaussian_covariance. |
| `ig_source_candidates (line 250)` | pid_repo: Path \| None | list[Path] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `find_ig_source (line 254)` | args: argparse.Namespace | Path | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: ig_source_candidates. |
| `require_shape (line 270)` | path: Path, expected: tuple[int, int], label: str | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: csv_shape. |
| `validate_inputs (line 278)` | args: argparse.Namespace | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: require_shape. |
| `run_r (line 314)` | args: argparse.Namespace | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: run, write_scalar_csv. |
| `write_scalar_csv (line 376)` | path: Path, value: float | Path | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `add_standard_table (line 381)` | result: dict | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `write_result (line 401)` | result: dict, output: Path \| None | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `print_result (line 411)` | result: dict | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `main (line 418)` | No inputs | int | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: apply_example_defaults, find_ig_source, validate_inputs, add_standard_table. |

### `library_wrappers/Idep_R.py`

File description: Small Python wrapper for JWKay/PID/IdepGauss.R.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `parse_args (line 153)` | No inputs | argparse.Namespace | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. |
| `csv_shape (line 171)` | path: Path | tuple[int, int] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `validate_matrix_args (line 190)` | args: argparse.Namespace | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: csv_shape. |
| `absolute_path (line 203)` | value: str | Path | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `main (line 210)` | No inputs | int | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: simple_example_args, parse_args, validate_matrix_args, absolute_path. |

### `library_wrappers/Thin_PID.py`

File description: Wrapper for warrenzha/flow-pid's exact Gaussian Thin-PID definition.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `parse_args (line 31)` | No inputs | argparse.Namespace | Parse CLI arguments for running Thin-PID on a covariance/correlation CSV. | Scope: entry point. |
| `simple_example_args (line 56)` | No inputs | argparse.Namespace | Small debug example: source1 and source2 are noisy copies of one target. | Scope: reusable/public. |
| `load_exact_gauss_thin_pid (line 69)` | No inputs | Unannotated return value; inspect implementation before reuse | Load and return flow-pid's original ''exact_gauss_thin_pid'' function. | Scope: reusable/public. Related: load_module. Conventions: whitening-sensitive. |
| `main (line 99)` | No inputs | int | Load input, run flow-pid Thin-PID, and save the result CSV. | Scope: entry point. Related: simple_example_args, parse_args, covariance_example_context, run_covariance_pid_wrapper. |

### `library_wrappers/Tilde_PID.py`

File description: Wrapper for the gpid Gaussian tilde PID definition.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `parse_args (line 39)` | No inputs | argparse.Namespace | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. |
| `simple_example_args (line 56)` | No inputs | argparse.Namespace | Small debug example: source1 and source2 are noisy copies of one target. | Scope: reusable/public. |
| `main (line 68)` | No inputs | int | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: simple_example_args, parse_args, covariance_example_context, run_covariance_pid_wrapper. |

### `library_wrappers/check_evil_twin_all.py`

File description: Run the IG and Idep evil-twin checks and combine their PID tables.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `parse_args (line 39)` | No inputs | argparse.Namespace | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. |
| `run (line 49)` | command: list[str], label: str, verbose: bool | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `require_file (line 70)` | path: Path, label: str | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `load_ig_rows (line 75)` | path: Path | list[dict[str, str]] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: normalize_rows. |
| `load_idep_rows (line 80)` | path: Path | list[dict[str, str]] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: normalize_rows. |
| `load_single_row_csv (line 107)` | path: Path | list[dict[str, str]] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: normalize_rows. |
| `normalize_rows (line 115)` | rows: list[dict] | list[dict[str, str]] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `write_csv (line 130)` | rows: list[dict[str, str]], path: Path | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `write_svg (line 138)` | rows: list[dict[str, str]], path: Path | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: esc. |
| `write_svg.esc (line 161)` | value: object | str | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `print_table (line 205)` | rows: list[dict[str, str]] | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `main (line 220)` | No inputs | int | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: parse_args, require_file, run, load_ig_rows. |

### `library_wrappers/compare_gpid_canonical.py`

File description: Compare GPID canonical examples by direct GPID calls and PID_calc wrappers.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `canonical_examples (line 38)` | No inputs | list[dict[str, object]] | Build the canonical covariance examples from external/gpid/scripts. | Scope: reusable/public. Related: add. |
| `canonical_examples.add (line 51)` | desc: str, case_id: int, cov: np.ndarray, sigma: float=np.nan, rho: float=np.nan | None | Append one canonical example to the local case list. | Scope: method/nested helper. |
| `unpack_gpid (line 92)` | values: tuple[float, ...] | dict[str, float] | Convert one GPID return tuple into named components. | Scope: reusable/public. |
| `unpack_wrapper (line 106)` | pid: dict[str, float], mi: dict[str, float] | dict[str, float] | Convert one PID_calc wrapper output into named components. | Scope: reusable/public. |
| `pid_calc (line 120)` | method: str, cov: np.ndarray, dm: int, dx: int, dy: int | dict[str, float] | Run one canonical covariance through the matching PID_calc wrapper. | Scope: reusable/public. Related: unpack_wrapper. Conventions: bias-correction aware. |
| `compare (line 140)` | No inputs | list[dict[str, object]] | Compare direct GPID calls with PID_calc wrappers for all canonical examples. | Scope: reusable/public. Related: canonical_examples, unpack_gpid, pid_calc. |
| `print_summary (line 174)` | rows: list[dict[str, object]] | None | Print the maximum absolute difference per method and component. | Scope: reusable/public. |
| `main (line 193)` | No inputs | int | Run the canonical GPID comparison command. | Scope: entry point. Related: parse_args, print_summary, compare. |

### `library_wrappers/missmda_ncp.py`

File description: Minimal in-process Python wrapper for R's ''missMDA::estim_ncpPCA''.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `estimate_ncp_pca (line 22)` | data, ncp_min=0, ncp_max=5, method='Regularized', scale=True, method_cv='gcv', nbsim=100, p_na=0.05, threshold=0.0001, seed=None, rscript=RSCRIPT, verbose=False | Unannotated return value; inspect implementation before reuse | Call ''missMDA::estim_ncpPCA'' on a Python sample table. | Scope: reusable/public. |

### `library_wrappers/r_idep_client.py`

File description: Programmatic Python client for JWKay/PID/IdepGauss.R.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `RIdePResult (line 93)` | class | RIdePResult instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. |
| `_to_2d_float_rows (line 100)` | matrix: Any | list[list[float]] | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `_write_matrix_csv (line 117)` | path: Path, rows: Sequence[Sequence[float]] | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `_read_result_csv (line 123)` | path: Path, stdout: str, stderr: str, *, bits_to_nats: bool | RIdePResult | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `run_idep_from_covariance (line 148)` | sigma: Any, sizes: Sequence[int], *, rscript: str \| Path \| None=None, idep_url: str=DEFAULT_IDEP_URL, local_idep: str \| Path \| None='IdepGauss.R', bits_to_nats: bool=True, keep_temp: bool=False | RIdePResult | Run R idepGM(sizes, sigma) and return named Idep/MMI atoms. | Scope: reusable/public. Related: _to_2d_float_rows, _resolve_local_idep, _write_matrix_csv, run. |
| `run_idep_for_cases (line 220)` | case_covariances: Mapping[str, Any], sizes: Sequence[int], **kwargs: Any | dict[str, RIdePResult] | Run R Idep/MMI for several named covariance matrices. | Scope: reusable/public. Related: run_idep_from_covariance. |
| `atoms_as_ordered_values (line 232)` | values: Mapping[str, float] | list[float] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `_resolve_local_idep (line 236)` | local_idep: str \| Path \| None | str | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |

### `library_wrappers/wrapper_utils.py`

File description: Shared utilities for the Python PID covariance wrappers.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `add_gpid_src_to_path (line 55)` | repo_root: Path \| None=None | Path | Put external/gpid/src on sys.path for script-style wrapper imports. | Scope: reusable/public. |
| `parse_sizes (line 66)` | value: str | tuple[int, int, int] | Parse source1,source2,target dimensions and reject invalid values. | Scope: reusable/public. |
| `read_matrix (line 77)` | path: Path | np.ndarray | Read a square covariance or correlation matrix from a CSV file. | Scope: reusable/public. |
| `validate_covariance (line 87)` | matrix: np.ndarray, expected_shape: tuple[int, int] | None | Validate shape and basic covariance/correlation symmetry. | Scope: reusable/public. |
| `source_source_target_to_target_source_source (line 95)` | matrix: np.ndarray, sizes: tuple[int, int, int] | np.ndarray | Reorder a [source1, source2, target] matrix to [target, source1, source2]. | Scope: reusable/public. Related: validate_covariance. |
| `write_simple_gaussian_covariance (line 104)` | path: Path | None | Write the shared simple [source1, source2, target] Gaussian covariance example. | Scope: reusable/public. |
| `simple_gaussian_samples (line 110)` | num_samples: int, seed: int | tuple[np.ndarray, np.ndarray, np.ndarray] | Generate raw samples from the shared simple Gaussian example. | Scope: reusable/public. |
| `covariance_example_context (line 130)` | args: argparse.Namespace | Unannotated return value; inspect implementation before reuse | Temporarily attach the shared simple covariance example to wrapper args. | Scope: reusable/public. Related: write_simple_gaussian_covariance. |
| `pid_result_row (line 145)` | values: tuple[float, ...], case: str, pid_definition: str, *, include_union_objective: bool=False | dict[str, object] | Convert a Gaussian PID tuple into the local one-row CSV schema. | Scope: reusable/public. |
| `write_pid_row (line 172)` | row: dict[str, object], path: Path, columns: list[str] | None | Write one standardized PID result row to a CSV file. | Scope: reusable/public. |
| `print_pid_result (line 181)` | row: dict[str, object], columns: list[str] | None | Print one PID result in a compact debug table. | Scope: reusable/public. |
| `load_module (line 213)` | module_name: str, path: Path | types.ModuleType | Load one source file as a module without importing package __init__ files. | Scope: reusable/public. |
| `run_covariance_pid_wrapper (line 224)` | args: argparse.Namespace, solver: Callable[..., tuple[float, ...]] \| None=None, *, pid_definition: str, columns: list[str], solver_loader: Callable[[], Callable[..., tuple[float, ...]]] \| None=None, solver_kwargs: dict[str, object] \| None=None, include_union_objective: bool=False, verbose_history: bool=False, read_message: bool=False, call_message: str \| None=None, written_message: str \| None=None | int | Run the common covariance-wrapper flow and write a one-row PID CSV. | Scope: reusable/public. Related: read_matrix, source_source_target_to_target_source_source, pid_result_row, print_pid_result. |


## repository root

### `my_utils.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `check_file_exists (line 24)` | file_path | Unannotated return value; inspect implementation before reuse | Check if a file exists at the given path. if it exists change it's name by adding a number at the end. | Scope: reusable/public. |
| `check_folder_exists (line 39)` | folder_path | Unannotated return value; inspect implementation before reuse | Check if a folder exists at the given path. if it doesn't exist, create it. | Scope: reusable/public. |
| `create_permuation (line 54)` | list_to_permute | Unannotated return value; inspect implementation before reuse | This function take a range of indices and return a permuted version of it. | Scope: reusable/public. |
| `standardize_np (line 73)` | X, eps: float=1e-12 | Unannotated return value; inspect implementation before reuse | Column-standardize a NumPy-compatible array. | Scope: reusable/public. |
| `Tee (line 82)` | class | Tee instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. Related: write, flush. |
| `Tee.__init__ (line 83)` | self, *files | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `Tee.write (line 86)` | self, data | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. Related: flush. |
| `Tee.flush (line 91)` | self | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `check_equal_type_invariance (line 95)` | a, b | bool | Check if two inputs are equal in value and type invariance. | Scope: reusable/public. |
| `meta_exists (line 115)` | meta_data: dict, csv_path | bool | Check whether a row with identical meta_data already exists in a CSV file. it is invariant to type differences (e.g., int vs float vs str). | Scope: reusable/public. Related: check_equal_type_invariance. |
| `_to_float_or_none (line 151)` | value | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `extract_all_components (line 159)` | ca_results: dict, pid_results: dict, mi_results: dict, global_results: dict=None, betas_dict: dict=None | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: _to_float_or_none. |
| `summarize_seed_results (line 191)` | results: list[dict] | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `print_seed_summary (line 206)` | summary: dict, n_seeds: int, seed_start: int | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `seed_summary_to_table (line 214)` | csv_path: Path \| str, decimals: int=5, save_path: Path \| str \| None=None | pd.DataFrame | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `save_csv_column_means (line 239)` | csv_path: Path \| str, output_csv_path: Path \| str, decimals: int=6 | pd.DataFrame | Compute the mean of all numeric CSV columns and save to a new CSV. | Scope: reusable/public. |
| `load_csv_and_add_data (line 274)` | csv_path: Path \| str, data: dict, mode: Literal['append_row', 'update_first_row', 'add_columns']='append_row', save_path: Path \| str \| None=None, detect_seed_metadata: bool=True | pd.DataFrame | Load a CSV, add data to it, and save it back. | Scope: reusable/public. |
| `save_seed_summary_table_image (line 342)` | csv_path: Path \| str, image_path: Path \| str, decimals: int=5, dpi: int=300 | Path \| None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: seed_summary_to_table. |
| `_normalize_config_value (line 376)` | value | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `get_experiment_name (line 388)` | config: dict | str | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: _normalize_config_value. |
| `_parse_csv_numeric (line 402)` | value: str | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `get_seed_runs_csv_path (line 414)` | config: dict | Path | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: get_experiment_name. |
| `get_seed_summary_csv_path (line 422)` | config: dict | Path | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: get_experiment_name. |
| `load_seed_run_checkpoint (line 430)` | config: dict | tuple[Path, list[dict], list[str]] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: get_seed_runs_csv_path, _parse_csv_numeric. |
| `_ensure_seed_runs_header (line 475)` | file_path: Path, config: dict, metric_names: list[str] | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. Related: _normalize_config_value. |
| `append_seed_run_checkpoint (line 493)` | config: dict, row: dict, metric_names: list[str] | Path | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: get_seed_runs_csv_path, _ensure_seed_runs_header. |
| `save_seed_summary_csv (line 505)` | summary: dict, config: dict | Path | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: get_seed_summary_csv_path, _normalize_config_value. |
| `run_multi_seed_experiment (line 544)` | config: dict, per_seed_runner: Callable[[int, dict], dict] | tuple[dict, list[dict]] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: load_seed_run_checkpoint, add, append_seed_run_checkpoint, summarize_seed_results. |
| `run_configured_multiseed (line 595)` | config: dict, per_seed_runner: Callable[[int, dict], dict] | tuple[dict, list[dict], Path, Path] | Run a configured multi-seed experiment and handle the standard reporting. | Scope: reusable/public. Related: run_multi_seed_experiment, print_seed_summary, get_seed_runs_csv_path, save_seed_summary_csv. |
| `create_distribution_plot (line 619)` | data: list[float], title: str, xlabel: str, ylabel: str, save_path: Path, bins: int=30, kde: bool=True | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `create_distribution_plot_with_colors (line 652)` | data: list[float], title: str, xlabel: str, ylabel: str, save_path: Path, bins: int=30, kde: bool=True, bar_color: str='#4C72B0', kde_color: str='#DD8452', bar_alpha: float=0.65, xlim: tuple[float, float] \| None=None | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `_load_results_dataframe (line 703)` | path: Path | pd.DataFrame | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `load_hist_kde_and_change_colors (line 729)` | csv_path: Path \| str, column: str, output_path: Path \| str, bins: int=30, bar_color: str='#4C72B0', kde_color: str='#DD8452', bar_alpha: float=0.65 | Path \| None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: _load_results_dataframe, create_distribution_plot_with_colors. |
| `create_test_histograms_with_kde (line 766)` | csv_path: Path \| str, output_dir: Path \| str, columns: list[str] \| None=None, bins: int=30, bar_color: str='#4C72B0', kde_color: str='#DD8452', bar_alpha: float=0.6, shared_x_axis: bool=True, shared_x_axis_groups: list[list[str]]=[['CA_R²_X1', 'CA_R²_X2', 'CA_R²_X12'], ['CA_unique_X1', 'CA_unique_X2', 'CA_common'], ['PID_red'], ['PID_unq1', 'PID_unq2'], ['PID_syn'], ['I(M1;T)', 'I(M2;T)', '"I(M1,M2;T)"']] | list[Path] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: _load_results_dataframe, create_distribution_plot_with_colors, _compute_xlim, add. |
| `create_test_histograms_with_kde._compute_xlim (line 804)` | arrays: list[np.ndarray] | tuple[float, float] \| None | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `get_config (line 866)` | config_path: Path \| str | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `inspect_function (line 874)` | function: Callable[..., Any], input_name: str | bool | Inspect a function's signature to check if it accepts a specific input name. | Scope: reusable/public. |


## pipeline/analysis

### `pipeline/analysis/anlysis_utils.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `_prepare_source_for_pid (line 14)` | source: np.ndarray, train_target: np.ndarray, shared_mask: np.ndarray, ridge: bool | np.ndarray | Prepare one model source for PID on the held-out shared images. | Scope: private helper. Related: find_alpha_per_pc, pca_source. |
| `to_deepdive_model_name (line 57)` | model_name: str | str | Convert a stored model alias to the canonical name expected by DeepDive. | Scope: task-specific. |


## pipeline/analysis/pca_analysis/all_models_pairwise

### `pipeline/analysis/pca_analysis/all_models_pairwise/pair_wise_comp.py`

File description: Run deterministic OTC PID comparisons across unordered model pairs.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `target_pca (line 85)` | target_context: np.ndarray, n_components: int, random_state: int | np.ndarray | PCA target, train target on unique images and return projected target for shared images. | Scope: task-specific. Conventions: float64. |
| `pca_model (line 107)` | features: np.ndarray, shared_mask: np.ndarray, n_components: int, random_state: int | np.ndarray | Fit source PCA on unique images and project shared images. | Scope: task-specific. Related: fit. |
| `extract_model_projection (line 144)` | model_name: str, target_context: dict[str, Any], choose_layer_kwargs: dict[str, Any], feature_extraction_kwargs: dict[str, Any], n_components: int, random_state: int | tuple[np.ndarray, int] | Extract one selected model layer and return its memory-safe PCA projection. | Scope: task-specific. Related: overall_best_layer, pca_model, batching. Conventions: float64. |
| `run_pairwise_pid_pipeline (line 253)` | model_1_names: list[str], model_2_names: list[str], otc_config: dict[str, Any], csv_path: str \| Path | Path | Run OTC PID once per unordered model pair and checkpoint results to CSV. | Scope: task-specific. Related: resolve_pipeline_function, target_pca, add, extract_model_projection. |

### `pipeline/analysis/pca_analysis/all_models_pairwise/ridge_pair_wise_comp.py`

File description: Run resumable prediction-level ridge PID across unordered model pairs.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `run_pairwise_pid_pipeline (line 24)` | model_1_names: list[str], model_2_names: list[str], otc_config: dict[str, Any], csv_path: str \| Path | Path | Run prediction-level PID once per unfinished unordered model pair. | Scope: task-specific. Related: _load_or_create_checkpoint, _unfinished_unordered_pairs, _required_models, resolve_pipeline_function. |
| `main (line 199)` | No inputs | None | Run the YAML-configured ridge analysis and plot its exact written CSV. | Scope: entry point. Related: run_pairwise_pid_pipeline, plot_pairwise_pid_matrices, _resolve_project_path. |

### `pipeline/analysis/pca_analysis/all_models_pairwise/ridge_pairwise_utils.py`

File description: Task-specific artifact, checkpoint, and prefetch helpers for ridge pairwise PID.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `RidgeModelArtifacts (line 54)` | class | RidgeModelArtifacts instance | Store one model's validated layer and ridge-alpha artifact. | Scope: class. |
| `safe_model_name (line 72)` | model_name: str | str | Replace path separators in a model identifier for artifact filenames. | Scope: task-specific. |
| `_resolve_project_path (line 85)` | path_value: str \| Path | Path | Resolve a configured path relative to the repository when necessary. | Scope: private helper. |
| `_load_or_create_checkpoint (line 101)` | output_path: Path | pd.DataFrame | Load a compatible pairwise CSV or create an empty checkpoint. | Scope: private helper. |
| `_unfinished_unordered_pairs (line 132)` | source_1_names: list[str], source_2_names: list[str], existing_results: pd.DataFrame | list[tuple[str, str]] | Choose the first requested orientation of every unfinished pair. | Scope: private helper. Related: add. |
| `_required_models (line 168)` | pairs: list[tuple[str, str]] | list[str] | Return each model needed by pending pairs once in first-use order. | Scope: private helper. Related: add. |
| `_model_artifact_path (line 188)` | artifact_config: Mapping[str, Any], model_name: str, artifact_label: str | Path | Resolve and require one model-specific configured artifact path. | Scope: private helper. Related: _resolve_project_path, safe_model_name. |
| `_validate_model_artifacts (line 238)` | model_names: list[str], config: Mapping[str, Any], expected_target_dim: int | dict[str, RidgeModelArtifacts] | Resolve and validate every required model artifact before extraction. | Scope: private helper. Related: _resolve_project_path, overall_best_layer, _model_artifact_path, load_ridge_alphas. |
| `_load_model_context (line 313)` | model_name: str | tuple[dict[str, Any], float] | Load one DeepDive model context and measure its loading duration. | Scope: private helper. Related: to_deepdive_model_name. |
| `_release_model_context (line 341)` | model_context: dict[str, Any] \| None | None | Clear one model context and release unused CPU and CUDA memory. | Scope: private helper. |
| `_prepare_ridge_prediction (line 358)` | model_name: str, artifacts: RidgeModelArtifacts, target_context: dict[str, Any], train_target: np.ndarray, shared_mask: np.ndarray, feature_extraction: Callable[..., Any], feature_extraction_kwargs: Mapping[str, Any], *, seed: int | tuple[np.ndarray, int] | Create one model's held-out ridge prediction and release intermediates. | Scope: private helper. Related: _load_model_context, feature_extraction, _release_model_context, ridge_predict_shared. |
| `_iter_ridge_prediction_pairs (line 435)` | pairs: list[tuple[str, str]], artifacts_by_model: Mapping[str, RidgeModelArtifacts], target_context: dict[str, Any], train_target: np.ndarray, shared_mask: np.ndarray, feature_extraction: Callable[..., Any], feature_extraction_kwargs: Mapping[str, Any], *, seed: int, prefetch_ridge_predictions: bool | Iterator[tuple[str, str, np.ndarray, int, np.ndarray, int]] | Yield PID-ready pairs while preprocessing the next model in one worker. | Scope: private helper. Related: _prepare_ridge_prediction. |


## pipeline/analysis/pca_analysis/function_as_pc

### `pipeline/analysis/pca_analysis/function_as_pc/pc_function.py`

File description: Calculate PID and mutual information as target PCs are accumulated.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `_prepare_source_for_pid (line 33)` | source: np.ndarray, train_target: np.ndarray, shared_mask: np.ndarray, ridge: bool | np.ndarray | Prepare one model source for PID on the held-out shared images. | Scope: private helper. Related: find_alpha_per_pc, pca_source. |
| `_save_pair_results (line 60)` | pair_results: dict[int, dict[str, Any]], model_1: str, model_2: str, results_dir: str \| Path | Path | Save the PC-dependent PID and MI results for one model pair. | Scope: private helper. |
| `pc_function_analysis (line 88)` | config: dict[str, Any], functions: PIDPipelineFunctions, model1_name: list[str], model2_name: list[str], pc_path: str \| Path, hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, results_dir: str \| Path \| None=None, plot_dir: str \| Path \| None=None | dict[str, dict[str, dict[int, dict[str, Any]]]] | Calculate PID and MI for each model pair and cumulative target-PC count. | Scope: task-specific. Related: prepare_target, prepare_ridge_target, feature_extraction, _prepare_source_for_pid. |
| `main (line 232)` | No inputs | None | Load the PC-function YAML, run all model pairs, and save their plots. | Scope: entry point. Related: pipeline_functions_from_config, pc_function_analysis. |

### `pipeline/analysis/pca_analysis/function_as_pc/plot_pc_results.py`

File description: Plot PID and mutual information as target PCs are accumulated.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `plot_pc_results_from_pickle (line 26)` | pkl_path: str \| Path, model_name: str, output_dir: str \| Path, name: str \| None=None | tuple[Path, Path] | Load one model pair's PC results and save its plots. | Scope: task-specific. Related: plot_pid_mi_as_function_of_pcs. |
| `plot_pid_mi_as_function_of_pcs (line 73)` | pair_results: dict[int, dict[str, Any]], model_1_name: str, model_2_name: str, output_dir: str \| Path, name: str \| None=None | tuple[Path, Path] | Plot absolute and trivariate-MI-normalized results for one model pair. | Scope: task-specific. |


## pipeline/analysis/pca_analysis/permuation_analysis

### `pipeline/analysis/pca_analysis/permuation_analysis/permuation_analysis.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `permuatation_analysis (line 22)` | pipeline_config: dict, permuation_config: dict | dict[str, Any] | Perform permutation analysis for the given pipeline and PCA configuration. | Scope: task-specific. Related: run_otc_experiment. |


## pipeline/analysis/pca_analysis/unique_search_outputs

### `pipeline/analysis/pca_analysis/unique_search_outputs/otc_unique_search_pca.py`

File description: Run PCA unique-information subset search on full OTC data and two sources.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `run_otc_unique_search_pca (line 27)` | config: dict[str, Any] \| str \| Path, max_source_components: int, *, search_source: str='X1', search_kwargs: dict[str, Any] \| None=None, all_csv_path: str \| Path \| None=None, best_csv_path: str \| Path \| None=None, pid_callable: Callable[..., Any] \| None=None | dict[str, Any] | Load OTC/source arrays, fix target/other-source PCs, and search one source. | Scope: task-specific. Related: _load_config, _load_otc_arrays, _pca_search_inputs, run_pid_pc_subset_search. |
| `run_otc_unique_search_from_yaml (line 77)` | config_path: str \| Path=DEFAULT_SEARCH_CONFIG | dict[str, Any] | Run OTC unique search from one YAML analysis config file. | Scope: task-specific. Related: _load_config, run_otc_unique_search_pca, _resolve_path. |
| `_load_config (line 98)` | config: dict[str, Any] \| str \| Path | dict[str, Any] | Load an OTC config dictionary or YAML file. | Scope: private helper. |
| `_resolve_path (line 113)` | value: str \| Path \| None, base_dir: Path | Path \| None | Resolve optional YAML paths relative to the YAML file directory. | Scope: private helper. |
| `_load_otc_arrays (line 126)` | config: dict[str, Any] | dict[str, Any] | Run PIDPipeline only through target/source/layer/feature extraction. | Scope: private helper. Related: validate_pipeline_config_sections, run_configured_pid_pipeline. |
| `_skip_pid (line 148)` | target: Any, source_1: Any, source_2: Any, **pid_kwargs: Any | None | Satisfy PIDPipeline while loading arrays without running PID. | Scope: private helper. |
| `_pca_search_inputs (line 159)` | context: dict[str, Any], config: dict[str, Any], search_source: str, max_source_components: int | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | Project target, searched source, and fixed source for subset search. | Scope: private helper. Related: _project_or_keep. |
| `_project_or_keep (line 191)` | features: Any, n_components: int \| None, name: str | np.ndarray | Apply PCA when requested, capping components to the valid matrix size. | Scope: private helper. |
| `main (line 216)` | config_path: str \| Path \| None=None | None | Run OTC PCA unique search from a YAML config file. | Scope: entry point. Related: run_otc_unique_search_from_yaml. |

### `pipeline/analysis/pca_analysis/unique_search_outputs/pca_as_function.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `pca_as_function (line 29)` | pipeline_config: str \| Path, pca_config: str \| Path | dict[str, Any] | Run the full-OTC PID experiment from the YAML config beside this file. | Scope: task-specific. Related: run_otc_experiment. |
| `plot_ (line 62)` | results_dict: dict[str, Any], pca_config: str, pipeline_config: str | None | Plot the results of the PID computation as a function of the number of PCA components. | Scope: task-specific. |

### `pipeline/analysis/pca_analysis/unique_search_outputs/unique_search_pca.py`

File description: Search source-1 PCA component subsets for source-1 unique PID information.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `run_pid_pc_subset_search (line 25)` | target: Any, source_1: Any, source_2: Any, pid_callable: Callable[..., Any] \| None=None, *, cmi_threshold: float=1e-06, unique_threshold: float=1e-06, beam_width: int=5, max_subset_size: int=3, initial_subset_size: int=1, initial_subset_count: int \| None=None, floating_tolerance: float=1e-09, max_runtime_seconds: float=600, rng_seed: int=56, pid_kwargs: dict[str, Any] \| None=None, all_csv_path: str \| Path \| None=None, best_csv_path: str \| Path \| None=None, use_floating_backward: bool=True | dict[str, Any] | Run beam search over source_1 PCA columns for source-1 unique PID. | Scope: task-specific. Related: _as_2d_array, _initial_subsets, _gaussian_cmi_bits, _top_rows. |
| `_initial_subsets (line 189)` | candidates: list[int], subset_size: int, subset_count: int \| None, rng_seed: int | list[tuple[int, ...]] | Create initial source_1 PC subsets for the beam search. | Scope: private helper. Related: add. |
| `_as_2d_array (line 210)` | value: Any, name: str | np.ndarray | Convert one input to a finite non-empty 2D float array. | Scope: private helper. |
| `_gaussian_cmi_bits (line 225)` | x: np.ndarray, y: np.ndarray, z: np.ndarray, eps: float=1e-10 | float | Calculate Gaussian conditional MI I(x; y \| z) in bits. | Scope: private helper. Related: _conditional_cov, _logdet. |
| `_conditional_cov (line 242)` | cov_a: np.ndarray, cross_ab: np.ndarray, cov_b: np.ndarray, eps: float | np.ndarray | Compute covariance of variable a conditioned on variable b. | Scope: private helper. |
| `_logdet (line 253)` | matrix: np.ndarray, eps: float | float | Return a stable log determinant for a covariance-like matrix. | Scope: private helper. |
| `_evaluate_subset (line 266)` | subset: tuple[int, ...], target: np.ndarray, source_1: np.ndarray, source_2: np.ndarray, pipeline: PIDPipeline, pid_kwargs: dict[str, Any], cache: dict[tuple[int, ...], dict[str, Any]], all_csv_path: str \| Path \| None, unique_threshold: float, start: float, cmi_score: float \| None | dict[str, Any] | Evaluate one source_1 PC subset with PIDPipeline and cache the result. | Scope: private helper. Related: run, _pid_components, _append_csv_row, _to_float. |
| `_floating_backward (line 310)` | row: dict[str, Any], target: np.ndarray, source_1: np.ndarray, source_2: np.ndarray, pipeline: PIDPipeline, pid_kwargs: dict[str, Any], cache: dict[tuple[int, ...], dict[str, Any]], all_csv_path: str \| Path \| None, unique_threshold: float, tolerance: float, start: float, max_runtime_seconds: float | dict[str, Any] | Prune PCs whose removal does not meaningfully reduce unique information. | Scope: private helper. Related: _evaluate_subset. |
| `_pid_components (line 346)` | pid_result: Any | dict[str, Any] | Extract a PID component dictionary from common project PID result shapes. | Scope: private helper. |
| `_to_float (line 362)` | value: Any | float | Convert numeric scalar-like values to float. | Scope: private helper. |
| `_top_rows (line 372)` | rows: list[dict[str, Any]], beam_width: int | list[dict[str, Any]] | Keep highest-unique rows, de-duplicated by subset. | Scope: private helper. |
| `_append_csv_row (line 383)` | path: str \| Path \| None, subset: tuple[int, ...], start: float, status: str, *, cmi_score: float \| None=None, row: dict[str, Any] \| None=None | None | Append one compact row to a CSV file, creating the header when needed. | Scope: private helper. Related: _to_float. |
| `_toy_pid (line 425)` | target: np.ndarray, source_1: np.ndarray, source_2: np.ndarray, **pid_kwargs: Any | dict[str, dict[str, float]] | Return a tiny Gaussian-CMI-based PID-like result for local smoke runs. | Scope: private helper. Related: _gaussian_cmi_bits. |


## pipeline/analysis/ridge_analysis

### `pipeline/analysis/ridge_analysis/pcIndex_predictions.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `each_pc_index_pred (line 37)` | model_name: str, n_pcs: list[int], pc_path: str \| Path, hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path | tuple[np.ndarray, int] | Predict individual target PCs from one model's selected layer. | Scope: task-specific. Related: prepare_target, prepare_ridge_target, nsd_feature_extraction, overall_best_layer. |
| `save_correlations_to_csv (line 110)` | correlations: np.ndarray, model_name: str, layer_index: int, pc_indexes: list[int], output_path: str \| Path | Path | Append one completed model's per-PC correlations to a checkpoint CSV. | Scope: task-specific. |
| `main (line 149)` | No inputs | None | Run every best-layer model and resume from a per-model CSV checkpoint. | Scope: entry point. Related: each_pc_index_pred, save_correlations_to_csv, add. |

### `pipeline/analysis/ridge_analysis/plot_pc_correlations.py`

File description: Plot per-PC ridge correlations for selected models on one figure.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `ModelCorrelationSeries (line 19)` | class | ModelCorrelationSeries instance | Store one model's layer and its correlations at individual PC indexes. | Scope: class. |
| `load_pc_correlation_series (line 28)` | csv_paths: Sequence[str \| Path], model_names: Sequence[str] | list[ModelCorrelationSeries] | Load selected models from CSV files produced by pcIndex_predictions.py. | Scope: task-specific. |
| `plot_pc_correlations (line 162)` | csv_paths: Sequence[str \| Path], model_names: Sequence[str], output_path: str \| Path, *, title: str='Ridge prediction correlation by PC index', dpi: int=300 | Path | Plot selected models' per-PC correlations together and save the figure. | Scope: task-specific. Related: load_pc_correlation_series. |
| `main (line 265)` | No inputs | Path | Run a three-model example using the alpha-search correlation CSV. | Scope: entry point. Related: plot_pc_correlations. |


## pipeline/full_OTC

### `pipeline/full_OTC/otc_experiment.py`

File description: Config-driven full-OTC experiment runner built on PIDPipeline.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `run_otc_experiment (line 26)` | config: dict[str, Any] | dict[str, Any] | Run one full-OTC experiment from an already-loaded config dictionary. | Scope: task-specific. Related: _print_workflow, run_configured_pid_pipeline. |
| `_print_workflow (line 52)` | config: dict[str, Any], function_registry: dict[str, PipelineFunction], started_at: datetime | None | Print the configured OTC function order, inputs, kwargs, and start time. | Scope: private helper. |
| `nsd_otc_target (line 129)` | hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, subj_id: str \| None=None, voxel_index: int \| None=None, n_images: int \| None=None | dict[str, Any] | Load the full OTC target matrix and expose it under the PIDPipeline target key. | Scope: task-specific. Related: prepare_target. |
| `_validate_config (line 170)` | config: dict[str, Any] | None | Validate the config sections needed to call the full-OTC runner. | Scope: private helper. Related: validate_pipeline_config_sections. |

### `pipeline/full_OTC/otc_run.py`

File description: Run the full-OTC PID experiment from the YAML config beside this file.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `make_str_as_path (line 17)` | config: dict[str, any] | dict[str, any] | Convert string paths in the config to Path objects. | Scope: task-specific. |
| `check_path_exists (line 26)` | config: dict[str, any] | None | Check if the paths in the config exist. | Scope: task-specific. |


## pipeline

### `pipeline/pid_pipeline.py`

File description: Strict, readable orchestration class for one PID pipeline run.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `PIDPipelineFunctions (line 10)` | class | PIDPipelineFunctions instance | Store the user-selected functions for one PID pipeline run. | Scope: class. |
| `PIDPipeline (line 42)` | class | PIDPipeline instance | Run the PID pipeline by connecting the provided functions in order. | Scope: class. Related: add_rng_to_kwargs, feature_extraction, inspect_function. |
| `PIDPipeline.__init__ (line 45)` | self, functions: PIDPipelineFunctions | None | Create a strict PID pipeline orchestrator. | Scope: method/nested helper. |
| `PIDPipeline.add_rng_to_kwargs (line 85)` | self, kwargs: dict[str, Any], func, rng: np.random.Generator | dict[str, Any] | Add the random number generator to the provided kwargs dictionary. | Scope: method/nested helper. Related: inspect_function. |
| `PIDPipeline.run (line 107)` | self, *, target_kwargs: dict[str, Any] \| None=None, sources_kwargs: dict[str, Any] \| None=None, choose_layer_kwargs: dict[str, Any] \| None=None, feature_extraction_kwargs: dict[str, Any] \| None=None, preprocess_kwargs: dict[str, Any] \| None=None, feature_manipulation_kwargs: dict[str, Any] \| None=None, pid_kwargs: dict[str, Any] \| None=None, report_kwargs: dict[str, Any] \| None=None | dict[str, Any] | Run target, sources, layers, features, optional transforms, PID, and report. | Scope: method/nested helper. Related: add_rng_to_kwargs, feature_extraction. |


## pipeline/pipeline_phases

### `pipeline/pipeline_phases/choosing_layer.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `random_layer_selection (line 9)` | n_layers | Unannotated return value; inspect implementation before reuse | Choose a random layer index from the available layers. | Scope: task-specific. |
| `specific_index_layer_selection (line 23)` | layer_names, index | Unannotated return value; inspect implementation before reuse | Choose a specific layer index from the available layers. | Scope: task-specific. |
| `voxel_best_layer (line 44)` | voxel_index: int=None, index_layer: int=None, path_to_results: str=None | dict | Choose the best model layer for one voxel, or a representative voxel for one layer. | Scope: task-specific. Related: _read_csv_rows. |
| `overall_best_layer (line 107)` | model_name: str, path_to_results: str | dict | Choose the overall best layer index for one model from an OTC CSV. | Scope: task-specific. Related: _read_csv_rows, _normalize_csv_value. |
| `_read_csv_rows (line 146)` | path_to_results: str | tuple[list[dict[str, str]], set[str]] | Read CSV rows and column names for layer-selection helpers. | Scope: private helper. |
| `_normalize_csv_value (line 163)` | value | str | Normalize CSV values before exact lookup comparisons. | Scope: private helper. |

### `pipeline/pipeline_phases/feature_manipulations.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `pca_projection (line 23)` | features, n_components | Unannotated return value; inspect implementation before reuse | Apply PCA projection to reduce dimensionality of features. | Scope: task-specific. |
| `jl_projection (line 40)` | features, n_samples, eps=0.1, jl_dim=None | Unannotated return value; inspect implementation before reuse | Apply Johnson-Lindenstrauss projection to reduce dimensionality of features. | Scope: task-specific. |
| `cca_projection (line 73)` | features1, features2, n_components | Unannotated return value; inspect implementation before reuse | Apply Canonical Correlation Analysis (CCA) to find linear combinations of two sets of features that are maximally correlated. | Scope: task-specific. |
| `prepare_ridge_target (line 106)` | target: np.ndarray, target_context: Mapping[str, Any], pc_target_path: str \| Path | tuple[np.ndarray, np.ndarray, np.ndarray] | Project a target with a saved PCA and split it by the shared mask. | Scope: task-specific. |
| `load_ridge_alphas (line 189)` | alphas_path: str \| Path, *, model_name: str, expected_target_dim: int, expected_layer_index: int \| None=None | np.ndarray | Load and validate a model's per-target-PC ridge penalties. | Scope: task-specific. Conventions: float64. |
| `ridge_predict_shared (line 314)` | source: np.ndarray, train_target: np.ndarray, shared_mask: np.ndarray, alphas: np.ndarray, *, seed: int | np.ndarray | Fit per-target ridge models and predict only the held-out rows. | Scope: task-specific. Related: fit. |
| `feature_manipulation_ridge (line 418)` | source1: np.ndarray, source2: np.ndarray, target: np.ndarray, target_context: Mapping[str, Any], seed: int, model_name_1: str, model_name_2: str, pc_target_path: str \| Path, alphas_source1_path: str \| Path, alphas_source2_path: str \| Path | tuple[np.ndarray, np.ndarray, np.ndarray] | Create held-out ridge predictions for two sources against PCA(T). | Scope: task-specific. Related: prepare_ridge_target, load_ridge_alphas, ridge_predict_shared. |
| `pca_source (line 502)` | source, shared_mask, max_features | Unannotated return value; inspect implementation before reuse | Train PCA on source features and return the held-out source features projected onto the PCA space. | Scope: task-specific. Related: fit. |

### `pipeline/pipeline_phases/mi_statistics.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `assert_mi (line 13)` | own_mi, pid_mi | Unannotated return value; inspect implementation before reuse | This function checks if the mutual information calculated by the PID method is equal to the mutual information calculated by the own method. | Scope: task-specific. |

### `pipeline/pipeline_phases/preprocessing_layer.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `permute_rv (line 22)` | source1, source2, target, source1_perm=False, source2_perm=False, target_perm=False, rng_seed=56 | Unannotated return value; inspect implementation before reuse | Permute the random variable rv according to the configuration provided in config. If rv is a tuple, all blocks are permuted with the same permutation. X is kept fixed, so any internal structure within X is preserved. | Scope: task-specific. |
| `_to_numpy (line 56)` | x | Unannotated return value; inspect implementation before reuse | Convert torch.Tensor to NumPy only if needed. | Scope: private helper. |
| `_from_numpy_like (line 63)` | x_np, reference | Unannotated return value; inspect implementation before reuse | Return x_np as torch if reference is torch, otherwise NumPy. | Scope: private helper. |
| `ridge_train_to_test_prediction (line 73)` | source_train, target_train, source_test, target_test=None, alphas=None, inner_cv: int=5, scoring: str='r2', shuffle: bool=True, random_state: int=0 | Unannotated return value; inspect implementation before reuse | Fit ridge encoding model on training data and return held-out test predictions. | Scope: task-specific. Related: _to_numpy, fit, _from_numpy_like. |
| `apply_saved_scaler (line 218)` | data: np.ndarray, scaler_path: str \| Path | np.ndarray | Transform an array with a previously fitted scaler. | Scope: task-specific. |
| `scale_func (line 262)` | source1: np.ndarray, source2: np.ndarray, target: np.ndarray, source1_scaler_path: str \| Path, source2_scaler_path: str \| Path, target_scaler_path: str \| Path | tuple[np.ndarray, np.ndarray, np.ndarray] | Scale two sources and a target with their saved fitted scalers. | Scope: task-specific. Related: apply_saved_scaler. |

### `pipeline/pipeline_phases/report_results.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `print_pid_mi (line 6)` | pid_results, mi_result | No explicit return; likely None / side effects | This functions takes the pid results and the mutual information results and prints them in a nice format. | Scope: task-specific. |

### `pipeline/pipeline_phases/sources_target_features.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `prepare_sources (line 21)` | model_name_1: str, model_name_2: str | dict[str, dict] | Prepare sources for feature extraction. | Scope: task-specific. |
| `prepare_target (line 42)` | hdf_path: Path, pkl_info_path: Path, neural_data_path: Path, n_samples: int=None | dict | Prepare target for feature extraction. | Scope: task-specific. |
| `shared1000_subj_target (line 79)` | hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, pca_model_path: str \| Path \| None=None, scaler_model_path: str \| Path \| None=None | dict[str, np.ndarray] | Load a subject's shared 1,000 stimuli and aligned neural responses. | Scope: task-specific. |
| `prepare_target_for_voxel (line 215)` | voxel_index: int, subj_id: str, hdf_path: Path, pkl_info_path: Path, neural_data_path: Path | dict | Prepare target for feature extraction for a specific voxel. | Scope: task-specific. Related: prepare_target. |
| `make_nsd_dataloader (line 241)` | model_context: dict, stim_dataset, image_ids: np.ndarray, batch_size: int | DataLoader | Create a DataLoader for an ordered subset of NSD images. | Scope: task-specific. |
| `batching (line 264)` | model_context: dict, batch_start: int, batch_end: int, stim_dataset, subj_image_ids: np.ndarray, layer_name: str, batch_size_dataloader: int | np.ndarray | Batch process a range of images for feature extraction. | Scope: task-specific. Related: make_nsd_dataloader. |
| `feature_extraction (line 295)` | layer_index: int, model_context: dict, subj_image_ids: np.ndarray, stim_dataset, batch_size_process: int, batch_size_dataloader: int=128 | np.ndarray | Extract features from the models and the neural data. | Scope: task-specific. Related: batching. |


## pipeline

### `pipeline/pipeline_utils.py`

File description: Utility helpers used by the thin PID pipeline orchestrator.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `choose_random_sources (line 19)` | sources_list: list[str], size: int=2, replace: bool=False | np.ndarray | Randomly select a source from the list of available sources. | Scope: task-specific. |
| `run_configured_pid_pipeline (line 35)` | config: dict[str, Any], function_registry: dict[str, PipelineFunction], choose_layer_kwargs: dict[str, Any] \| None=None | dict[str, Any] | Run PIDPipeline from a config dictionary and function registry. | Scope: task-specific. Related: pipeline_functions_from_config, run. |
| `pipeline_functions_from_config (line 68)` | function_config: dict[str, Any], function_registry: dict[str, PipelineFunction] | PIDPipelineFunctions | Resolve configured function names into PIDPipelineFunctions. | Scope: task-specific. Related: resolve_pipeline_function. |
| `resolve_pipeline_function (line 95)` | function_config: dict[str, Any], function_registry: dict[str, PipelineFunction], step_name: str, required: bool | PipelineFunction \| None | Resolve one configured pipeline step name from a registry. | Scope: task-specific. |
| `validate_pipeline_config_sections (line 124)` | config: dict[str, Any], required_sections: tuple[str, ...] | None | Validate that required config sections exist. | Scope: task-specific. |
| `nsd_sources (line 140)` | model_name_1: str, model_name_2: str | dict[str, dict[str, Any]] | Load two model contexts and expose them under X1 and X2. | Scope: task-specific. Related: prepare_sources. |
| `specific_layer_index (line 157)` | sources: dict[str, dict[str, Any]], X1_index: int, X2_index: int | dict[str, int] | Select one configured layer index for each source. | Scope: task-specific. Related: specific_index_layer_selection, _layer_index_values. |
| `random_layer_selection_for_sources (line 181)` | sources: dict[str, dict[str, Any]], random_seed: int \| None=None | dict[str, int] | Select a random valid layer index for each source. | Scope: task-specific. Related: _random_layer_index_for_source. |
| `voxel_best_layer_for_sources (line 203)` | sources: dict[str, dict[str, Any]], X1_index: int, X2_index: int, X1_path_to_results: str \| Path, X2_path_to_results: str \| Path | dict[str, int] | Select each source model's best layer for one voxel index. | Scope: task-specific. Related: voxel_best_layer. |
| `overall_best_layer_for_sources (line 236)` | sources: dict[str, dict[str, Any]], path_to_results: str \| Path | dict[str, int] | Select each source model's overall best OTC layer from one CSV file. | Scope: task-specific. Related: _model_name_for_source, overall_best_layer, _overall_best_layer_model_names. |
| `nsd_feature_extraction (line 267)` | source_context: dict[str, Any], layer_index: int, target_context: dict[str, Any], batch_size_process: int=128, batch_size_dataloader: int=128 | Any | Extract features for one NSD source and selected layer. | Scope: task-specific. Related: feature_extraction. |
| `pca_each_source (line 298)` | source_1: Any, source_2: Any, target: Any, n_components_source_1: int, n_components_source_2: int, n_components_target: int | tuple[Any, Any] | Apply PCA separately to source_1 and source_2. | Scope: task-specific. Related: pca_projection. |
| `pid_calc_adapter (line 336)` | target: Any, source_1: Any, source_2: Any, method: str, config: dict[str, Any] \| None=None, rng_seed: int=56, **pid_kwargs: Any | dict[str, Any] | Call pid_calc using the strict PIDPipeline array order. | Scope: task-specific. Related: _as_2d_tensor, pid_calc. |
| `print_pid_mi_adapter (line 385)` | pid_results: dict[str, Any], context: dict[str, Any], **report_kwargs: Any | Any | Print PID and MI outputs from pid_calc_adapter. | Scope: task-specific. Related: print_pid_mi. |
| `_as_2d_tensor (line 403)` | value: Any | Any | Convert samples to a 2D torch tensor. | Scope: private helper. |
| `_random_layer_index_for_source (line 424)` | sources: dict[str, dict[str, Any]], source_name: str, rng: np.random.Generator | int | Select one random layer index for one source context. | Scope: private helper. Related: source_context, random_layer_selection. |
| `_layer_index_values (line 454)` | sources: dict[str, dict[str, Any]], source_name: str, requested_index: int | list[int] | Create valid layer-index values for one source. | Scope: private helper. Related: source_context. |
| `_model_name_for_source (line 472)` | sources: dict[str, dict[str, Any]], source_name: str | str | Read a source model name from a source context. | Scope: private helper. Related: source_context. |
| `_overall_best_layer_model_names (line 489)` | path_to_results: str \| Path | list[str] | Read model names from an overall best-layer CSV for diagnostics. | Scope: private helper. |
| `source_context (line 511)` | sources: Any, source_name: str | Any | Read one source context from the sources object. | Scope: task-specific. |
| `choose_one_layer (line 527)` | layer_func: Callable[..., Any], source_context_value: Any, layer_kwargs: dict[str, Any] | Any | Choose one layer by adapting to the common layer-selection helper signatures. | Scope: task-specific. |
| `ridge_preprocessing (line 556)` | source_1: Any, source_2: Any, target: Any, rng, test_size, **preprocess_kwargs: Any | tuple[Any, Any, Any] | Apply ridge regression preprocessing to sources and target. | Scope: task-specific. Related: ridge_train_to_test_prediction, pca_projection. |


## pipeline/plotting

### `pipeline/plotting/plot_functions.py`

File description: Create matrix heatmaps from pairwise PID pipeline CSV results.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `plot_pairwise_pid_matrices (line 12)` | csv_path: str \| Path, output_dir: str \| Path, *, model_order: list[str] \| None=None, value_format: str='.3f', cmap: str='viridis', figsize: tuple[float, float] \| None=None, dpi: int=300 | dict[str, Path] | Create and save unique-information, redundancy, and synergy matrices. | Scope: task-specific. |
| `comulative_sum (line 229)` | array: np.ndarray | np.ndarray | Return the cumulative sum of a 1D array, ignoring NaN values. | Scope: task-specific. |


## pipeline/ridge_find_alpha

### `pipeline/ridge_find_alpha/find_alpha.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `find_alpha_per_pc (line 28)` | predictor: np.ndarray, target: np.ndarray, device: str \| None='cuda' | tuple[np.ndarray, Pipeline] | Find one ridge alpha per target PC. | Scope: task-specific. Related: fit. Conventions: float64. |
| `load_and_apply_pca (line 115)` | data: np.ndarray, pca_path: str \| Path, n_pcs: int \| None=None | np.ndarray | Transform raw neural data with a saved centered PCA model. | Scope: task-specific. |
| `split_alphas_csv_by_model (line 148)` | alphas_csv_path: str \| Path, output_dir: str \| Path \| None=None | list[Path] | Split an aggregate per-PC alpha CSV into one NumPy file per model. | Scope: task-specific. Related: add. |
| `main (line 270)` | source_name: str, path_to_results: str \| Path, pc_path: str \| Path, hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, alphas_csv_path: str \| Path, n_pcs: int \| None=None | tuple[np.ndarray, Pipeline] | Find raw-feature ridge alphas and save model/layer/PC metadata. | Scope: entry point. Related: prepare_target, load_and_apply_pca, overall_best_layer, nsd_feature_extraction. |
| `check_path_exists (line 399)` | config: dict[str, Any] | None | Require every configured input path before alpha generation. | Scope: task-specific. |


## pipeline

### `pipeline/ridge_pipeline.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `feature_manipulation_ridge (line 29)` | source1, source2, target, seed, source1_name, source2_name, pc_target_path, alphas_source1_path, alphas_source2_path, shared1000_subj | Unannotated return value; inspect implementation before reuse | Perform feature manipulation on the source and target data. | Scope: task-specific. Related: fit. |


## pipeline/subj_PCs

### `pipeline/subj_PCs/calculate_mean_mse.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `calculate_mean_mse (line 8)` | csv_path: Path, max_pcs: int, column_name: str='press_', n_features: int=8088 | dict[int, float] | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |
| `report_top_minimal_mean (line 20)` | mean_mse_dict: dict[int, float], top_n: int=5 | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: task-specific. |

### `pipeline/subj_PCs/plotting.py`

File description: Plot subject-level PCA variance and pairwise unique information.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `plot_cumulative_unique_information (line 12)` | csv_path: str \| Path, output_path: str \| Path \| None=None, *, model_1_column: str='model_1', model_2_column: str='model_2', unique_1_column: str='unq1', unique_2_column: str='unq2', dpi: int=300 | Path | Plot cumulative pairwise unique information for each model. | Scope: task-specific. |
| `plot_heldout_variance_explained (line 127)` | variance_csv_path: str \| Path, output_path: str \| Path \| None=None, *, show_cumulative: bool=True, show_training: bool=False, plot_training_minus_heldout: bool=False, separate_panels: bool=False, number_of_pcs: int \| None=None, dpi: int=300 | Path | Plot variance explained on held-out data by each retained PC. | Scope: task-specific. Related: add. |

### `pipeline/subj_PCs/subj_pc_analysis.py`

File description: Fit subject-level PCA models and evaluate them on shared held-out images.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `split_unique_shared (line 27)` | subj_id: str, hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, variance_threshold: float=0.99 | dict[str, Any] | Load one subject and split its rows into unique and shared-image sets. | Scope: task-specific. Related: prepare_target. |
| `pca_by_variance (line 90)` | neural_data: np.ndarray, variance_threshold: float=0.99 | dict[str, Any] | Fit centered PCA without per-feature variance standardization. | Scope: task-specific. |
| `heldout_pca (line 128)` | pca_model: PCA, heldout_data: np.ndarray | np.ndarray | Project raw held-out responses with a fitted centered PCA model. | Scope: task-specific. |
| `pca_func (line 154)` | data: np.ndarray, mode: str='eigenvector_CV', max_features: int \| None=None | dict[str, Any] | Select a component count and fit centered, unstandardized PCA. | Scope: task-specific. Related: pca_by_variance, eigenvector_pca_cv, estimate_ncp_pca. |
| `main (line 222)` | subj_id: str, hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, variance_threshold: float=0.99, save_models_path: str \| Path \| None=None, max_features: int \| None=None, pca_mode: str='missmda_CV' | dict[str, Any] | Fit centered unstandardized PCA and evaluate shared held-out data. | Scope: entry point. Related: split_unique_shared, pca_func, heldout_pca. |


## pipeline/toy_examples

### `pipeline/toy_examples/pid_pipeline_toy_example.py`

File description: Tiny no-data-loading example for debugging the strict PID pipeline flow.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `toy_target_extraction (line 17)` | No inputs | dict[str, Any] | Create tiny fake target data. | Scope: task-specific. |
| `toy_sources_extraction (line 30)` | model_1: str, model_2: str | dict[str, dict[str, Any]] | Create tiny fake source contexts with two layers per source. | Scope: task-specific. |
| `toy_choose_layer (line 61)` | sources: dict[str, dict[str, Any]], layer_index: int=0 | dict[str, str] | Choose one layer for each source by index. | Scope: task-specific. |
| `toy_feature_extraction (line 78)` | source_context: dict[str, Any], layer_name: str, target_context: dict[str, Any] | list[list[float]] | Read fake features for one selected source layer. | Scope: task-specific. |
| `toy_preprocess (line 98)` | target: list[list[float]], source_1: list[list[float]], source_2: list[list[float]], scale: float=1.0 | tuple[list[list[float]], list[list[float]], list[list[float]]] | Scale target and source values together as a visible preprocessing step. | Scope: task-specific. |
| `toy_feature_manipulation (line 123)` | source_1: list[list[float]], source_2: list[list[float]], keep_columns: int=1 | tuple[list[list[float]], list[list[float]]] | Keep the first columns from both source feature matrices. | Scope: task-specific. |
| `toy_pid_calculation (line 145)` | target: list[list[float]], source_1: list[list[float]], source_2: list[list[float]], method_name: str='toy_pid' | dict[str, Any] | Return a readable dummy PID result from tiny arrays. | Scope: task-specific. |
| `toy_pid_report (line 177)` | pid_results: dict[str, Any], context: dict[str, Any] | str | Print a compact toy pipeline report. | Scope: task-specific. |
| `main (line 198)` | No inputs | dict[str, Any] | Run the tiny toy PID pipeline. | Scope: entry point. Related: run. |

### `pipeline/toy_examples/voxel_experiment_smoke_example.py`

File description: No-data smoke example for the config-driven voxel experiment runner.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `smoke_voxel_target (line 17)` | voxel_index: int, subj_id: str, n_images: int=3 | dict[str, Any] | Create a tiny fake voxel target context. | Scope: task-specific. |
| `smoke_sources (line 38)` | model_name_1: str, model_name_2: str | dict[str, dict[str, Any]] | Create tiny fake source contexts. | Scope: task-specific. |
| `smoke_choose_layer (line 69)` | sources: dict[str, dict[str, Any]], X1_index: int, X2_index: int | dict[str, int] | Choose fake layer indexes from config. | Scope: task-specific. |
| `smoke_feature_extraction (line 85)` | source_context: dict[str, Any], layer_index: int, target_context: dict[str, Any], feature_shift: float=0.0 | list[list[float]] | Read fake source features for one selected layer. | Scope: task-specific. |
| `smoke_feature_manipulation (line 111)` | source_1: list[list[float]], source_2: list[list[float]], keep_columns: int=1 | tuple[list[list[float]], list[list[float]]] | Keep the first columns from both fake source matrices. | Scope: task-specific. |
| `smoke_pid_calculation (line 133)` | target: list[list[float]], source_1: list[list[float]], source_2: list[list[float]], method: str='smoke_pid' | dict[str, Any] | Return a tiny deterministic PID-like result. | Scope: task-specific. |
| `smoke_report (line 167)` | pid_results: dict[str, Any], context: dict[str, Any] | str | Print a compact smoke run report. | Scope: task-specific. |
| `register_smoke_functions (line 188)` | No inputs | None | Register smoke wrapper functions for this example run. | Scope: task-specific. |
| `smoke_config (line 211)` | No inputs | dict[str, Any] | Create a small YAML-shaped config for run_voxel_experiment. | Scope: task-specific. |
| `main (line 242)` | No inputs | dict[str, Any] | Run the voxel experiment smoke example. | Scope: entry point. Related: register_smoke_functions, run_voxel_experiment, smoke_config. |


## pipeline/voxel_experiments

### `pipeline/voxel_experiments/voxel_experiment.py`

File description: Config-driven voxel experiment runner built on PIDPipeline.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `run_voxel_experiment (line 23)` | config: dict[str, Any] | dict[str, Any] | Run one voxel experiment from an already-loaded config dictionary. | Scope: task-specific. Related: _validate_config, _choose_layer_kwargs, run_configured_pid_pipeline. |
| `nsd_voxel_target (line 43)` | voxel_index: int, subj_id: str, hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, n_images: int \| None=None | dict[str, Any] | Load one voxel target and expose it under the PIDPipeline target key. | Scope: task-specific. Related: prepare_target_for_voxel. |
| `_choose_layer_kwargs (line 84)` | config: dict[str, Any] | dict[str, Any] | Arrange voxel-specific kwargs for the configured choose_layer function. | Scope: private helper. |
| `_validate_config (line 100)` | config: dict[str, Any] | None | Validate the config sections needed to call the voxel runner. | Scope: private helper. Related: validate_pipeline_config_sections. |

### `pipeline/voxel_experiments/voxel_run.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.


## source_conwell_code

### `source_conwell_code/__init__.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.


## source_conwell_code/pressures

### `source_conwell_code/pressures/__init__.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.


## source_conwell_code/pressures/brain_data

### `source_conwell_code/pressures/brain_data/__init__.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `average_rdms (line 7)` | rdm_array | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: fisherz_inv, fisherz. |
| `average_rdms.fisherz (line 8)` | r, eps=1e-05 | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `average_rdms.fisherz_inv (line 11)` | z | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `NSDBenchmark (line 17)` | class | NSDBenchmark instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. Related: get_rdm_indices, get_rdms, get_splithalf_rdms, average_rdms. |
| `NSDBenchmark.__init__ (line 18)` | self, image_set='shared1000', voxel_set='OTC-only', train_test_split=False, clean_rdms_only=True, anatomical_roi_subset=None, functional_roi_subset=None | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. Related: get_rdm_indices, get_rdms, get_splithalf_rdms. |
| `NSDBenchmark.get_sample_stimulus (line 80)` | self, image_index=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `NSDBenchmark.get_rdm_indices (line 90)` | self, roi_subset=None, row_number=False | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `NSDBenchmark.get_rdms (line 111)` | self, roi_subset=None, include_group_average=False | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. Related: get_rdm_indices, average_rdms. |
| `NSDBenchmark.get_splithalf_rdms (line 136)` | self | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |

### `source_conwell_code/pressures/brain_data/benchmark.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.

### `source_conwell_code/pressures/brain_data/nsd_parser.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `get_nsd_path (line 21)` | config=None, key='NSD_PATH' | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `_nsd_path_prompt (line 42)` | No inputs | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `_check_space (line 69)` | space | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `get_subj_dims (line 73)` | subj, space='func1pt8mm' | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: _check_space. |
| `load_voxel_info (line 81)` | subj, space | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `load_NSD_voxel_metadata (line 210)` | subjs, roi_group, space, voxels_to_include=None, savedir=None, overwrite=False | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: load_voxel_info, get_subj_dims. |
| `load_NSD_brain_data (line 356)` | subjs, space, roi_group, voxel_metadata, annotations, savedir, output=False | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `load_NSD_benchmark_ROI_metadata (line 455)` | subjs, space, ncsnr_threshold=0.2, t_threshold=1, savedir=None, overwrite=False, **kwargs | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: load_NSD_voxel_metadata, load_voxel_info, logic, get_subj_dims. |
| `load_NSD_benchmark_ROI_metadata.get_idx_by_roi_group (line 464)` | metadata, group_label, roi_names, logic='include', t_threshold=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. Related: logic. |
| `load_NSD_benchmark_ROI_metadata.get_idx_by_roi_group.logic (line 472)` | values, names | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `load_NSD_benchmark_ROI_metadata.get_idx_by_roi_group.logic (line 472)` | values, names | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |


## source_conwell_code/pressures

### `source_conwell_code/pressures/main_analysis.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `get_splithalf_xy (line 13)` | feature_map, response_data, scale=True | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `permute_benchmark (line 26)` | benchmark | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: get_rdms. |
| `run_benchmarking (line 30)` | benchmark, model_option, precomputed_feature_maps=None, layers_to_retain=None, metrics=['crsa', 'srpr', 'wrsa'], alpha_values=np.logspace(-1, 5, 7).tolist(), regression_means=True, precompute_rdms=True | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: get_splithalf_rdms, get_rdm_indices, permute_benchmark, get_splithalf_xy. |
| `get_results_max (line 166)` | results, average_over=None, average_when='after' | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |

### `source_conwell_code/pressures/ridge_gcv_mod.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `pearson_r_score (line 19)` | y_true, y_pred, multioutput=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `_RidgeGCVMod (line 24)` | class bases: _RidgeGCV | _RidgeGCVMod instance | Ridge regression with built-in Leave-one-out Cross-Validation. | Scope: class. Related: pearson_r_score. |
| `_RidgeGCVMod.__init__ (line 27)` | self, alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, copy_X=True, gcv_mode=None, store_cv_values=False, is_clf=False, alpha_per_target=False | No explicit return; likely None / side effects | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `_RidgeGCVMod.fit (line 50)` | self, X, y, sample_weight=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. Related: pearson_r_score. Conventions: float64. |
| `_BaseRidgeCVMod (line 174)` | class bases: _BaseRidgeCV | _BaseRidgeCVMod instance | No docstring; verify behavior, types, and conventions before reuse. | Scope: class. Related: fit. |
| `_BaseRidgeCVMod.fit (line 175)` | self, X, y, sample_weight=None | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: method/nested helper. |
| `RidgeCVMod (line 226)` | class bases: MultiOutputMixin, RegressorMixin, _BaseRidgeCVMod | RidgeCVMod instance | Ridge regression with built-in cross-validation. | Scope: class. |


## supression_effect

### `supression_effect/Suppresed_Encoder.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `_append_project_root_to_path (line 8)` | No inputs | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `get_run_config (line 37)` | No inputs | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `load_model_and_fmri (line 63)` | config: dict | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: train_save_or_load. |
| `prepare_inputs (line 70)` | config: dict, real_features: np.ndarray, fmri_dict: dict | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: get_specific_roi_fmri, create_encoder. |
| `run_suppression_pipeline (line 94)` | config: dict, selected_features: np.ndarray, encoder, verbose: bool=True | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: create_predictions, create_supression_model, commonality_analysis, idep. |
| `save_all_seed_runs_results (line 129)` | seed_rows: list[dict], config: dict | Path | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: get_seed_runs_csv_path, append_seed_run_checkpoint. |
| `run_single_seed (line 149)` | seed: int, config: dict, real_features: np.ndarray, fmri_dict: dict | dict | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: prepare_inputs, run_suppression_pipeline, extract_all_components. |
| `run_encoding_multi_seed_experiment (line 168)` | config: dict | tuple[dict, list[dict]] | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: load_model_and_fmri, run_multi_seed_experiment, run_single_seed. |
| `save_seed_summary (line 181)` | summary: dict, config: dict | Path | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: save_seed_summary_csv. |
| `print_results (line 185)` | outputs: dict, mi: dict, pid: dict | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |
| `main (line 199)` | No inputs | None | No docstring; verify behavior, types, and conventions before reuse. | Scope: entry point. Related: get_run_config, load_model_and_fmri, run_configured_multiseed, run_single_seed. |

### `supression_effect/__init__.py`

File description: No module docstring; inspect before reuse.

No functions, methods, or classes defined in this file.

### `supression_effect/gauss_univariate.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `gauss_simple_example (line 23)` | N=1000, P=1, rng_seed=1, noise_seed=1, simple_example=True, snr=1.0, method='ridge_cv', mixing_dimension=None | Unannotated return value; inspect implementation before reuse | Run a simple Gaussian example with no mixing deminsion each experiement has different random seeds. | Scope: reusable/public. Related: run_experiment, idep. |
| `check_supression_effect (line 76)` | vp_results, pid_results | Unannotated return value; inspect implementation before reuse | Check for suppression effect in the results. | Scope: reusable/public. |
| `main (line 122)` | No inputs | No explicit return; likely None / side effects | Main function to run the Gaussian simple example and compare results. | Scope: entry point. Related: gauss_simple_example, compare_results, check_supression_effect. |

### `supression_effect/supp_gauss_multivariate.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `crossfit_residualize (line 42)` | Y_raw, X_raw, n_splits=5, seed=0 | Unannotated return value; inspect implementation before reuse | Residualize Y_raw against X_raw using cross-fitted linear regression. Returns residuals Y - E[Y\|X] predicted out-of-fold. | Scope: reusable/public. Related: fit. Conventions: float64. |
| `test_suppression (line 60)` | N=1000, P=1, suppression_strength=0.5, rng_seed=1, mode='simple', snr=1.0, method='ridge_cv', mixing_dimension=None | Unannotated return value; inspect implementation before reuse | Run a simple Gaussian example with no mixing deminsion each experiement has different random seeds. | Scope: reusable/public. Related: run_experiment, idep. |
| `plot_pid_results (line 119)` | mi_results=None, pid_results=None, sub_title=None | No explicit return; likely None / side effects | Plot bar chart for PID results. | Scope: reusable/public. |
| `get_seed_sweep_config (line 167)` | No inputs | dict | Configuration for fixed-parameter suppression simulations across seeds. | Scope: reusable/public. |
| `run_single_seed_fixed (line 189)` | seed: int, config: dict | dict | Run one suppression experiment seed with all other parameters fixed. | Scope: reusable/public. Related: test_suppression, extract_all_components. |
| `run_fixed_params_across_seeds (line 204)` | config: dict \| None=None | tuple[dict, list[dict]] | Sweep over seeds while keeping all other simulation parameters fixed, then save the per-seed results and mean/std summary. | Scope: reusable/public. Related: run_multi_seed_experiment, print_seed_summary, get_seed_runs_csv_path, save_seed_summary_csv. |
| `main (line 232)` | No inputs | No explicit return; likely None / side effects | Main function to run the Gaussian simple example and compare results. | Scope: entry point. Related: test_suppression, compare_results. |


## toy_examples

### `toy_examples/Theortical_cov_toy_example.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `make_spd_matrix (line 15)` | p, rng, eig_min=0.5, eig_max=2.0 | Unannotated return value; inspect implementation before reuse | Create a random symmetric positive definite covariance matrix. | Scope: reusable/public. |
| `check_spd (line 30)` | Sigma, name='Sigma', tol=1e-10 | No explicit return; likely None / side effects | Check whether a matrix is symmetric positive definite. | Scope: reusable/public. |
| `theoretical_covariance_multivariate (line 52)` | Sigma_R, Sigma_U, Sigma_N, Sigma_eps, order=('X1', 'X2', 'Y') | Unannotated return value; inspect implementation before reuse | Theoretical covariance for the multivariate generative process: | Scope: reusable/public. Related: check_spd. |
| `simulate_multivariate_process (line 123)` | n, Sigma_R, Sigma_U, Sigma_N, Sigma_eps, rng=None | Unannotated return value; inspect implementation before reuse | Simulate: | Scope: reusable/public. |
| `extract_covariance_blocks (line 161)` | Sigma_full, p | Unannotated return value; inspect implementation before reuse | Extract covariance blocks assuming order [X1, X2, Y]. | Scope: reusable/public. |
| `whiten_theoretical_covariance_blocks (line 187)` | Sigma_full, p, device='cpu', dtype=torch.float64 | Unannotated return value; inspect implementation before reuse | Given the full covariance matrix of [X1, X2, Y], compute the whitened cross-covariance blocks: | Scope: reusable/public. Related: whiten_block. Conventions: float64, whitening-sensitive. |
| `validate_multivariate_covariance (line 281)` | n=1000000, p=5, seed=123, device='cpu' | Unannotated return value; inspect implementation before reuse | Validate the theoretical covariance by simulation, then compute whitened covariance blocks. | Scope: reusable/public. Related: make_spd_matrix, theoretical_covariance_multivariate, simulate_multivariate_process, whiten_theoretical_covariance_blocks. Conventions: whitening-sensitive. |

### `toy_examples/simple_problems.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `bern_pair_as_channels (line 8)` | p1=0.5, p2=0.5 | Unannotated return value; inspect implementation before reuse | Convert two independent Bernoulli(p1), Bernoulli(p2) into PXgS, PYgS, PS format compatible with computeQUI_numpy. | Scope: reusable/public. |
| `dist_from_tensor (line 59)` | Q, names=('S', 'X', 'Y') | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. |

### `toy_examples/suppression_pipeline_example.py`

File description: Example script demonstrating the use of the suppression analysis pipeline.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `example_with_synthetic_data (line 24)` | No inputs | Unannotated return value; inspect implementation before reuse | Example using synthetic data to demonstrate the pipeline. | Scope: reusable/public. Related: fit, suppression_analysis_pipeline. |
| `example_with_pretrained_models (line 109)` | No inputs | Unannotated return value; inspect implementation before reuse | Example showing how to use the pipeline with pre-trained models. | Scope: reusable/public. Related: fit, suppression_analysis_pipeline. |

### `toy_examples/suppression_toy_runner.py`

File description: Shared runners for suppression toy examples.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `generate_correlated_features (line 12)` | n, p, rho, rng | Unannotated return value; inspect implementation before reuse | Generate samples with an AR(1)-style covariance structure. | Scope: reusable/public. |
| `_apply_mixing (line 19)` | rng, X_M1, X_M2, mixing_dimension | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `_split_signal_sources (line 28)` | rng, n, p, snr | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. |
| `_feature_correlation_sources (line 50)` | rng, n, p, snr, rho | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: private helper. Related: generate_correlated_features. |
| `build_toy_sources (line 69)` | rng, n, p, snr, experiment_kind, rho=1 | Unannotated return value; inspect implementation before reuse | No docstring; verify behavior, types, and conventions before reuse. | Scope: reusable/public. Related: _split_signal_sources, _feature_correlation_sources. |
| `run_toy_experiment (line 80)` | rng, n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', experiment_kind=SPLIT_SIGNAL, rho=1 | Unannotated return value; inspect implementation before reuse | Run one toy suppression/commonality experiment. | Scope: reusable/public. Related: build_toy_sources, _apply_mixing, commonality_analysis. |
| `run_all_toy_methods (line 128)` | rng_seed, n, p, mixing_dimension, snr, experiment_kind, methods=DEFAULT_METHODS, report_negative_common=False, rho=1 | Unannotated return value; inspect implementation before reuse | Run all requested commonality methods with a fixed seed. | Scope: reusable/public. Related: run_toy_experiment. |
| `run_default_factorial_scenarios (line 172)` | experiment_kind, n=1000, p=100, seed=42, report_negative_common=False | No explicit return; likely None / side effects | Run the standard low/high SNR by mixing-dimension toy scenarios. | Scope: reusable/public. Related: run_all_toy_methods. |

### `toy_examples/toy_example.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `unq2_zero_with_red_unq1_syn (line 15)` | rng, n, p, noise_std=0.9 | Unannotated return value; inspect implementation before reuse | Continuous Gaussian-like example where theoretically: | Scope: reusable/public. |
| `run_experiment (line 91)` | rng, suppresion_strength, mode='permuted', n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', show_diagnostic_plots=False | Unannotated return value; inspect implementation before reuse | Run commonality analysis experiment. | Scope: reusable/public. Related: commonality_analysis, diagnostic_plots, permutate_models, unq2_zero. |
| `run_ridge_toy_method (line 175)` | rng_seed, n, p, mixing_dimension, snr | Unannotated return value; inspect implementation before reuse | Run the ridge-CV analysis for this specialized toy example. | Scope: reusable/public. Related: run_experiment. |
| `main (line 196)` | No inputs | No explicit return; likely None / side effects | Run the 2x3 factorial experiment design. | Scope: entry point. Related: run_ridge_toy_method. |
| `plot_components (line 240)` | comp_dict, title='Variance Decomposition' | No explicit return; likely None / side effects | Plot bar chart for component dictionary. | Scope: reusable/public. |
| `gauss_simple_example (line 259)` | No inputs | No explicit return; likely None / side effects | Run a simple Gaussian example with no mixing deminsion each experiement has different random seeds. | Scope: reusable/public. Related: run_ridge_toy_method, plot_components. |

### `toy_examples/toy_example_feature_correlation.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `main (line 16)` | No inputs | No explicit return; likely None / side effects | Run the 2x3 factorial experiment design. | Scope: entry point. Related: run_default_factorial_scenarios. |

### `toy_examples/toy_example_new.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / relationships / conventions |
|---|---|---|---|---|
| `main (line 16)` | No inputs | No explicit return; likely None / side effects | Run the 2x3 factorial experiment design. | Scope: entry point. Related: run_default_factorial_scenarios. |
