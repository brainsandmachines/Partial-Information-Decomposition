# Function Registry

Purpose: search this document before creating, moving, or changing a function. Reuse an existing callable only after verifying its data order, shapes, estimator convention, and side effects.

Scope: tracked project Python files under `pid/`, `pipeline/`, and `simulations/`. Excludes `external/`, `tests/`, tool metadata, generated caches, and ignored local code.

Generated from the current AST after the repository reorganization. Paths, signatures, and line numbers reflect the working tree.

## Convention checks before reuse

- Verify random-variable order: `[X1, X2, T]`, `[T, X1, X2]`, or another documented order.
- Verify samples versus covariance input, raw versus whitened covariance, NumPy versus torch, device/dtype, biased versus unbiased estimation, log base, and PID definition.
- Keep representation-level and prediction-level pairwise analyses distinct.
- Keep learned PCA/scaling fits inside training data and apply them unchanged to held-out data.

## Folder Overview

- `pid/`: 5 Python files.
- `pipeline/`: 3 Python files.
- `pipeline/analysis/`: 1 Python file.
- `pipeline/analysis/pairwise/`: 2 Python files.
- `pipeline/analysis/pca/`: 4 Python files.
- `pipeline/analysis/predictivity/`: 2 Python files.
- `pipeline/examples/`: 2 Python files.
- `pipeline/experiments/`: 4 Python files.
- `pipeline/pca/`: 5 Python files.
- `pipeline/pca/cross_decomposition/`: 2 Python files.
- `pipeline/phases/`: 4 Python files.
- `pipeline/testing/gsn/`: 2 Python files.
- `simulations/`: 4 Python files.
- `simulations/evil_twin/`: 4 Python files.
- `simulations/pca/rank/`: 5 Python files.
- `simulations/pca/ridge/`: 3 Python files.
- `simulations/rsa_vs_pid/`: 3 Python files.

## `pid/__init__.py`

File description: No module docstring; inspect before reuse.

No functions or classes defined in this file.

## `pid/bias.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `logdet_wishart_bias (line 4)` | df: int, d: int | float | Exact finite-sample bias for log\|S\| when S is the unbiased sample covariance | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device. |
| `mi_wishart_bias (line 24)` | dims: list, n_samples: int | Unannotated | Bias correction for Gaussian mutual information estimates | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device; verify random-variable order and logarithm base. |

## `pid/calculation.py`

File description: Eigen-PID calculation with the project's sample/covariance interface.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `pid_calc (line 9)` | config: dict \| None=None, sources: list \| None=None, target: list \| None=None, rng: torch.Generator \| None=None, method: str \| None=None, on_rvs: callable=None, covariance: torch.Tensor \| None=None | tuple[dict, dict] | Calculate Gaussian BROJA PID in bits with Eigen-PID. | Scope: reusable/public. Related: external `eigen_pid`, external `debias_eigen_pid`. Conventions: data/covariance order [T, X1, X2]; `bias_correction=False` calls `eigen_pid`; `bias_correction=True` requires samples and returns `debias_eigen_pid`'s combined estimate; outputs are bits. |

## `pid/mutual_information.py`

File description: Gaussian mutual-information calculation from a joint covariance matrix.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `gaussian_mutual_information (line 11)` | covariance: torch.Tensor \| np.ndarray, x_dim: int, *, n_samples: int \| None=None, bias_correction: Literal['wishart'] \| None=None | dict[str, float \| str \| None] | Calculate Gaussian MI between two covariance blocks and report raw, bias, and corrected values. | Scope: reusable/public. Related: `mi_wishart_bias`. Conventions: covariance order `[X, Y]`; output bits; Wishart correction requires an unbiased Gaussian sample covariance and `n_samples > dim(X,Y)`; corrected MI is not clipped. |

## `pid/utils.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `LinearRegression_fit (line 11)` | X, y | Unannotated | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `compute_ridge_cv_r2 (line 25)` | X, y, alphas=None | Unannotated | Compute cross-validated R² using RidgeCV with efficient LOO cross-validation. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify scaling, alpha, and train/test conventions. |
| `cond_cov (line 61)` | sigma_1, sigma_2, sigma12, sigma21 | Unannotated | This function will compute the conditional covariance matrix of two Gaussian variables | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device. |
| `ledoit_wolf_cov_torch (line 75)` | X: torch.Tensor, assume_centered: bool=False | torch.Tensor | Fit Ledoit-Wolf on X (N,פ) and return covariance as torch.Tensor on same device/dtype. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device. |
| `create_cov_matrix (line 87)` | rvs: list=[], verbose=False, Sigma=None, dims: list=None, device='cpu', check_singular=True | Unannotated | This function will create the covariance matrix for the three variables M1,M2,T | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device. |
| `reorder_cov_blocks (line 152)` | Sigma: torch.Tensor, dims: dict[str, int], old_order: list[str], new_order: list[str] | torch.Tensor | Reorder covariance matrix blocks according to variable names. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device. |
| `para_create_cov_matrix (line 193)` | dims, Sigmas=None, verbose=False | Unannotated | This function will create the covariance matrix for the three variables M1,M2,T | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device. |
| `whiten_block (line 241)` | Sigma_xx: torch.Tensor, Sigma_xy: torch.Tensor, Sigma_yy: torch.Tensor | torch.Tensor | return Ux^{-T} @ Sigma_xy @ Uy^{-1} | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device. |
| `plot_cov_blocks (line 258)` | cov_dict, x0_dim, x1_dim, x2_dim, *, title='Covariance (block view)', cmap='Blues', vmin=None, vmax=None, fine_grid=False, show_colorbar=True | Unannotated | Plot covariance blocks and save the figure as PDF. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device. |
| `standardize (line 299)` | X: torch.Tensor, eps: float=1e-12 | torch.Tensor | Standardize columns of X to zero mean and unit variance. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `assert_full_rank (line 311)` | X: torch.Tensor, jitter=0 | None | Assert that the input matrix X is full rank. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `correlation_matrix (line 335)` | X | Unannotated | Compute the correlation matrix of the columns of X. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `block_singularity_check (line 350)` | X, tol=1e-10 | Unannotated | Check if a block is singular or ill-conditioned. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `singularity_report (line 371)` | X_M1, X_M2, y_real, tol=1e-10, return_printing_required=False | Unannotated | Return min eigenvalue and singularity flag for blocks and combinations. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `diagnostic_plots (line 401)` | X_M1, X_M2, y_real, method, mixing_dimension | Unannotated | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `diagnostic_plots.cross_correlation (line 402)` | X, Y | Unannotated | No docstring; inspect implementation before reuse. | Scope: method/nested helper. Conventions: verify PID definition, source/target order, and bit units. |
| `vif_summary (line 436)` | X | Unannotated | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `std_scaling_summary (line 468)` | X | Unannotated | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `eigvenvalue_summary (line 487)` | X | Unannotated | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `_get_first (line 495)` | mapping, *keys | Unannotated | No docstring; inspect implementation before reuse. | Scope: private helper. Conventions: verify PID definition, source/target order, and bit units. |
| `compare_results (line 502)` | vp_results, pid_results, mi_results=None | Unannotated | Compare Variance Partitioning and Partial Information Decomposition results. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify random-variable order and logarithm base. |
| `pid_comparison_table (line 548)` | results: dict, decimals: int=4, print_table: bool=True | Unannotated | Print and return a compact table comparing PID and MI outputs. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `save_pid_comparison_table (line 577)` | results: dict, save_path: str, decimals: int=4, title: str='PID Method Comparison', config: dict=None | Unannotated | Save the PID comparison table as a clean Matplotlib PDF. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `commonality_comparison_table (line 621)` | results: dict, decimals: int=4, print_table: bool=True | list[dict] | Normalize, print, and return commonality-analysis result rows. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `save_commonality_comparison_table (line 699)` | results: dict, save_path: str, decimals: int=4, title: str='Commonality Analysis Comparison', config: dict \\| None=None | str | Save normalized commonality results as a Matplotlib PDF table. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `commanility_analysis (line 796)` | results: dict, decimals: int=4, print_table: bool=True | list[dict] | Call the commonality table formatter through its legacy misspelling. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `plot_mi_heatmap (line 819)` | csv_path, value_col, *, n_col='N', p_col='p', figsize=(7, 5), title=None, save_path=None, annotate=True, fmt='.3f', cmap='viridis' | Unannotated | Plot a block heatmap from an averaged CSV and save it as PDF when requested. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify random-variable order and logarithm base. |
| `plot_all_mi_heatmaps (line 939)` | csv_path, title='Mutual Information Heatmaps', *, n_col='N', p_col='p', figsize=(16, 5), save_path=None, annotate=True, mean_fmt='.2f', std_fmt='.2f', log_scale=False, cmap='viridis', annotation_mode='pm', fontsize=9, aggfunc='mean' | Unannotated | Plot theoretical, naive, and bias-corrected MI heatmaps in one PDF figure. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify random-variable order and logarithm base. |
| `plot_block_heatmap (line 1106)` | csv_path, save_path=None | Unannotated | Plot the MI block heatmap and save it as PDF when requested. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `anchored_oas_shrinkage (line 1166)` | Sigma_full: torch.Tensor, cov_loo_all: torch.Tensor, n_samples: int | Unannotated | Calculates OAS parameters ONCE on the full matrix, | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `oas_cov_torch (line 1207)` | S: torch.Tensor, N: int | torch.Tensor | Apply Oracle Approximating Shrinkage (OAS) to a covariance matrix. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device. |
| `residual_rvs (line 1238)` | rv_list: list, predictor_index=0 | Unannotated | Given a list of random variables (Torch.Tensors), | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `save_sample_simulation_results_table (line 1263)` | results: dict, config: dict, save_path: str \\| pathlib.Path, decimals: int=4, title: str='PID Method Comparison', dpi: int=200 | pathlib.Path | Save sample-simulation PID/MI summaries as a styled PDF table. | Scope: reusable/public. Related: PCA–Ridge simulation reporting. Conventions: verify PID definition, source/target order, and bit units. |

## `pipeline/analysis/pairwise/pairwise.py`

File description: Run resumable pairwise PID across unordered model pairs; `source_mode` in the YAML selects prediction-level (`prediction.yaml`) or representation-level (`representation.yaml`) sources.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `run_pairwise_pid_pipeline (line 47)` | model_1_names: list[str], model_2_names: list[str], otc_config: dict[str, Any], csv_path: str \| Path | Path | Run PID once per unfinished unordered model pair in either source mode and append each row to a resumable CSV. | Scope: reusable/public. Conventions: `source_mode` defaults to "prediction"; PID input order (target, X1, X2); held-out shared images only; outputs bits; config checks run before the NSD target load. |
| `_build_mode_input_preparer (line 132)` | source_mode: str, config: dict[str, Any], registry: dict[str, Any], required_models: list[str], layer_results_path: Path, seed: int | Callable[[np.ndarray, dict[str, Any]], tuple[np.ndarray, SourcePreparer]] | Validate the selected pairwise source mode and bind its mode-specific input-preparation arguments. | Scope: private helper. Related: `prepare_prediction_inputs`, `_representation_inputs`. Conventions: configuration errors are raised before the NSD target load; add future source-mode branching here rather than in the main runner. |
| `_representation_inputs (line 180)` | raw_target: np.ndarray, target_context: dict[str, Any], config: dict[str, Any], layer_results_path: Path, n_source_components: int, seed: int | tuple[np.ndarray, SourcePreparer] | Fit target PCA on unique images, project shared images, and bind the PCA source preparer. | Scope: private helper. Related: `_fit_unique_project_shared`, `_prepare_representation_projection`. |
| `main (line 214)` | config_path: str \| Path=DEFAULT_CONFIG_PATH | None | Run the YAML-configured pairwise analysis and plot the exact CSV it wrote. | Scope: entry point. Conventions: relative config paths resolve from the repository root; default config is `prediction.yaml`. |
| `parse_args (line 248)` |  | argparse.Namespace | Parse the `--config` command-line option. | Scope: entry point. |

## `pipeline/analysis/pairwise/pairwise_utils.py`

File description: Shared helpers for pairwise PID and cumulative target-PC analyses, including target preparation, model loading, checkpointing, and reusable source preparation.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `safe_model_name (line 58)` | model_name: str | str | Replace path separators in a model identifier for artifact filenames. | Scope: reusable/public. |
| `pop_raw_target (line 71)` | target_context: dict[str, Any] | np.ndarray | Remove and return the full neural target, dropping the large arrays from the context. | Scope: reusable/public. Conventions: returns (n_images, n_voxels). |
| `_resolve_project_path (line 93)` | path_value: str \| Path | Path | Resolve a configured path relative to the repository when necessary. | Scope: private helper. |
| `_load_or_create_checkpoint (line 109)` | output_path: Path | pd.DataFrame | Load a compatible pairwise CSV or create an empty checkpoint. | Scope: private helper. Conventions: schema is `PAIRWISE_RESULT_COLUMNS`. |
| `_unfinished_unordered_pairs (line 138)` | source_1_names: list[str], source_2_names: list[str], existing_results: pd.DataFrame | list[tuple[str, str]] | Choose the first requested orientation of every unfinished pair. | Scope: private helper. Conventions: skips self pairs and pairs completed in either orientation. |
| `required_models (line 167)` | pairs: list[tuple[str, str]] | list[str] | Return each model needed by pending pairs once in first-use order. | Scope: reusable/public. |
| `_result_row (line 187)` | model_1: str, model_2: str, layer_1: int, layer_2: int, target: np.ndarray, source_1: np.ndarray, source_2: np.ndarray, pid_results: Mapping[str, Any], config: Mapping[str, Any] | dict[str, Any] | Build one CSV row for a finished pair. | Scope: private helper. Conventions: `subj_id` from `metadata.subj_id`; unq1 belongs to X1 = model_1. |
| `_layer_results_path (line 237)` | config: Mapping[str, Any] | Path | Resolve and require the best-layer results CSV. | Scope: private helper. |
| `_check_prediction_config (line 261)` | config: dict[str, Any], registry: dict[str, Any] | tuple[Callable[..., Any], Path] | Resolve feature extraction and require the saved target-PCA artifact before loading data. | Scope: private helper. Conventions: prediction mode only. |
| `_check_representation_config (line 291)` | config: dict[str, Any] | int | Require equal source PCA sizes before loading data. | Scope: private helper. Conventions: representation mode only. |
| `_best_layer_index (line 318)` | model_name: str, layer_results_path: Path | int | Look up one model's overall best layer index. | Scope: private helper. Related: `overall_best_layer`. |
| `_check_layer_index (line 343)` | model_context: Mapping[str, Any], model_name: str, layer_index: int | None | Require the selected layer to exist in a loaded model. | Scope: private helper. |
| `_load_model_context (line 364)` | model_name: str | tuple[dict[str, Any], float] | Load one DeepDive model context and measure its loading duration. | Scope: private helper. Conventions: applies `to_deepdive_model_name`. |
| `_release_model_context (line 390)` | model_context: dict[str, Any] \| None | None | Clear one model context and release unused CPU and CUDA memory. | Scope: private helper. |
| `RidgeModelArtifacts (line 413)` | See class definition | RidgeModelArtifacts instance | Store one model's validated layer and ridge-alpha artifact. | Scope: class. Conventions: prediction mode. |
| `_model_artifact_path (line 432)` | artifact_config: Mapping[str, Any], model_name: str, artifact_label: str | Path | Resolve and require one model-specific configured artifact path. | Scope: private helper. |
| `_validate_model_artifacts (line 468)` | model_names: list[str], config: Mapping[str, Any], layer_results_path: Path, expected_target_dim: int | dict[str, RidgeModelArtifacts] | Resolve and validate every required layer and per-PC alpha artifact before extraction. | Scope: private helper. Conventions: prediction mode; alphas must match the target PC count and layer. |
| `_prepare_ridge_prediction (line 508)` | model_name: str, *, artifacts_by_model: Mapping[str, RidgeModelArtifacts], target_context: dict[str, Any], train_target: np.ndarray, shared_mask: np.ndarray, feature_extraction: Callable[..., Any], feature_extraction_kwargs: Mapping[str, Any], seed: int | tuple[np.ndarray, int] | Prediction-mode source preparer: extract raw best-layer features, fit ridge on training rows, predict shared rows. | Scope: private helper. Related: `ridge_predict_shared`. Conventions: returns (n_shared, n_target_pcs); no feature standardization. |
| `prepare_prediction_inputs (line 569)` | raw_target: np.ndarray, target_context: dict[str, Any], config: Mapping[str, Any], required_models: list[str], layer_results_path: Path, feature_extraction: Callable[..., Any], pc_target_path: Path, seed: int, *, target_preparation: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | tuple[np.ndarray, SourcePreparer] | Project and split the neural target, validate saved per-PC ridge artifacts, and bind a reusable held-out prediction preparer. | Scope: reusable/public. Related: `feature_manipulations.prepare_target`, `_prepare_ridge_prediction`. Conventions: target PCA is fitted outside held-out shared images; returns target and predictions in the same PC space. |
| `_fit_unique_project_shared (line 626)` | data: np.ndarray, shared_mask: np.ndarray, n_components: int, random_state: int | np.ndarray | Fit randomized PCA on unique-image rows and project shared-image rows. | Scope: private helper. Conventions: used for both target and sources in representation mode; returns float64 (n_shared, n_components). |
| `_extract_layer_features (line 660)` | model_context: dict[str, Any], layer_name: str, target_context: Mapping[str, Any], feature_extraction_kwargs: Mapping[str, Any] | np.ndarray | Extract one layer's features for every image in batches, with optional sparse random projection. | Scope: private helper. Conventions: returns float64 (n_images, n_features). |
| `_prepare_representation_projection (line 727)` | model_name: str, *, target_context: dict[str, Any], layer_results_path: Path, feature_extraction_kwargs: Mapping[str, Any], n_components: int, random_state: int, expected_shape: tuple[int, int] | tuple[np.ndarray, int] | Representation-mode source preparer: extract best-layer features and project them with unique-image PCA. | Scope: private helper. Conventions: projection must match the target shape. |
| `iter_prepared_pairs (line 788)` | pairs: list[tuple[str, str]], prepare_source: SourcePreparer, *, prefetch_next_model: bool | Iterator[tuple[str, str, np.ndarray, int, np.ndarray, int]] | Yield PID-ready pairs, preparing each model once and optionally the next model in one background thread. | Scope: reusable/public. Conventions: works with prediction and representation source preparers. |

## `pipeline/analysis/pca/pc_function.py`

File description: Calculate PID and mutual information as target PCs are accumulated.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `_save_pair_results (line 32)` | pair_results: dict[int, dict[str, Any]], model_1: str, model_2: str, results_dir: str \\| Path | Path | Save the PC-dependent PID and MI results for one model pair. | Scope: private helper. Conventions: verify PID definition, source/target order, and bit units; verify PCA fitting split and component convention. |
| `pc_function_analysis (line 60)` | config: dict[str, Any], functions: PIDPipelineFunctions, model1_name: list[str], model2_name: list[str], pc_path: str \\| Path, hdf_path: str \\| Path, pkl_info_path: str \\| Path, neural_data_path: str \\| Path, results_dir: str \\| Path \\| None=None, plot_dir: str \\| Path \\| None=None | dict[str, dict[str, dict[int, dict[str, Any]]]] | Calculate PID and MI for each model pair and cumulative target-PC count from full held-out predictions prepared once per model. | Scope: reusable/public. Related: `prepare_prediction_inputs`, `iter_prepared_pairs`. Conventions: PID input order (target, X1, X2); saved PCA and per-PC alphas must match; each PC step slices the same full predictions. |
| `main (line 197)` |  | None | Load the PC-function YAML, run all model pairs, and save their plots. | Scope: entry point. Conventions: verify PCA fitting split and component convention. |

## `pipeline/analysis/pca/pca_as_function.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `pca_as_function (line 30)` | pipeline_config: str \\| Path, pca_config: str \\| Path | dict[str, Any] | Run the full-OTC PID experiment from the YAML config beside this file. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify PCA fitting split and component convention. |
| `plot_ (line 63)` | results_dict: dict[str, Any], pca_config: str, pipeline_config: str | None | Plot PID versus PCA component count and save it as PDF. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify PCA fitting split and component convention. |

## `pipeline/analysis/pca/permutation.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `permuatation_analysis (line 22)` | pipeline_config: dict, permuation_config: dict | dict[str, Any] | Perform permutation analysis for the given pipeline and PCA configuration. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |

## `pipeline/analysis/pca/plot_pc_results.py`

File description: Plot PID and mutual information as target PCs are accumulated.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `plot_pc_results_from_pickle (line 26)` | pkl_path: str \\| Path, model_name: str, output_dir: str \\| Path, name: str \\| None=None | tuple[Path, Path] | Load one model pair's PC results and save its PDF plots. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `plot_pid_mi_as_function_of_pcs (line 73)` | pair_results: dict[int, dict[str, Any]], model_1_name: str, model_2_name: str, output_dir: str \\| Path, name: str \\| None=None | tuple[Path, Path] | Plot absolute and trivariate-MI-normalized results for one model pair as PDFs. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify random-variable order and logarithm base; verify PCA fitting split and component convention. |

## `pipeline/analysis/predictivity/pc_index_predictions.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `each_pc_index_pred (line 37)` | model_name: str, n_pcs: list[int], pc_path: str \\| Path, hdf_path: str \\| Path, pkl_info_path: str \\| Path, neural_data_path: str \\| Path | tuple[np.ndarray, int] | Predict individual target PCs from one model's selected layer. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `save_correlations_to_csv (line 110)` | correlations: np.ndarray, model_name: str, layer_index: int, pc_indexes: list[int], output_path: str \\| Path | Path | Append one completed model's per-PC correlations to a checkpoint CSV. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `main (line 149)` |  | None | Run every best-layer model and resume from a per-model CSV checkpoint. | Scope: entry point. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/analysis/predictivity/plot_pc_correlations.py`

File description: Plot per-PC ridge correlations for selected models on one figure.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `ModelCorrelationSeries (line 19)` | See class definition | ModelCorrelationSeries instance | Store one model's layer and its correlations at individual PC indexes. | Scope: class. Conventions: inspect implementation assumptions before reuse. |
| `load_pc_correlation_series (line 28)` | csv_paths: Sequence[str \\| Path], model_names: Sequence[str] | list[ModelCorrelationSeries] | Load selected models from CSV files produced by pcIndex_predictions.py. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `plot_pc_correlations (line 162)` | csv_paths: Sequence[str \\| Path], model_names: Sequence[str], output_path: str \\| Path, *, title: str='Ridge prediction correlation by PC index', dpi: int=300 | Path | Plot selected models' per-PC correlations together and save the PDF. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `main (line 265)` |  | Path | Run a three-model example using the alpha-search correlation CSV. | Scope: entry point. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/analysis/utils.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `_prepare_source_for_pid (line 14)` | source: np.ndarray, train_target: np.ndarray, shared_mask: np.ndarray, ridge: bool | np.ndarray | Prepare one model source for PID on the held-out shared images. | Scope: private helper. Conventions: verify PID definition, source/target order, and bit units; verify scaling, alpha, and train/test conventions. |
| `to_deepdive_model_name (line 57)` | model_name: str | str | Convert a stored model alias to the canonical name expected by DeepDive. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/examples/pid_pipeline_toy_example.py`

File description: Tiny no-data-loading example for debugging the strict PID pipeline flow.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `toy_target_extraction (line 17)` |  | dict[str, Any] | Create tiny fake target data. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `toy_sources_extraction (line 30)` | model_1: str, model_2: str | dict[str, dict[str, Any]] | Create tiny fake source contexts with two layers per source. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `toy_choose_layer (line 61)` | sources: dict[str, dict[str, Any]], layer_index: int=0 | dict[str, str] | Choose one layer for each source by index. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `toy_feature_extraction (line 78)` | source_context: dict[str, Any], layer_name: str, target_context: dict[str, Any] | list[list[float]] | Read fake features for one selected source layer. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `toy_preprocess (line 98)` | target: list[list[float]], source_1: list[list[float]], source_2: list[list[float]], scale: float=1.0 | tuple[list[list[float]], list[list[float]], list[list[float]]] | Scale target and source values together as a visible preprocessing step. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `toy_feature_manipulation (line 123)` | source_1: list[list[float]], source_2: list[list[float]], keep_columns: int=1 | tuple[list[list[float]], list[list[float]]] | Keep the first columns from both source feature matrices. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `toy_pid_calculation (line 145)` | target: list[list[float]], source_1: list[list[float]], source_2: list[list[float]], method_name: str='toy_pid' | dict[str, Any] | Return a readable dummy PID result from tiny arrays. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `toy_pid_report (line 177)` | pid_results: dict[str, Any], context: dict[str, Any] | str | Print a compact toy pipeline report. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `main (line 198)` |  | dict[str, Any] | Run the tiny toy PID pipeline. | Scope: entry point. Conventions: verify PID definition, source/target order, and bit units. |

## `pipeline/examples/voxel_experiment_smoke_example.py`

File description: No-data smoke example for the config-driven voxel experiment runner.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `smoke_voxel_target (line 17)` | voxel_index: int, subj_id: str, n_images: int=3 | dict[str, Any] | Create a tiny fake voxel target context. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `smoke_sources (line 38)` | model_name_1: str, model_name_2: str | dict[str, dict[str, Any]] | Create tiny fake source contexts. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `smoke_choose_layer (line 69)` | sources: dict[str, dict[str, Any]], X1_index: int, X2_index: int | dict[str, int] | Choose fake layer indexes from config. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `smoke_feature_extraction (line 85)` | source_context: dict[str, Any], layer_index: int, target_context: dict[str, Any], feature_shift: float=0.0 | list[list[float]] | Read fake source features for one selected layer. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `smoke_feature_manipulation (line 111)` | source_1: list[list[float]], source_2: list[list[float]], keep_columns: int=1 | tuple[list[list[float]], list[list[float]]] | Keep the first columns from both fake source matrices. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `smoke_pid_calculation (line 133)` | target: list[list[float]], source_1: list[list[float]], source_2: list[list[float]], method: str='smoke_pid' | dict[str, Any] | Return a tiny deterministic PID-like result. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `smoke_report (line 167)` | pid_results: dict[str, Any], context: dict[str, Any] | str | Print a compact smoke run report. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `register_smoke_functions (line 188)` |  | None | Register smoke wrapper functions for this example run. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `smoke_config (line 211)` |  | dict[str, Any] | Create a small YAML-shaped config for run_voxel_experiment. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `main (line 242)` |  | dict[str, Any] | Run the voxel experiment smoke example. | Scope: entry point. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/experiments/otc.py`

File description: Config-driven full-OTC experiment runner built on PIDPipeline.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `run_otc_experiment (line 26)` | config: dict[str, Any] | dict[str, Any] | Run one full-OTC experiment from an already-loaded config dictionary. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `_print_workflow (line 52)` | config: dict[str, Any], function_registry: dict[str, PipelineFunction], started_at: datetime | None | Print the configured OTC function order, inputs, kwargs, and start time. | Scope: private helper. Conventions: verify PID definition, source/target order, and bit units. |
| `nsd_otc_target (line 129)` | hdf_path: str \\| Path, pkl_info_path: str \\| Path, neural_data_path: str \\| Path, subj_id: str \\| None=None, voxel_index: int \\| None=None, n_images: int \\| None=None | dict[str, Any] | Load the full OTC target matrix and expose it under the PIDPipeline target key. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `_validate_config (line 170)` | config: dict[str, Any] | None | Validate the config sections needed to call the full-OTC runner. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/experiments/run_otc.py`

File description: Run the full-OTC PID experiment from the YAML config beside this file.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `make_str_as_path (line 17)` | config: dict[str, any] | dict[str, any] | Convert string paths in the config to Path objects. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `check_path_exists (line 26)` | config: dict[str, any] | None | Check if the paths in the config exist. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/experiments/run_voxel.py`

File description: No module docstring; inspect before reuse.

No functions or classes defined in this file.

## `pipeline/experiments/voxel.py`

File description: Config-driven voxel experiment runner built on PIDPipeline.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `run_voxel_experiment (line 23)` | config: dict[str, Any] | dict[str, Any] | Run one voxel experiment from an already-loaded config dictionary. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `nsd_voxel_target (line 43)` | voxel_index: int, subj_id: str, hdf_path: str \\| Path, pkl_info_path: str \\| Path, neural_data_path: str \\| Path, n_images: int \\| None=None | dict[str, Any] | Load one voxel target and expose it under the PIDPipeline target key. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `_choose_layer_kwargs (line 84)` | config: dict[str, Any] | dict[str, Any] | Arrange voxel-specific kwargs for the configured choose_layer function. | Scope: private helper. Conventions: verify PID definition, source/target order, and bit units. |
| `_validate_config (line 100)` | config: dict[str, Any] | None | Validate the config sections needed to call the voxel runner. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/pca/calculate_mean_mse.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `calculate_mean_mse (line 8)` | csv_path: Path, max_pcs: int, column_name: str='press_', n_features: int=8088 | dict[int, float] | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `report_top_minimal_mean (line 20)` | mean_mse_dict: dict[int, float], top_n: int=5 | None | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: verify random-variable order and logarithm base; verify PCA fitting split and component convention. |

## `pipeline/pca/find_alpha.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `find_alpha_per_pc (line 28)` | predictor: np.ndarray, target: np.ndarray, device: str \\| None='cuda' | tuple[np.ndarray, Pipeline] | Find one ridge alpha per target PC. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `load_and_apply_pca (line 115)` | data: np.ndarray, pca_path: str \\| Path, n_pcs: int \\| None=None | np.ndarray | Transform raw neural data with a saved centered PCA model. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `split_alphas_csv_by_model (line 148)` | alphas_csv_path: str \\| Path, output_dir: str \\| Path \\| None=None | list[Path] | Split an aggregate per-PC alpha CSV into one NumPy file per model. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `main (line 270)` | source_name: str, path_to_results: str \\| Path, pc_path: str \\| Path, hdf_path: str \\| Path, pkl_info_path: str \\| Path, neural_data_path: str \\| Path, alphas_csv_path: str \\| Path, n_pcs: int \\| None=None | tuple[np.ndarray, Pipeline] | Find raw-feature ridge alphas, save the unchanged alpha CSV plus a separate per-PC CV-diagnostics CSV, and print CV MSE/R² and alpha-boundary warnings. | Scope: entry point. The sidecar path is `<alphas_csv_stem>_diagnostics.csv`; CV R² is normalized by each training target PC's variance, and the aggregate is variance-weighted. Conventions: verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |

## `pipeline/pca/missmda_ncp.py`

File description: Minimal in-process Python wrapper for R's ``missMDA::estim_ncpPCA``.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `estimate_ncp_pca (line 22)` | data, ncp_min=0, ncp_max=5, method='Regularized', scale=True, method_cv='gcv', nbsim=100, p_na=0.05, threshold=0.0001, seed=None, rscript=RSCRIPT, verbose=False | Unannotated | Call ``missMDA::estim_ncpPCA`` on a Python sample table. | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention. |

## `pipeline/pca/choosing_minimal_best_pca/choosing_minimal_best_pca.py`

File description: Load repeated NSD OTC responses and run a configurable target-PC count selector.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `raise_on_nonfinite_images (line 47)` | responses: np.ndarray, image_ids: np.ndarray, repeat_ids: np.ndarray, subj_id: str, image_set: str, report_dir: str \| Path=DEFAULT_NONFINITE_DIR | None | Report every image/repeat pair containing a non-finite response and stop the run. | Scope: task-specific/public. Conventions: responses use `(image, voxel, repeat)` order; writes a diagnostic CSV only on failure. |
| `load_repeat_responses (line 108)` | subj_id: str, image_set: str='unique', pkl_info_path: str \| Path=DEFAULT_PKL_INFO_PATH, neural_data_dir: str \| Path=DEFAULT_NEURAL_DATA_DIR, nonfinite_report_dir: str \| Path=DEFAULT_NONFINITE_DIR | dict[str, Any] | Load finite repeated OTC responses and optionally exclude shared1000 images. | Scope: task-specific/public. Conventions: output responses use `(image, voxel, repeat)` order; `unique` excludes shared1000. |
| `selection_results_to_table (line 195)` | results: dict[str, Any], subj_id: str | pd.DataFrame | Convert scalar, per-candidate, and per-fold selection outputs into a candidate-level table. | Scope: task-specific/public. Conventions: requires `selected_n_components` and one-dimensional `candidate_n_components`; unsupported result shapes raise. |
| `choose_minimal_pcs (line 249)` | subj_id: str, selection_function: Callable[..., dict[str, Any]], image_set: str='unique', selection_kwargs: dict[str, Any] \| None=None, neural_data_dir: str \| Path=DEFAULT_NEURAL_DATA_DIR, pkl_info_path: str \| Path=DEFAULT_PKL_INFO_PATH, output_dir: str \| Path \| None=DEFAULT_OUTPUT_DIR, nonfinite_report_dir: str \| Path=DEFAULT_NONFINITE_DIR | dict[str, Any] | Load repeated responses, run a component selector, tabulate its outputs, and optionally save them. | Scope: task-specific/public. Conventions: passes responses unchanged in `(image, voxel, repeat)` order; when `selection_kwargs` contains `seed`, the output filename ends in `_seed{seed}.csv`. |

## `pipeline/pca/choosing_minimal_best_pca/save_gsn_pca_transform.py`

File description: Fit GSN on the selection CSV's image set and save only a pipeline-compatible signal-PCA transform.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `selection_from_csv (line 43)` | selection_csv_path: str \| Path, subj_id: str | tuple[int, str] | Read and validate one subject's error-minimizing component count and image set from a selection CSV. | Scope: task-specific/public. Conventions: reads `best_n_components`, not the one-standard-error `selected_n_components`; requires one supported `image_set`. |
| `save_gsn_pca_transform (line 102)` | subj_id: str, selection_csv_path: str \| Path, output_dir: str \| Path, n_components: int \| None=None, neural_data_dir: str \| Path=DEFAULT_NEURAL_DATA_DIR, pkl_info_path: str \| Path=DEFAULT_PKL_INFO_PATH | Path | Fit GSN on the CSV's image set and save a centered transform using the leading signal-covariance eigenvectors. | Scope: task-specific/public. Conventions: `n_components` overrides the CSV count; artifact records `image_set_` and `selection_csv_path_`; no responses, covariances, or projected observations are saved. |

## `pipeline/pca/cross_decomposition/cross_decomposition.py`

File description: Reusable training-only cross-covariance SVD and independent-repeat reconstruction selection with the existing GSN evaluation conventions.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `fit_cross_decomposition (line 15)` | first: np.ndarray, second: np.ndarray, n_components: int, seed: int=0, solver: str='auto' | dict[str, Any] | Fit descending left/right singular directions and training means. | Scope: reusable/public. Arrays `(n_train,d)`; separate group means; raw units, no scaling; C=A.T B/n_train. Matrix-free iterative or reduced-QR exact SVD; K cannot exceed min(d,n_train-1). Related: `select_cross_decomposition_components`. |
| `fit_cross_decomposition.multiply (line 61)` | v: np.ndarray | np.ndarray | Apply cross-covariance without constructing it. | Scope: nested operator helper. `(d,)` or `(d,q)` input/output; training data only. |
| `fit_cross_decomposition.transpose_multiply (line 66)` | v: np.ndarray | np.ndarray | Apply transposed cross-covariance without constructing it. | Scope: nested operator helper. `(d,)` or `(d,q)` input/output. |
| `normalized_reconstruction_curve (line 102)` | observed: np.ndarray, target: np.ndarray, mean: np.ndarray, basis: np.ndarray | np.ndarray | Compute normalized independent-repeat MSE for K=0..basis width via orthogonal SSE identity. | Scope: reusable/public. Inputs `(n,d)`, `(n,d)`, `(d,)`, orthonormal `(d,K)`; output `(K+1,)`; denominator is target error about supplied training mean. |
| `validation_design (line 134)` | n_images: int, n_repeats: int, n_folds: int, seed: int, repeat_split_weighting: str | tuple[list[np.ndarray], list[tuple[tuple, tuple]], np.ndarray] | Mirror seeded GSN image folds and adjacent forward/reverse repetition groups. | Scope: task-specific/public. Related: `select_gsn_pca_components`; weighting equal or proportional to input repeat count; >=2 training images per fold. |
| `select_cross_decomposition_components (line 171)` | responses: np.ndarray, max_components: int \| None=None, n_folds: int=5, seed: int=0, repeat_split_weighting: str='input_repeat_count', solver: str='auto' | dict[str, Any] | Select dimensions using independent-repeat validation errors, returning both argmin and one-SE choices. | Scope: reusable/public. Input `(images,voxels,repeats)`; U/V per training repeat partition; pooled training reconstruction mean matches GSN. Candidate grid 1..K; rejects unsupported fold rank. Fold errors `(folds,K)` and directed errors `(folds*directions,K)` in fold-major order. Selection CV only, no final-test claim. |

## `pipeline/pca/cross_decomposition/compare_reconstruction.py`

File description: Standalone YAML-configured paired OTC reconstruction comparison without modifying the original GSN selector.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `save_comparison (line 31)` | results: dict[str, dict[str, Any]], output_dir: Path, subj_id: str, image_set: str | pd.DataFrame | Save method curves, fold variability, summary, paired cross-minus-GSN differences and PDF. | Scope: task-specific/public. Requires matched candidate/fold arrays; reuses `selection_results_to_table`; descriptive fold SE, not final-test uncertainty. |
| `run_comparison (line 115)` | config_path: str \| Path=DEFAULT_CONFIG_PATH | list[Path] | Load the source GSN YAML and OTC data, run both methods on identical inputs, save fresh timestamped outputs. | Scope: entry point. Reuses `load_repeat_responses`, `select_gsn_pca_components`; saves image folds, repeat groups and resolved settings; fixed seeds; no argparse; no projected responses or overwrite of prior GSN files. |

## `pipeline/pca/plotting.py`

File description: Plot subject-level PCA variance and pairwise unique information.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `plot_cumulative_unique_information (line 12)` | csv_path: str \\| Path, output_path: str \\| Path \\| None=None, *, model_1_column: str='model_1', model_2_column: str='model_2', unique_1_column: str='unq1', unique_2_column: str='unq2', dpi: int=300 | Path | Plot cumulative pairwise unique information for each model as PDF. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify PCA fitting split and component convention. |
| `plot_heldout_variance_explained (line 127)` | variance_csv_path: str \\| Path, output_path: str \\| Path \\| None=None, *, show_cumulative: bool=True, show_training: bool=False, plot_training_minus_heldout: bool=False, separate_panels: bool=False, number_of_pcs: int \\| None=None, dpi: int=300 | Path | Plot variance explained on held-out data by each retained PC as PDF. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |

## `pipeline/pca/subject_pca.py`

File description: Fit subject-level PCA models and evaluate them on shared held-out images.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `split_unique_shared (line 27)` | subj_id: str, hdf_path: str \\| Path, pkl_info_path: str \\| Path, neural_data_path: str \\| Path, variance_threshold: float=0.99 | dict[str, Any] | Load one subject and split its rows into unique and shared-image sets. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `pca_by_variance (line 90)` | neural_data: np.ndarray, variance_threshold: float=0.99 | dict[str, Any] | Fit centered PCA without per-feature variance standardization. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `heldout_pca (line 128)` | pca_model: PCA, heldout_data: np.ndarray | np.ndarray | Project raw held-out responses with a fitted centered PCA model. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `pca_func (line 154)` | data: np.ndarray, mode: str='eigenvector_CV', max_features: int \\| None=None | dict[str, Any] | Select a component count and fit centered, unstandardized PCA. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `main (line 222)` | subj_id: str, hdf_path: str \\| Path, pkl_info_path: str \\| Path, neural_data_path: str \\| Path, variance_threshold: float=0.99, save_models_path: str \\| Path \\| None=None, max_features: int \\| None=None, pca_mode: str='missmda_CV' | dict[str, Any] | Fit centered unstandardized PCA and evaluate shared held-out data. | Scope: entry point. Conventions: verify PCA fitting split and component convention. |

## `pipeline/phases/choosing_layer.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `random_layer_selection (line 9)` | n_layers | Unannotated | Choose a random layer index from the available layers. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `specific_index_layer_selection (line 23)` | layer_names, index | Unannotated | Choose a specific layer index from the available layers. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `voxel_best_layer (line 44)` | voxel_index: int=None, index_layer: int=None, path_to_results: str=None | dict | Choose the best model layer for one voxel, or a representative voxel for one layer. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `overall_best_layer (line 107)` | model_name: str, path_to_results: str | dict | Choose the overall best layer index for one model from an OTC CSV. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `_read_csv_rows (line 146)` | path_to_results: str | tuple[list[dict[str, str]], set[str]] | Read CSV rows and column names for layer-selection helpers. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `_normalize_csv_value (line 163)` | value | str | Normalize CSV values before exact lookup comparisons. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/phases/feature_manipulations.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `pca_projection (line 23)` | features, n_components | Unannotated | Apply PCA projection to reduce dimensionality of features. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `jl_projection (line 40)` | features, n_samples, eps=0.1, jl_dim=None | Unannotated | Apply Johnson-Lindenstrauss projection to reduce dimensionality of features. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `cca_projection (line 73)` | features1, features2, n_components | Unannotated | Apply Canonical Correlation Analysis (CCA) to find linear combinations of two sets of features that are maximally correlated. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `prepare_target (line 106)` | target: np.ndarray, target_context: Mapping[str, Any], pc_target_path: str \\| Path | tuple[np.ndarray, np.ndarray, np.ndarray] | Project a target with a saved PCA and split it by the shared mask. | Scope: reusable/public. Conventions: this feature-manipulation callable is distinct from `sources_target_features.prepare_target`; verify PCA fitting split, component, scaling, and train/test conventions. |
| `load_ridge_alphas (line 189)` | alphas_path: str \\| Path, *, model_name: str, expected_target_dim: int, expected_layer_index: int \\| None=None | np.ndarray | Load and validate a model's per-target-PC ridge penalties. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `ridge_predict_shared (line 314)` | source: np.ndarray, train_target: np.ndarray, shared_mask: np.ndarray, alphas: np.ndarray, *, seed: int | np.ndarray | Fit per-target ridge models and predict only the held-out rows. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `feature_manipulation_ridge (line 418)` | source1: np.ndarray, source2: np.ndarray, target: np.ndarray, target_context: Mapping[str, Any], seed: int, model_name_1: str, model_name_2: str, pc_target_path: str \\| Path, alphas_source1_path: str \\| Path, alphas_source2_path: str \\| Path | tuple[np.ndarray, np.ndarray, np.ndarray] | Create held-out ridge predictions for two sources against PCA(T). | Scope: reusable/public. Conventions: verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `pca_source (line 502)` | source, shared_mask, max_features | Unannotated | Train PCA on source features and return the held-out source features projected onto the PCA space. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |

## `pipeline/phases/preprocessing_layer.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `permute_rv (line 22)` | source1, source2, target, source1_perm=False, source2_perm=False, target_perm=False, rng_seed=56 | Unannotated | Permute the random variable rv according to the configuration provided in config. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `_to_numpy (line 56)` | x | Unannotated | Convert torch.Tensor to NumPy only if needed. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `_from_numpy_like (line 63)` | x_np, reference | Unannotated | Return x_np as torch if reference is torch, otherwise NumPy. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `ridge_train_to_test_prediction (line 73)` | source_train, target_train, source_test, target_test=None, alphas=None, inner_cv: int=5, scoring: str='r2', shuffle: bool=True, random_state: int=0 | Unannotated | Fit ridge encoding model on training data and return held-out test predictions. | Scope: reusable/public. Conventions: verify scaling, alpha, and train/test conventions. |
| `apply_saved_scaler (line 218)` | data: np.ndarray, scaler_path: str \\| Path | np.ndarray | Transform an array with a previously fitted scaler. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `scale_func (line 262)` | source1: np.ndarray, source2: np.ndarray, target: np.ndarray, source1_scaler_path: str \\| Path, source2_scaler_path: str \\| Path, target_scaler_path: str \\| Path | tuple[np.ndarray, np.ndarray, np.ndarray] | Scale two sources and a target with their saved fitted scalers. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/phases/sources_target_features.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `prepare_sources (line 21)` | model_name_1: str, model_name_2: str | dict[str, dict] | Prepare sources for feature extraction. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `prepare_target (line 42)` | hdf_path: Path, pkl_info_path: Path, neural_data_path: Path, n_samples: int=None | dict | Prepare target for feature extraction. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `shared1000_subj_target (line 79)` | hdf_path: str \\| Path, pkl_info_path: str \\| Path, neural_data_path: str \\| Path, pca_model_path: str \\| Path \\| None=None, scaler_model_path: str \\| Path \\| None=None | dict[str, np.ndarray] | Load a subject's shared 1,000 stimuli and aligned neural responses. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `prepare_target_for_voxel (line 215)` | voxel_index: int, subj_id: str, hdf_path: Path, pkl_info_path: Path, neural_data_path: Path | dict | Prepare target for feature extraction for a specific voxel. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `make_nsd_dataloader (line 241)` | model_context: dict, stim_dataset, image_ids: np.ndarray, batch_size: int | DataLoader | Create a DataLoader for an ordered subset of NSD images. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `batching (line 264)` | model_context: dict, batch_start: int, batch_end: int, stim_dataset, subj_image_ids: np.ndarray, layer_name: str, batch_size_dataloader: int | np.ndarray | Batch process a range of images for feature extraction. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `feature_extraction (line 295)` | layer_index: int, model_context: dict, subj_image_ids: np.ndarray, stim_dataset, batch_size_process: int, batch_size_dataloader: int=128 | np.ndarray | Extract features from the models and the neural data. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/testing/gsn/gsn_pca.py`

File description: Estimate GSN covariances and eigendecompose them for PCA diagnostics.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `estimate_gsn_covariances (line 17)` | responses: np.ndarray | dict[str, Any] | Estimate optimized signal and noise covariances with official GSN. | Scope: task-specific/public. Related: `pca_signal_noise_covariances`. Conventions: input order `(image, voxel, repeat)`; GSN receives `(voxel, image, repeat)`; returns optimized raw covariances `cSb` and `cNb`; no scaling or DNN/PID dependency. |
| `pca_signal_noise_covariances (line 50)` | signal_covariance: np.ndarray, noise_covariance: np.ndarray | dict[str, np.ndarray] | Eigendecompose optimized GSN signal and noise covariance matrices. | Scope: task-specific/public. Related: `estimate_gsn_covariances`. Conventions: inputs are raw covariance matrices, not samples or correlation matrices; eigenvectors are columns and eigenvalues are descending. |
| `select_gsn_pca_components (line 113)` | responses: np.ndarray, max_components: int \| None=None, n_folds: int=5, seed: int=0, repeat_split_weighting: str='input_repeat_count' | dict[str, Any] | Select the smallest GSN signal-PC count within one standard error of the best independent-repeat reconstruction error. | Scope: task-specific/public. Related: `estimate_gsn_covariances`, `pca_signal_noise_covariances`. Conventions: GSN is fit on training-image folds; `repeat_split_weighting='input_repeat_count'` gives two-repeat inputs total weight 2/3 and one-repeat inputs 1/3 when three repeats are available; `'equal'` gives directed splits equal weight; no DNN/PID dependency. |

## `pipeline/testing/gsn/gsn_pca_smoke.py`

File description: Debuggable smoke example for GSN PCA with known signal rank.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `run_gsn_pca_smoke (line 20)` | n_images: int=500, n_voxels: int=50, true_rank: int=5, n_repeats: int=4, seed: int=0 | dict[str, Any] | Run GSN PCA on repeated responses with known low-rank signal. | Scope: smoke/debug entry point. Related: `estimate_gsn_covariances`, `pca_signal_noise_covariances`, `select_gsn_pca_components`. Conventions: responses are `(image, voxel, repeat)`; prints only the cross-validated number of retained signal eigenvalues; reports selected count, error-minimizing count, eigengap after the true rank, and subspace overlap; no PID, DNN, or real-data dependency. |

## `pipeline/pid_pipeline.py`

File description: Strict, readable orchestration class for one PID pipeline run.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `PIDPipelineFunctions (line 11)` | See class definition | PIDPipelineFunctions instance | Store the user-selected functions for one PID pipeline run. | Scope: class. Conventions: verify PID definition, source/target order, and bit units. |
| `PIDPipeline (line 43)` | self, functions: PIDPipelineFunctions | PIDPipeline instance | Run the PID pipeline by connecting the provided functions in order. | Scope: class. Conventions: verify PID definition, source/target order, and bit units. |
| `PIDPipeline.__init__ (line 46)` | self, functions: PIDPipelineFunctions | None | Create a strict PID pipeline orchestrator. | Scope: private helper. Conventions: verify PID definition, source/target order, and bit units. |
| `PIDPipeline.add_rng_to_kwargs (line 86)` | self, kwargs: dict[str, Any], func, rng: np.random.Generator | dict[str, Any] | Add the random number generator to the provided kwargs dictionary. | Scope: method/nested helper. Conventions: verify PID definition, source/target order, and bit units. |
| `PIDPipeline.run (line 108)` | self, *, target_kwargs: dict[str, Any] \\| None=None, sources_kwargs: dict[str, Any] \\| None=None, choose_layer_kwargs: dict[str, Any] \\| None=None, feature_extraction_kwargs: dict[str, Any] \\| None=None, preprocess_kwargs: dict[str, Any] \\| None=None, feature_manipulation_kwargs: dict[str, Any] \\| None=None, pid_kwargs: dict[str, Any] \\| None=None, report_kwargs: dict[str, Any] \\| None=None | dict[str, Any] | Run target, sources, layers, features, optional transforms, PID, and report. | Scope: method/nested helper. Conventions: verify PID definition, source/target order, and bit units. |

## `pipeline/plotting.py`

File description: Create matrix heatmaps from pairwise PID pipeline CSV results.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `plot_pairwise_pid_matrices (line 12)` | csv_path: str \\| Path, output_dir: str \\| Path, *, model_order: list[str] \\| None=None, value_format: str='.3f', cmap: str='viridis', figsize: tuple[float, float] \\| None=None, dpi: int=300 | dict[str, Path] | Create and save unique-information, redundancy, and synergy matrices as PDFs. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `comulative_sum (line 229)` | array: np.ndarray | np.ndarray | Return the cumulative sum of a 1D array, ignoring NaN values. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |

## `pipeline/utils.py`

File description: Utility helpers used by the PID pipeline orchestrator.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `choose_random_sources (line 19)` | sources_list: list[str], size: int=2, replace: bool=False | np.ndarray | Randomly select a source from the list of available sources. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `run_configured_pid_pipeline (line 35)` | config: dict[str, Any], function_registry: dict[str, PipelineFunction], choose_layer_kwargs: dict[str, Any] \\| None=None | dict[str, Any] | Run PIDPipeline from a config dictionary and function registry. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `pipeline_functions_from_config (line 68)` | function_config: dict[str, Any], function_registry: dict[str, PipelineFunction] | PIDPipelineFunctions | Resolve configured function names into PIDPipelineFunctions. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `resolve_pipeline_function (line 95)` | function_config: dict[str, Any], function_registry: dict[str, PipelineFunction], step_name: str, required: bool | PipelineFunction \| None | Resolve one configured pipeline step name from a registry. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units. |
| `validate_pipeline_config_sections (line 124)` | config: dict[str, Any], required_sections: tuple[str, ...] | None | Validate that required config sections exist. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `nsd_sources (line 140)` | model_name_1: str, model_name_2: str | dict[str, dict[str, Any]] | Load two model contexts and expose them under X1 and X2. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `specific_layer_index (line 157)` | sources: dict[str, dict[str, Any]], X1_index: int, X2_index: int | dict[str, int] | Select one configured layer index for each source. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `random_layer_selection_for_sources (line 181)` | sources: dict[str, dict[str, Any]], random_seed: int \\| None=None | dict[str, int] | Select a random valid layer index for each source. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `voxel_best_layer_for_sources (line 203)` | sources: dict[str, dict[str, Any]], X1_index: int, X2_index: int, X1_path_to_results: str \\| Path, X2_path_to_results: str \\| Path | dict[str, int] | Select each source model's best layer for one voxel index. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `overall_best_layer_for_sources (line 236)` | sources: dict[str, dict[str, Any]], path_to_results: str \\| Path | dict[str, int] | Select each source model's overall best OTC layer from one CSV file. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `nsd_feature_extraction (line 267)` | source_context: dict[str, Any], layer_index: int, target_context: dict[str, Any], batch_size_process: int=128, batch_size_dataloader: int=128 | Any | Extract features for one NSD source and selected layer. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `pca_each_source (line 298)` | source_1: Any, source_2: Any, target: Any, n_components_source_1: int, n_components_source_2: int, n_components_target: int | tuple[Any, Any] | Apply PCA separately to source_1 and source_2. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `pid_calc_adapter (line 336)` | target: Any, source_1: Any, source_2: Any, method: str='eigen_pid', config: dict[str, Any] \\| None=None, rng_seed: int=56, **pid_kwargs: Any | dict[str, Any] | Call pid_calc using the strict PIDPipeline array order. | Scope: reusable/public. Conventions: input order target, X1, X2; delegates to Eigen-PID; verify PID definition, source/target order, and bit units. |
| `print_pid_mi_adapter (line 385)` | pid_results: dict[str, Any], context: dict[str, Any], **report_kwargs: Any | Any | Print PID and MI outputs from pid_calc_adapter. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify random-variable order and logarithm base. |
| `print_pid_mi (line 402)` | pid_results, mi_result | Unannotated | This functions takes the pid results and the mutual information results and prints them in a nice format. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify random-variable order and logarithm base. |
| `_as_2d_tensor (line 412)` | value: Any | Any | Convert samples to a 2D torch tensor. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `_random_layer_index_for_source (line 433)` | sources: dict[str, dict[str, Any]], source_name: str, rng: np.random.Generator | int | Select one random layer index for one source context. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `_layer_index_values (line 463)` | sources: dict[str, dict[str, Any]], source_name: str, requested_index: int | list[int] | Create valid layer-index values for one source. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `_model_name_for_source (line 481)` | sources: dict[str, dict[str, Any]], source_name: str | str | Read a source model name from a source context. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `_overall_best_layer_model_names (line 498)` | path_to_results: str \\| Path | list[str] | Read model names from an overall best-layer CSV for diagnostics. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `source_context (line 520)` | sources: Any, source_name: str | Any | Read one source context from the sources object. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `choose_one_layer (line 536)` | layer_func: Callable[..., Any], source_context_value: Any, layer_kwargs: dict[str, Any] | Any | Choose one layer by adapting to the common layer-selection helper signatures. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `ridge_preprocessing (line 565)` | source_1: Any, source_2: Any, target: Any, rng, test_size, **preprocess_kwargs: Any | tuple[Any, Any, Any] | Apply ridge regression preprocessing to sources and target. | Scope: reusable/public. Conventions: verify scaling, alpha, and train/test conventions. |

## `simulations/pca/rank/eigenvector_pca.py`

File description: Numerical eigenvector PCA cross-validation and loading estimators.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `EigenvectorPCACVResult (line 41)` | See class definition | EigenvectorPCACVResult instance | No docstring; inspect implementation before reuse. | Scope: class. Conventions: verify PCA fitting split and component convention. |
| `fit_pca_loadings_svd (line 51)` | X_train: np.ndarray, n_components: int | np.ndarray | Fit PCA loadings with SVD. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `_eigenvector_pca_cv_sample_press (line 65)` | X: np.ndarray, sample_index: int, max_components: int, pca_fit_fn: PCAFitFunction, center: bool, scale: bool, include_zero_components: bool, method_pca: str \\| None, eps: float | np.ndarray | Calculate one held-out sample's PRESS contribution. | Scope: private helper. Conventions: verify PCA fitting split and component convention. |
| `eigenvector_pca_cv (line 149)` | X: np.ndarray, max_components: int \\| None=None, pca_fit_fn: PCAFitFunction \\| None=None, center: bool=True, scale: bool=False, include_zero_components: bool=True, method_pca: str \\| None=None, eps: float=1e-12, checkpoint_csv_path: str \\| Path \\| None=None | EigenvectorPCACVResult | Run eigenvector cross-validation for PCA component selection. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `fit_pca_loadings_sklearn (line 265)` | X_train: np.ndarray, n_components: int | np.ndarray | Fit PCA loadings with sklearn. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `regular_PCA (line 285)` | X: np.ndarray, variance_threshold: float | np.ndarray | Fit sklearn PCA using a variance threshold. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |

## `simulations/pca/rank/eigenvector_pca_wrapper.py`

File description: Checkpoint persistence helpers for eigenvector PCA cross-validation.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `_checkpoint_fields (line 31)` | max_components: int | tuple[list[str], list[str]] | Build the checkpoint CSV columns. | Scope: private helper. Conventions: verify PCA fitting split and component convention. |
| `build_checkpoint_metadata (line 45)` | X: np.ndarray, max_components: int, pca_fit_fn: PCAFitFunction, center: bool, scale: bool, include_zero_components: bool, method_pca: str \\| None, eps: float | dict[str, str] | Build strict metadata used to validate a resumable checkpoint. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `load_eigenvector_pca_checkpoint (line 99)` | checkpoint_csv_path: str \\| Path, metadata: dict[str, str], max_components: int | dict[int, np.ndarray] | Load a compatible eigenvector PCA checkpoint. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `write_eigenvector_pca_checkpoint (line 161)` | checkpoint_csv_path: str \\| Path, metadata: dict[str, str], completed_press: dict[int, np.ndarray], max_components: int | None | Atomically write completed eigenvector PCA checkpoint rows. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |

## `simulations/pca/rank/pca_simulation.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `create_T_and_P (line 30)` | n_samples: int, n_features: int, rank: int, loading_corr: float=0.0, component_strengths: np.ndarray \\| list[float] \\| None=None, random_state: int \\| None=None | tuple[np.ndarray, np.ndarray] | Create scores and correlated loadings. Inputs define sizes, rank, correlation, strengths, and seed; outputs are T and P arrays. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `generate_rank_simulation_data (line 69)` | n_samples: int, n_features: int, rank: int, loading_corr: float, noise_std: float, random_state: int, component_strengths: list[float] \\| np.ndarray \\| None=None, center_columns: bool=True | dict | Generate noisy known-rank data. Inputs define the condition; output contains X, T, P, normalized signal, and metadata. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |
| `run_rank_simulation (line 95)` | grid: dict[str, list], output_dir: str \\| Path=OUTPUT_DIR, nbsim: int=NBSIM | tuple[pd.DataFrame, pd.DataFrame] | Run EM/K-fold rank selection and save raw/summary CSVs plus a PDF heatmap. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |

## `simulations/pca/rank/rowwise_pca.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `RowwiseLOOVariancePCAResult (line 10)` | See class definition | RowwiseLOOVariancePCAResult instance | No docstring; inspect implementation before reuse. | Scope: class. Conventions: verify PCA fitting split and component convention. |
| `rowwise_loo_pca_variance_threshold (line 23)` | X: np.ndarray, variance_threshold: float=0.99 | RowwiseLOOVariancePCAResult | Row-wise leave-one-out PCA where the number of PCs is chosen by | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |

## `simulations/pca/rank/run_simulation.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `run_simulation (line 13)` | n_samples: int, n_features: int, rank: int, loading_corr: float, noise_std: float, random_state: int | dict | Run PCA simulations with eigenvector and row-wise LOOCV methods. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention. |

## `simulations/pca/ridge/pid_feature_middleware.py`

File description: Shared PCA, Ridge-CV, theoretical-PID, and trial-loop helpers.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `pca_target (line 34)` | target: Any, shared_mask: np.ndarray, n_components_target: int, random_state: int=0 | tuple[np.ndarray, np.ndarray] | Fit target PCA and return its scores and population linear map. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `pca_sources (line 64)` | source_1: Any, source_2: Any, target: Any, shared_mask: np.ndarray, n_components_source_1: int, n_components_source_2: int, random_state: int=0 | tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | Fit source PCAs and return held-out arrays and population linear maps. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `_ridge_prediction_and_map (line 109)` | source_train: np.ndarray, target_train: np.ndarray, source_test: np.ndarray, alphas: np.ndarray | tuple[np.ndarray, np.ndarray] | Return native per-target-alpha RidgeCV predictions and RAW-input map. | Scope: private helper. Conventions: verify PID definition, source/target order, and bit units; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `ridge_sources_on_target (line 158)` | source_1: Any, source_2: Any, target: Any, shared_mask: np.ndarray, alphas: np.ndarray | tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | Fit per-target-PC Ridge-CV models and return arrays and linear maps. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `expand_independent_covariance (line 196)` | coordinate_covariance: torch.Tensor, n_replicas: int | torch.Tensor | Expand independent coordinate covariance into grouped variable order. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `transform_population_covariance (line 220)` | covariance: torch.Tensor, linear_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | torch.Tensor | Propagate [X1, X2, T] covariance through three fitted linear maps. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `standardize_covariance (line 247)` | covariance: torch.Tensor | torch.Tensor | Convert a covariance to a correlation matrix without changing PID. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `calculate_theoretical_pid (line 272)` | covariance: torch.Tensor, dims: list[int], method: str | dict[str, float] | Calculate exact Gaussian PID and MI values from a population covariance. | Scope: reusable/public. Related: `gaussian_mutual_information`, `pid_calc`. Conventions: population covariance order [X1, X2, T]; outputs bits; verify PCA fitting split and component convention. |
| `prepare_pid_routes (line 326)` | source_1: Any, source_2: Any, target: Any, shared_mask: np.ndarray, n_components: int, seed: int, population_covariance: torch.Tensor, population_dims: list[int] | dict[str, tuple[tuple[np.ndarray, np.ndarray, np.ndarray], torch.Tensor]] | Prepare held-out PID arrays and exact route population covariances. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `run_pid_feature_comparison (line 386)` | sample_for_seed: Callable[[int], tuple[Any, Any, Any]], population_covariance: torch.Tensor, population_dims: list[int], *, n_samples: int, n_train: int, n_components: int, n_trials: int, base_seed: int, pid_method: str, bias_correction: bool, experiment_name: str, plot_path: str \\| Path, plot_title: str, metadata: dict[str, Any] \\| None=None | dict[str, dict[str, dict[str, float]]] | Run a seeded RAW/PCA/Ridge-CV PID comparison and save its table plot as PDF. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `build_sonic_covariance (line 518)` | p: int | torch.Tensor | Build the full Sonic covariance in grouped [X1, X2, T] order. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |

## `simulations/pca/ridge/run_pca_ridge.py`

File description: Run any registered PCA–Ridge PID comparison through one command-line entry point.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `parse_args (line 18)` |  | argparse.Namespace | Parse common PCA–Ridge simulation arguments. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `run_scenario (line 47)` | scenario_name: str, *, n_samples: int=10000, n_train: int=9000, p: int=70, noise_std: float=1.0, n_components: int \\| None=None, n_trials: int=2, base_seed: int=0, pid_method: str \\| None=None, bias_correction: bool=True, output_dir: Path \\| str=PROJECT_ROOT / 'simulations/pca/ridge/results' | dict[str, Any] | Run one registered PCA–Ridge scenario with shared orchestration. | Scope: reusable/public. Conventions: verify PID definition, source/target order, and bit units; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `main (line 127)` |  | dict[str, Any] | Run the scenario selected on the command line. | Scope: entry point. Conventions: verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |

## `simulations/pca/ridge/scenarios.py`

File description: Scenario definitions shared by the consolidated PCA–Ridge runner.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `PcaRidgeScenario (line 26)` | See class definition | PcaRidgeScenario instance | Describe one PCA–Ridge simulation without duplicating runner code. | Scope: class. Conventions: verify PID definition, source/target order, and bit units; verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `build_all_above_zero_covariance (line 57)` | p: int, noise_std: float, unique1_weight: float, unique2_weight: float, redundant_weight: float, shared_noise_weight: float | torch.Tensor | Build the all-above-zero covariance in grouped ``[X1, X2, T]`` order. | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `build_concatenated_all_above_zero_covariance (line 106)` | p: int, noise_std: float, redundant_weight: float, shared_noise_weight: float | torch.Tensor | Build the concatenated all-above-zero covariance. | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `build_concatenated_equal_unique_covariance (line 143)` | p: int, noise_std: float | torch.Tensor | Build the concatenated equal-unique covariance. | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `build_equal_unique_covariance (line 173)` | p: int, noise_std: float | torch.Tensor | Build the equal-unique covariance. | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `build_full_suppression_covariance (line 195)` | p: int, noise_std: float | torch.Tensor | Build the full-suppression covariance. | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `build_unq2_zero_covariance (line 218)` | p: int, noise_std: float | torch.Tensor | Build the zero-source-2-unique covariance. | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device; verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |
| `get_scenario (line 327)` | name: str | PcaRidgeScenario | Return one registered PCA–Ridge scenario. | Scope: reusable/public. Conventions: verify PCA fitting split and component convention; verify scaling, alpha, and train/test conventions. |

## `simulations/evil_twin/__init__.py`

File description: Sonic/Shadow evil-twin covariance examples.

No functions or classes defined in this file.

## `simulations/evil_twin/covariance_example.py`

File description: Torch Sonic/Shadow evil-twin covariance example.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `empirical_covariance_matrix_torch (line 6)` | X1, X2, T, correction=1 | Unannotated | Compute the empirical covariance matrix of (X1, X2, T). | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device. |
| `check_evil_twin_covariances_torch (line 24)` | data, atol=0.05, rtol=0.05, verbose=True | Unannotated | Compare the empirical covariance matrices of Sonic and Shadow. | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device. |
| `evil_twin_example_torch (line 70)` | generator, n, p, device='cpu', dtype=torch.float64 | Unannotated | Generate the Sonic and Shadow evil-twin Gaussian examples. | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device. |
| `evil_twin_example_torch.randn_scaled (line 88)` | var_total | Unannotated | No docstring; inspect implementation before reuse. | Scope: method/nested helper. Conventions: verify covariance order, whitening state, dtype, and device. |
| `run_covariance_comparison (line 126)` | n=1000, p=300, seed=0, device='cpu', dtype=torch.float64, atol=0.05, rtol=0.05, verbose=True | Unannotated | Generate Sonic/Shadow samples and compare their empirical covariances. | Scope: reusable/public. Conventions: verify covariance order, whitening state, dtype, and device. |

## `simulations/evil_twin/plot_results.py`

File description: Plot Sonic and Shadow PID atoms from an evil-twin results CSV.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `plot_pid_atoms (line 33)` | results_csv: Path \\| str, output_path: Path \\| str \\| None=None | Path | Plot mean redundancy, both unique-information atoms, and synergy for Sonic and Shadow as PDF. | Scope: reusable/public. Conventions: successful rows are averaged across seeds; values are Eigen-PID bits with sources ordered X1, X2 and target T. |
| `main (line 138)` | config: dict \\| None=None | Path | Plot the configured evil-twin results CSV. | Scope: entry point. Conventions: verify input CSV metadata and output path. |

## `simulations/evil_twin/run_from_config.py`

File description: Run the Sonic/Shadow Eigen-PID experiment from an editable configuration.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `validate_config (line 50)` | config: dict | None | Validate the editable evil-twin experiment configuration. | Scope: reusable/public. Conventions: requires enough samples for a full-rank joint empirical covariance. |
| `result_row (line 101)` | seed: int, twin: str, n: int, p: int, bias_correction: bool, pid: dict, mi: dict | dict | Create one successful detailed Eigen-PID result row. | Scope: task-specific. Conventions: PID and MI values are in bits; sources are X1, X2 and target is T. |
| `error_row (line 142)` | seed: int, twin: str, n: int, p: int, bias_correction: bool, error: Exception | dict | Create one failed evil-twin result row. | Scope: task-specific. Conventions: preserves experiment metadata and exception text. |
| `run_from_config (line 181)` | config: dict | dict | Run Eigen-PID for Sonic and Shadow, save one detailed CSV, and plot all PID atoms as PDF. | Scope: reusable/public. Related: `pid_calc`, `plot_pid_atoms`. Conventions: bias_correction selects raw or corrected Eigen-PID; sources are X1, X2 and target is T. |
| `parse_args (line 277)` |  | argparse.Namespace | Parse optional command-line overrides. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `config_from_args (line 306)` | args: argparse.Namespace | dict | Apply explicitly supplied command-line values to `CONFIG`. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `main (line 332)` |  | dict | Run the configured evil-twin experiment. | Scope: entry point. Conventions: verify PID definition, source/target order, and bit units. |

## `simulations/rsa_vs_pid/__init__.py`

File description: Gaussian experiment comparing Pearson-RDM RSA with BROJA unique information.

No functions or classes defined in this file.

## `simulations/rsa_vs_pid/run_rsa_vs_pid.py`

File description: Compare Pearson-RDM RSA and Gaussian BROJA unique information on sources that differ only along the all-ones direction.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `validate_config (line 81)` | config: dict | None | Validate the experiment configuration. | Scope: task-specific. Conventions: requires n >= 3d + 1 for every grid point (full-rank (3d, 3d) sample covariance). |
| `sample_shift_invariant_example (line 129)` | rng: np.random.Generator, n: int, d: int, noise_variance: float | tuple[np.ndarray, np.ndarray, np.ndarray] | Sample T = Z + M1 + eps_T, X = Z + M1 + eps_X, Y = Z + N1 + eps_Y with Z ~ N(0, I - 11^T/d). | Scope: task-specific. Conventions: returns samples ordered [X1, X2, T], each (n, d) float64; fixed draw order G, M, N, eps_T, eps_X, eps_Y. |
| `rsa_scores (line 165)` | target: np.ndarray, x: np.ndarray, y: np.ndarray | dict[str, float] | Spearman RSA between rsatoolbox correlation-distance RDMs of T and each source. | Scope: reusable/public. Related: rsatoolbox `calc_rdm(method="correlation")`, `compare(method="spearman")`. Conventions: rows are stimuli, columns are features. |
| `run_trial (line 201)` | dimension: int, n_samples: int, trial: int, seed: int, config: dict | dict | Run RSA and bias-corrected Gaussian BROJA PID on the same samples and validate shapes, covariance, finiteness, and PID identities. | Scope: task-specific. Related: `pid_calc`, `create_cov_matrix`. Conventions: PID order [T, X1=X, X2=Y]; outputs bits; unsuffixed atoms are `debias_eigen_pid` combined estimates, `_raw` atoms are uncorrected. |
| `summarize_trials (line 322)` | trials_csv: str \| Path, summary_csv: str \| Path | Path | Long-format per-(d, n) mean, median, SD, and 25th/75th percentiles across trials of every metric. | Scope: task-specific. Conventions: q25/q75 describe the trial spread (IQR), not a CI of the mean; `frac_positive` only for delta_ui columns. |
| `run_experiment (line 365)` | config: dict | dict[str, Path] | Run the (d, n, trial) grid, write the trial CSV incrementally, summarize, and plot. | Scope: entry point helper. Conventions: seed = base_seed + condition_index * n_trials + trial; relative output_dir resolves from the repository root. |
| `parse_args (line 416)` |  | argparse.Namespace | Parse optional command-line overrides. | Scope: entry point. |
| `config_from_args (line 435)` | args: argparse.Namespace | dict | Apply command-line overrides to a copy of `CONFIG`. | Scope: entry point. |
| `main (line 452)` |  | dict[str, Path] | Run the configured experiment. | Scope: entry point. |

## `simulations/rsa_vs_pid/plot_results.py`

File description: Plot RSA scores, BROJA PID atoms, and their paired differences from saved CSVs.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `_load_results (line 69)` | trials_csv: str \| Path, summary_csv: str \| Path | tuple[pd.DataFrame, pd.DataFrame] | Load trial and summary CSVs; index the summary by (dimension, n_samples, metric). | Scope: private helper. |
| `_condition_grid (line 85)` | trials: pd.DataFrame | tuple[list[int], list[int]] | Return the sorted dimensions and sample sizes. | Scope: private helper. |
| `_summary_series (line 97)` | summary: pd.DataFrame, dimension: int, sample_sizes: list[int], metric: str | dict[str, np.ndarray] | Read median, 25th/75th percentiles, and SD of one metric across sample sizes. | Scope: private helper. |
| `_jitter (line 119)` | rng: np.random.Generator, center: float, count: int, width: float=0.07 | np.ndarray | Horizontal jitter around a categorical x position. | Scope: private helper. |
| `_finish_n_axis (line 134)` | ax: plt.Axes, sample_sizes: list[int] | None | Label a categorical sample-size x-axis. | Scope: private helper. |
| `plot_rsa_comparison (line 150)` | trials_csv: str \| Path, summary_csv: str \| Path, output_path: str \| Path | Path | Figure 1 PDF: RSA(T, X), RSA(T, Y), and delta RSA across n per dimension. | Scope: task-specific. |
| `plot_pid_decomposition (line 229)` | trials_csv: str \| Path, summary_csv: str \| Path, output_path: str \| Path | Path | Figure 2 PDF: box plots of corrected PID atoms with raw medians, per dimension and n. | Scope: task-specific. Conventions: bits; X1 = X, X2 = Y. |
| `plot_rsa_vs_unique (line 298)` | trials_csv: str \| Path, summary_csv: str \| Path, output_path: str \| Path | Path | Figure 3 PDF: per-trial (delta RSA, corrected delta UI) scatter with per-n median and 25th-75th percentiles. | Scope: task-specific. Conventions: separate axes for Spearman units and bits. |
| `plot_all (line 363)` | trials_csv: str \| Path, summary_csv: str \| Path, output_dir: str \| Path | dict[str, Path] | Create all three figures as PDFs. | Scope: task-specific. |
| `main (line 385)` | config: dict \| None=None | dict[str, Path] | Plot the configured CSVs. | Scope: entry point. |

## `simulations/random_variables.py`

File description: Sample generators shared by simulation scenarios.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `all_above_zero_weighted (line 6)` | rng: np.random.Generator, n: int, p: int, noise_std: float, unique1_weight: float=5.0, unique2_weight: float=5.0, redundant_weight: float=1.0, shared_noise_weight: float=1.0 | tuple[np.ndarray, np.ndarray, np.ndarray] | Generate a Gaussian example in which every PID atom can be positive. | Scope: reusable/public. Conventions: returns samples ordered [X1, X2, T]. |
| `con_all_above_zero_weighted (line 61)` | rng: np.random.Generator, n: int, p: int, noise_std: float, unique1_weight: float=5.0, unique2_weight: float=5.0, redundant_weight: float=1.0, shared_noise_weight: float=1.0 | tuple[np.ndarray, np.ndarray, np.ndarray] | Generate concatenated unique and redundant Gaussian blocks. | Scope: reusable/public. Conventions: returns samples ordered [X1, X2, T]. |
| `equal_unique (line 105)` | rng: np.random.Generator, n: int, p: int, noise_std: float | tuple[np.ndarray, np.ndarray, np.ndarray] | Generate independent source-specific signals with equal weights. | Scope: reusable/public. Conventions: returns samples ordered [X1, X2, T]. |
| `concatenated_equal_unique (line 132)` | rng: np.random.Generator, n: int, p: int, noise_std: float | tuple[np.ndarray, np.ndarray, np.ndarray] | Generate equal unique information in separate target blocks. | Scope: reusable/public. Conventions: returns samples ordered [X1, X2, T]. |

## `simulations/commonality.py`

File description: Commonality analysis for comparison with BROJA PID.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `commonality_analysis (line 7)` | features_X1, features_X2, target, method='standard', alphas=None, scale_by_target_variance=False | dict | Decompose predictive power into unique, common, and unexplained terms. | Scope: reusable/public. Related: suppression examples. Conventions: sources are sample-by-feature arrays; methods are standard OLS or ridge_cv; this is variance partitioning, not PID. |

## `simulations/suppression_examples.py`

File description: Gaussian suppression generators selectable by name.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `full_suppression (line 29)` | rng: np.random.Generator, n: int, p: int, noise_std: float | tuple[np.ndarray, np.ndarray, np.ndarray] | Generate the full-suppression example. | Scope: reusable/public. Related: `commonality_analysis`. Conventions: X2 is a pure suppressor; returns samples ordered [X1, X2, T]. |
| `unq12_zero (line 58)` | rng: np.random.Generator, n: int, p: int, noise_std: float | tuple[np.ndarray, np.ndarray, np.ndarray] | Generate a suppression example with zero unique information in both sources. | Scope: reusable/public. Conventions: returns samples ordered [X1, X2, T]. |
| `unq2_zero (line 88)` | rng: np.random.Generator, n: int, p: int, noise_std: float | tuple[np.ndarray, np.ndarray, np.ndarray] | Generate a suppression example with zero source-2 unique information. | Scope: reusable/public. Conventions: returns samples ordered [X1, X2, T]. |
| `run_suppression_example (line 126)` | example: str, *, n_samples: int, p: int, noise_std: float, seed: int, bias_correction: bool=False, commonality_method: str='standard', commonality_alphas=None, n_resamples: int=20, n_shuffles: int=20 | dict | Run Eigen-PID and commonality on the same suppression samples. | Scope: reusable/public. Related: `pid_calc`, `commonality_analysis`. Conventions: PID values are Gaussian BROJA bits; commonality is variance partitioning calculated from the identical X1, X2, T samples. |

## `simulations/utils.py`

File description: No module docstring; inspect before reuse.

| Callable | Inputs | Outputs | Purpose | Scope / conventions |
|---|---|---|---|---|
| `check_file_exists (line 22)` | file_path | Unannotated | Check if a file exists at the given path. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `check_folder_exists (line 37)` | folder_path | Unannotated | Check if a folder exists at the given path. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `create_permuation (line 52)` | list_to_permute | Unannotated | This function take a range of indices | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `standardize_np (line 71)` | X, eps: float=1e-12 | Unannotated | Column-standardize a NumPy-compatible array. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `Tee (line 80)` | self, *files | Tee instance | No docstring; inspect implementation before reuse. | Scope: class. Conventions: inspect implementation assumptions before reuse. |
| `Tee.__init__ (line 81)` | self, *files | Unannotated | No docstring; inspect implementation before reuse. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `Tee.write (line 84)` | self, data | Unannotated | No docstring; inspect implementation before reuse. | Scope: method/nested helper. Conventions: inspect implementation assumptions before reuse. |
| `Tee.flush (line 89)` | self | Unannotated | No docstring; inspect implementation before reuse. | Scope: method/nested helper. Conventions: inspect implementation assumptions before reuse. |
| `check_equal_type_invariance (line 93)` | a, b | bool | Check if two inputs are equal in value and type invariance. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `meta_exists (line 113)` | meta_data: dict, csv_path | bool | Check whether a row with identical meta_data already exists in a CSV file. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `_to_float_or_none (line 149)` | value | Unannotated | No docstring; inspect implementation before reuse. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `extract_all_components (line 157)` | ca_results: dict, pid_results: dict, mi_results: dict, global_results: dict=None, betas_dict: dict=None | dict | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `summarize_seed_results (line 189)` | results: list[dict] | dict | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `print_seed_summary (line 204)` | summary: dict, n_seeds: int, seed_start: int | None | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `seed_summary_to_table (line 212)` | csv_path: Path \\| str, decimals: int=5, save_path: Path \\| str \\| None=None | pd.DataFrame | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `save_csv_column_means (line 237)` | csv_path: Path \\| str, output_csv_path: Path \\| str, decimals: int=6 | pd.DataFrame | Compute the mean of all numeric CSV columns and save to a new CSV. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `load_csv_and_add_data (line 272)` | csv_path: Path \\| str, data: dict, mode: Literal['append_row', 'update_first_row', 'add_columns']='append_row', save_path: Path \\| str \\| None=None, detect_seed_metadata: bool=True | pd.DataFrame | Load a CSV, add data to it, and save it back. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `save_seed_summary_table_image (line 340)` | csv_path: Path \\| str, image_path: Path \\| str, decimals: int=5, dpi: int=300 | Path \| None | Save the seed-summary table as PDF, replacing any supplied suffix. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `_normalize_config_value (line 374)` | value | Unannotated | No docstring; inspect implementation before reuse. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `get_experiment_name (line 386)` | config: dict | str | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `_parse_csv_numeric (line 400)` | value: str | Unannotated | No docstring; inspect implementation before reuse. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `get_seed_runs_csv_path (line 412)` | config: dict | Path | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `get_seed_summary_csv_path (line 420)` | config: dict | Path | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `load_seed_run_checkpoint (line 428)` | config: dict | tuple[Path, list[dict], list[str]] | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `_ensure_seed_runs_header (line 473)` | file_path: Path, config: dict, metric_names: list[str] | None | No docstring; inspect implementation before reuse. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `append_seed_run_checkpoint (line 491)` | config: dict, row: dict, metric_names: list[str] | Path | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `save_seed_summary_csv (line 503)` | summary: dict, config: dict | Path | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `run_multi_seed_experiment (line 542)` | config: dict, per_seed_runner: Callable[[int, dict], dict] | tuple[dict, list[dict]] | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `run_configured_multiseed (line 593)` | config: dict, per_seed_runner: Callable[[int, dict], dict] | tuple[dict, list[dict], Path, Path] | Run a configured multi-seed experiment and handle the standard reporting. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `create_distribution_plot_with_colors (line 617)` | data: list[float], title: str, xlabel: str, ylabel: str, save_path: Path, bins: int=30, kde: bool=True, bar_color: str='#4C72B0', kde_color: str='#DD8452', bar_alpha: float=0.65, xlim: tuple[float, float] \\| None=None | None | Save a histogram/KDE distribution plot as PDF. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `_load_results_dataframe (line 668)` | path: Path | pd.DataFrame | No docstring; inspect implementation before reuse. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `load_hist_kde_and_change_colors (line 694)` | csv_path: Path \\| str, column: str, output_path: Path \\| str, bins: int=30, bar_color: str='#4C72B0', kde_color: str='#DD8452', bar_alpha: float=0.65 | Path \| None | Load one result column and save its histogram/KDE as PDF. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `create_test_histograms_with_kde (line 731)` | csv_path: Path \\| str, output_dir: Path \\| str, columns: list[str] \\| None=None, bins: int=30, bar_color: str='#4C72B0', kde_color: str='#DD8452', bar_alpha: float=0.6, shared_x_axis: bool=True, shared_x_axis_groups: list[list[str]]=[['CA_R²_X1', 'CA_R²_X2', 'CA_R²_X12'], ['CA_unique_X1', 'CA_unique_X2', 'CA_common'], ['PID_red'], ['PID_unq1', 'PID_unq2'], ['PID_syn'], ['I(M1;T)', 'I(M2;T)', '"I(M1,M2;T)"']] | list[Path] | Create one PDF histogram/KDE per selected result column. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
| `create_test_histograms_with_kde._compute_xlim (line 769)` | arrays: list[np.ndarray] | tuple[float, float] \| None | No docstring; inspect implementation before reuse. | Scope: private helper. Conventions: inspect implementation assumptions before reuse. |
| `get_config (line 831)` | config_path: Path \\| str | dict | No docstring; inspect implementation before reuse. | Scope: reusable/public. Conventions: inspect implementation assumptions before reuse. |
