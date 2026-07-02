# Function Registry

Purpose: before adding a new function, search this document first. If a function already exists, reuse it. If a new function is created, add it here with path, inputs, outputs, and a short purpose.

Scope: project Python files only. Excludes `external/`, `uni_tests/`, `.git/`, and `__pycache__/`.

Generated from AST, so signatures and line numbers reflect the current code. Descriptions come from docstrings when present; entries without docstrings are marked so they can be improved later.

## Folder Overview

- repository root: Repository-level package markers and broad utilities.
- `pipeline/`: Real-data source/target feature extraction helpers, layer utilities, and agnostic PID comparison runners.
- `pipeline/subj_PCs/`: Subject-level PCA fitting, held-out variance evaluation, and plotting.
- `Partial_Information_Decomposition/`: PID calculation, mutual information helpers, bias correction, plotting, and PID-specific utilities.
- `Partial_Information_Decomposition/Idep/`: Idep PID estimators and Gaussian implementation classes.
- `Partial_Information_Decomposition/Idep/Idep_Simulations/`: Simulation, covariance, shrinkage, and analysis helpers for Idep experiments.
- `Partial_Information_Decomposition/Idep/non_parametric_bias_corr/`: Bootstrap, jackknife, and resampling utilities for non-parametric bias correction.
- `encoding_model/`: Encoding model, commonality analysis, regression, and prediction pipeline utilities.
- `data/`: Data-specific loading/parsing scripts.
- `library_wrappers/`: CLI and Python wrappers around external PID/R implementations.
- `Simulations/Encoder_simulation/`: Encoder-based simulation scripts for unique/shared information examples.
- `Simulations/Theoretical_Examples/Covariance/`: Theoretical covariance examples, sampling, and result utilities.
- `Simulations/Theoretical_Examples/RVs_Story/`: Random-variable story examples, truth helpers, batching, and Flow-PID grid-search tooling.
- `Simulations/Theoretical_Examples/RVs_Story/regular_examples/`: Regular theoretical examples, including equal-unique source examples.
- `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/`: Suppression/suppressor-variable theoretical examples.
- `Simulations/Theoretical_Examples/RVs_Story/Non-gaussian/`: Non-Gaussian random-variable story examples.
- `Simulations/evil_twin/`: Evil-twin covariance examples, PID sweeps, and related checks.
- `source_conwell_code/`: Source Conwell analysis scripts kept inside the project tree.
- `source_conwell_code/pressures/`: Source Conwell pressure analysis scripts.
- `source_conwell_code/pressures/brain_data/`: Source Conwell brain-data benchmark and parsing helpers.
- `supression_effect/`: Suppression-effect model experiments and encoder definitions.
- `toy_examples/`: Small toy scripts for covariance, PID, and suppression demonstrations.

## Repository Root

Repository-level package markers and broad utilities.

### `__init__.py`

File description: Python module for   init  -related project logic.

No functions or methods defined in this file.

### `utils.py`

File description: General utility functions shared by analysis and simulation scripts.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `check_file_exists (line 21)` | file_path | `new_file_path` | Check if a file exists at the given path. if it exists change it's name by adding a number at the end. |
| `check_folder_exists (line 36)` | folder_path | `new_folder_path` | Check if a folder exists at the given path. if it doesn't exist, create it. |
| `create_permuation (line 51)` | list_to_permute | call `permute_type(...)` | This function take a range of indices and return a permuted version of it. |
| `standardize_np (line 70)` | X, eps: float=1e-12 | `(X - mean) / (std + eps)` | Column-standardize a NumPy-compatible array. |
| `Tee class (line 79)` | class | class | Class with methods listed below. |
| `Tee.__init__ (line 80)` | self, *files | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `Tee.write (line 83)` | self, data | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `Tee.flush (line 88)` | self | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `check_equal_type_invariance (line 92)` | a, b | Annotated: `bool` | Check if two inputs are equal in value and type invariance. |
| `meta_exists (line 112)` | meta_data: dict, csv_path | Annotated: `bool` | Check whether a row with identical meta_data already exists in a CSV file. it is invariant to type differences (e.g., int vs float vs str). |
| `_to_float_or_none (line 148)` | value | None; call `float(...)` | No docstring; infer behavior from name/signature before reuse. |
| `extract_all_components (line 156)` | ca_results: dict, pid_results: dict, mi_results: dict, global_results: dict=None, betas_dict: dict=None | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `summarize_seed_results (line 188)` | results: list[dict] | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `print_seed_summary (line 203)` | summary: dict, n_seeds: int, seed_start: int | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `seed_summary_to_table (line 211)` | csv_path: Path \| str, decimals: int=5, save_path: Path \| str \| None=None | Annotated: `pd.DataFrame` | No docstring; infer behavior from name/signature before reuse. |
| `save_csv_column_means (line 236)` | csv_path: Path \| str, output_csv_path: Path \| str, decimals: int=6 | Annotated: `pd.DataFrame` | Compute the mean of all numeric CSV columns and save to a new CSV. |
| `load_csv_and_add_data (line 271)` | csv_path: Path \| str, data: dict, mode: Literal['append_row', 'update_first_row', 'add_columns']='append_row', save_path: Path \| str \| None=None, detect_seed_metadata: bool=True | Annotated: `pd.DataFrame` | Load a CSV, add data to it, and save it back. |
| `save_seed_summary_table_image (line 339)` | csv_path: Path \| str, image_path: Path \| str, decimals: int=5, dpi: int=300 | Annotated: `Path \| None` | No docstring; infer behavior from name/signature before reuse. |
| `_normalize_config_value (line 373)` | value | `value`; `'np.random.Generator'`; call `str(...)`; call `float(...)`; ... | No docstring; infer behavior from name/signature before reuse. |
| `get_experiment_name (line 385)` | config: dict | Annotated: `str` | No docstring; infer behavior from name/signature before reuse. |
| `_parse_csv_numeric (line 399)` | value: str | `''`; call `float(...)`; `stripped` | No docstring; infer behavior from name/signature before reuse. |
| `get_seed_runs_csv_path (line 411)` | config: dict | Annotated: `Path` | No docstring; infer behavior from name/signature before reuse. |
| `get_seed_summary_csv_path (line 419)` | config: dict | Annotated: `Path` | No docstring; infer behavior from name/signature before reuse. |
| `load_seed_run_checkpoint (line 427)` | config: dict | Annotated: `tuple[Path, list[dict], list[str]]` | No docstring; infer behavior from name/signature before reuse. |
| `_ensure_seed_runs_header (line 472)` | file_path: Path, config: dict, metric_names: list[str] | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `append_seed_run_checkpoint (line 490)` | config: dict, row: dict, metric_names: list[str] | Annotated: `Path` | No docstring; infer behavior from name/signature before reuse. |
| `save_seed_summary_csv (line 502)` | summary: dict, config: dict | Annotated: `Path` | No docstring; infer behavior from name/signature before reuse. |
| `run_multi_seed_experiment (line 541)` | config: dict, per_seed_runner: Callable[[int, dict], dict] | Annotated: `tuple[dict, list[dict]]` | No docstring; infer behavior from name/signature before reuse. |
| `run_configured_multiseed (line 592)` | config: dict, per_seed_runner: Callable[[int, dict], dict] | Annotated: `tuple[dict, list[dict], Path, Path]` | Run a configured multi-seed experiment and handle the standard reporting. |
| `create_distribution_plot (line 616)` | data: list[float], title: str, xlabel: str, ylabel: str, save_path: Path, bins: int=30, kde: bool=True | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `create_distribution_plot_with_colors (line 649)` | data: list[float], title: str, xlabel: str, ylabel: str, save_path: Path, bins: int=30, kde: bool=True, bar_color: str='#4C72B0', kde_color: str='#DD8452', bar_alpha: float=0.65, xlim: tuple[float, float] \| None=None | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `_load_results_dataframe (line 700)` | path: Path | Annotated: `pd.DataFrame` | No docstring; infer behavior from name/signature before reuse. |
| `load_hist_kde_and_change_colors (line 726)` | csv_path: Path \| str, column: str, output_path: Path \| str, bins: int=30, bar_color: str='#4C72B0', kde_color: str='#DD8452', bar_alpha: float=0.65 | Annotated: `Path \| None` | No docstring; infer behavior from name/signature before reuse. |
| `create_test_histograms_with_kde (line 763)` | csv_path: Path \| str, output_dir: Path \| str, columns: list[str] \| None=None, bins: int=30, bar_color: str='#4C72B0', kde_color: str='#DD8452', bar_alpha: float=0.6, shared_x_axis: bool=True, shared_x_axis_groups: list[list[str]]=[['CA_R²_X1', 'CA_R²_X2', 'CA_R²_X12'], ['CA_unique_X1', 'CA_unique_X2', 'CA_common'], ['PID_red'], ['PID_unq1', 'PID_unq2'], ['PID_syn'], ['I(M1;T)', 'I(M2;T)', '"I(M1,M2;T)"']] | Annotated: `list[Path]` | No docstring; infer behavior from name/signature before reuse. |
| `create_test_histograms_with_kde._compute_xlim (line 801)` | arrays: list[np.ndarray] | Annotated: `tuple[float, float] \| None` | No docstring; infer behavior from name/signature before reuse. |
| `get_config (line 863)` | config_path: Path \| str | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `_to_numpy_samples (line 871)` | data | `data` | Convert torch/numpy samples to the numpy format expected by flow-pid. |

## pipeline

Real-data source/target feature extraction helpers, layer utilities, and agnostic PID comparison runners.

### `pipeline/analysis/pca_analysis/all_models_pairwise/pair_wise_comp.py`

File description: Extract deterministic memory-safe model projections once per job and run resumable PID comparisons for every unordered model pair.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `to_deepdive_model_name (line 55)` | model_name: str | Annotated: `str` | Convert stored CLIP ViT aliases to canonical DeepDive identifiers and print each conversion; return all other identifiers unchanged. Task-specific naming-boundary helper. |
| `deterministic_pca (line 73)` | features: Any, n_components: int, random_state: int | Annotated: `np.ndarray` | Apply seeded randomized PCA to a 2D sample matrix and return a float64 projection, capping the requested components to the valid matrix dimensions. |
| `extract_model_projection (line 105)` | model_name: str, target_context: dict[str, Any], choose_layer_kwargs: dict[str, Any], feature_extraction_kwargs: dict[str, Any], n_components: int, random_state: int | Annotated: `tuple[np.ndarray, int]` | Extract one selected model layer in bounded batches, optionally apply deterministic SRP, discard raw batches, and return the final PCA projection plus layer index. A null SRP component setting uses the JL dimension from target_context; an integer overrides it. With SRP disabled, the full raw-width intermediate is retained until PCA. |
| `run_pairwise_pid_pipeline (line 210)` | model_1_names: list[str], model_2_names: list[str], otc_config: dict[str, Any], csv_path: str \| Path | Annotated: `Path` | Project the target once, retain only final model PCA arrays in an in-memory per-job cache, compute each unordered PID pair, and checkpoint successful rows to CSV. |

### `pipeline/analysis/pca_analysis/pca_as_function.py`

File description: Python module for pca as function-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `pca_as_function (line 29)` | pipeline_config: str \| Path, pca_config: str \| Path | Annotated: `dict[str, Any]` | Run the full-OTC PID experiment from the YAML config beside this file. |
| `plot_ (line 62)` | results_dict: dict[str, Any], pca_config: str, pipeline_config: str | Annotated: `None` | Plot the results of the PID computation as a function of the number of PCA components. |

### `pipeline/analysis/pca_analysis/otc_unique_search_pca.py`

File description: Run PCA unique-information subset search on full OTC data and two sources.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `run_otc_unique_search_pca (line 27)` | config: dict[str, Any] \| str \| Path, max_source_components: int, *, search_source: str='X1', search_kwargs: dict[str, Any] \| None=None, all_csv_path: str \| Path \| None=None, best_csv_path: str \| Path \| None=None, pid_callable: Callable[..., Any] \| None=None | Annotated: `dict[str, Any]` | Load OTC/source arrays, fix target/other-source PCs, and search one source. |
| `run_otc_unique_search_from_yaml (line 77)` | config_path: str \| Path=DEFAULT_SEARCH_CONFIG | Annotated: `dict[str, Any]` | Run OTC unique search from one YAML analysis config file. |
| `_load_config (line 98)` | config: dict[str, Any] \| str \| Path | Annotated: `dict[str, Any]` | Load an OTC config dictionary or YAML file. |
| `_resolve_path (line 113)` | value: str \| Path \| None, base_dir: Path | Annotated: `Path \| None` | Resolve optional YAML paths relative to the YAML file directory. |
| `_load_otc_arrays (line 126)` | config: dict[str, Any] | Annotated: `dict[str, Any]` | Run PIDPipeline only through target/source/layer/feature extraction. |
| `_skip_pid (line 148)` | target: Any, source_1: Any, source_2: Any, **pid_kwargs: Any | Annotated: `None` | Satisfy PIDPipeline while loading arrays without running PID. |
| `_pca_search_inputs (line 159)` | context: dict[str, Any], config: dict[str, Any], search_source: str, max_source_components: int | Annotated: `tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]` | Project target, searched source, and fixed source for subset search. |
| `_project_or_keep (line 191)` | features: Any, n_components: int \| None, name: str | Annotated: `np.ndarray` | Apply scikit-learn PCA when requested, capping components to the valid matrix size. |
| `main (line 216)` | config_path: str \| Path \| None=None | Annotated: `None` | Run OTC PCA unique search from a YAML config file. |

### `pipeline/analysis/pca_analysis/unique_search_pca.py`

File description: Search source-1 PCA component subsets for source-1 unique PID information.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `run_pid_pc_subset_search (line 25)` | target: Any, source_1: Any, source_2: Any, pid_callable: Callable[..., Any] \| None=None, *, cmi_threshold: float=1e-06, unique_threshold: float=1e-06, beam_width: int=5, max_subset_size: int=3, initial_subset_size: int=1, initial_subset_count: int \| None=None, floating_tolerance: float=1e-09, max_runtime_seconds: float=600, rng_seed: int=56, pid_kwargs: dict[str, Any] \| None=None, all_csv_path: str \| Path \| None=None, best_csv_path: str \| Path \| None=None, use_floating_backward: bool=True | Annotated: `dict[str, Any]` | Run beam search over source_1 PCA columns for source-1 unique PID. |
| `_initial_subsets (line 189)` | candidates: list[int], subset_size: int, subset_count: int \| None, rng_seed: int | Annotated: `list[tuple[int, ...]]` | Create initial source_1 PC subsets for the beam search. |
| `_as_2d_array (line 210)` | value: Any, name: str | Annotated: `np.ndarray` | Convert one input to a finite non-empty 2D float array. |
| `_gaussian_cmi_bits (line 225)` | x: np.ndarray, y: np.ndarray, z: np.ndarray, eps: float=1e-10 | Annotated: `float` | Calculate Gaussian conditional MI I(x; y \| z) in bits. |
| `_conditional_cov (line 242)` | cov_a: np.ndarray, cross_ab: np.ndarray, cov_b: np.ndarray, eps: float | Annotated: `np.ndarray` | Compute covariance of variable a conditioned on variable b. |
| `_logdet (line 253)` | matrix: np.ndarray, eps: float | Annotated: `float` | Return a stable log determinant for a covariance-like matrix. |
| `_evaluate_subset (line 266)` | subset: tuple[int, ...], target: np.ndarray, source_1: np.ndarray, source_2: np.ndarray, pipeline: PIDPipeline, pid_kwargs: dict[str, Any], cache: dict[tuple[int, ...], dict[str, Any]], all_csv_path: str \| Path \| None, unique_threshold: float, start: float, cmi_score: float \| None | Annotated: `dict[str, Any]` | Evaluate one source_1 PC subset with PIDPipeline and cache the result. |
| `_floating_backward (line 310)` | row: dict[str, Any], target: np.ndarray, source_1: np.ndarray, source_2: np.ndarray, pipeline: PIDPipeline, pid_kwargs: dict[str, Any], cache: dict[tuple[int, ...], dict[str, Any]], all_csv_path: str \| Path \| None, unique_threshold: float, tolerance: float, start: float, max_runtime_seconds: float | Annotated: `dict[str, Any]` | Prune PCs whose removal does not meaningfully reduce unique information. |
| `_pid_components (line 346)` | pid_result: Any | Annotated: `dict[str, Any]` | Extract a PID component dictionary from common project PID result shapes. |
| `_to_float (line 362)` | value: Any | Annotated: `float` | Convert numeric scalar-like values to float. |
| `_top_rows (line 372)` | rows: list[dict[str, Any]], beam_width: int | Annotated: `list[dict[str, Any]]` | Keep highest-unique rows, de-duplicated by subset. |
| `_append_csv_row (line 383)` | path: str \| Path \| None, subset: tuple[int, ...], start: float, status: str, *, cmi_score: float \| None=None, row: dict[str, Any] \| None=None | Annotated: `None` | Append one compact row to a CSV file, creating the header when needed. |
| `_toy_pid (line 425)` | target: np.ndarray, source_1: np.ndarray, source_2: np.ndarray, **pid_kwargs: Any | Annotated: `dict[str, dict[str, float]]` | Return a tiny Gaussian-CMI-based PID-like result for local smoke runs. |

### `pipeline/plotting/pairwise_pid_heatmaps.py`

File description: Create publication-friendly PID component matrices from the resumable pairwise OTC CSV.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `plot_pairwise_pid_matrices (line 12)` | csv_path: str \| Path, output_dir: str \| Path, *, model_order: list[str] \| None=None, value_format: str='.3f', cmap: str='viridis', figsize: tuple[float, float] \| None=None, dpi: int=300 | Annotated: `dict[str, Path]` | Validate ordered pair rows, construct directional unique-information and symmetric redundancy/synergy matrices, then save annotated PNG and CSV outputs. Plotting-specific; preserves the X1/unq1 and X2/unq2 convention from `run_pairwise_pid_pipeline`. |

### `pipeline/subj_PCs/subj_pc_analysis.py`

File description: Fit subject-level PCA on unique NSD images and evaluate retained components on shared held-out images.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `split_unique_shared (line 17)` | subj_id: str, hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, variance_threshold: float=0.99 | Annotated: `dict[str, Any]` | Load a subject context and use its one-dimensional `shared1000_subj` mask to split aligned neural rows and subject image IDs. Task-specific; `unique_neural_data` is training data and `shared_neural_data` is held out. |
| `pca_by_variance (line 78)` | neural_data: np.ndarray, variance_threshold: float=0.99 | Annotated: `dict[str, Any]` | Fit `StandardScaler` and full-SVD PCA on two-dimensional training samples, retaining the requested training-variance fraction. Returns the fitted models and training scores; expects a float threshold in `(0, 1]`, with `1.0` retaining all components. |
| `heldout_pca (line 121)` | pca_model: PCA, scaler_model: StandardScaler, heldout_data: np.ndarray | Annotated: `np.ndarray` | Apply the training scaler and fitted PCA basis to aligned two-dimensional held-out samples without refitting either transformation. |
| `main (line 150)` | subj_id: str, hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, variance_threshold: float=0.99, save_models_path: str \| Path \| None=None | Annotated: `dict[str, Any]` | Run the subject PCA analysis and optionally save models plus a per-PC held-out variance CSV. Held-out ratios divide each PC score's sample variance by total held-out sample variance after training-set scaling; PC indices are one-based. |

### `pipeline/subj_PCs/plotting.py`

File description: Plot saved held-out PCA variance explained by component index.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `plot_heldout_variance_explained (line 12)` | variance_csv_path: str \| Path, output_path: str \| Path \| None=None, *, show_cumulative: bool=True, dpi: int=300 | Annotated: `Path` | Read the CSV produced by `subj_pc_analysis.main`, validate PC indices and held-out explained-variance ratios, and save a per-PC percentage bar plot with an optional cumulative line. |

### `pipeline/full_OTC/otc_experiment.py`

File description: Thin config-driven full-OTC experiment runner that uses the full neural response matrix as target.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `run_otc_experiment (line 21)` | config: dict[str, Any] | Annotated: `dict[str, Any]` | Run one full-OTC experiment from an already-loaded config dictionary. |
| `nsd_otc_target (line 39)` | hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, subj_id: str \| None=None, voxel_index: int \| None=None, n_images: int \| None=None | Annotated: `dict[str, Any]` | Load the full OTC target matrix and expose it under the PIDPipeline target key. |
| `_validate_config (line 78)` | config: dict[str, Any] | Annotated: `None` | Validate the config sections needed to call the full-OTC runner. |

### `pipeline/full_OTC/otc_run.py`

File description: Run the full-OTC PID experiment from the YAML config beside this file.

No functions or methods defined in this file.

### `pipeline/pid_pipeline.py`

File description: Strict orchestrator for one PID pipeline run from user-selected target/source/layer/extraction/PID/report functions.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `PIDPipelineFunctions class (line 10)` | class | class | Store the user-selected functions for one PID pipeline run. |
| `PIDPipeline class (line 42)` | class | class | Run the PID pipeline by connecting the provided functions in order. |
| `PIDPipeline.__init__ (line 45)` | self, functions: PIDPipelineFunctions | Annotated: `None` | Create a strict PID pipeline orchestrator. |
| `PIDPipeline.run (line 85)` | self, *, target_kwargs: dict[str, Any] \| None=None, sources_kwargs: dict[str, Any] \| None=None, choose_layer_kwargs: dict[str, Any] \| None=None, feature_extraction_kwargs: dict[str, Any] \| None=None, preprocess_kwargs: dict[str, Any] \| None=None, feature_manipulation_kwargs: dict[str, Any] \| None=None, pid_kwargs: dict[str, Any] \| None=None, report_kwargs: dict[str, Any] \| None=None | Annotated: `dict[str, Any]` | Run target, sources, layers, features, optional transforms, PID, and report. |

### `pipeline/pipeline_phases/choosing_layer.py`

File description: Helpers for selecting model layers globally, by index, or voxel-wise from saved encoding results.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `random_layer_selection (line 9)` | n_layers | `layer_idx` | Choose a random layer index from the available layers. |
| `specific_index_layer_selection (line 23)` | layer_names, index | `layer_names[index]` | Choose a specific layer index from the available layers. |
| `voxel_best_layer (line 44)` | voxel_index: int=None, index_layer: int=None, path_to_results: str=None | Annotated: `dict` | Choose the best model layer for one voxel, or a representative voxel for one layer. |
| `overall_best_layer (line 107)` | model_name: str, path_to_results: str | Annotated: `dict` | Choose the overall best layer index for one model from an OTC CSV. |
| `_read_csv_rows (line 146)` | path_to_results: str | Annotated: `tuple[list[dict[str, str]], set[str]]` | Read CSV rows and column names for layer-selection helpers. |
| `_normalize_csv_value (line 163)` | value | Annotated: `str` | Normalize CSV values before exact lookup comparisons. |

### `pipeline/pipeline_phases/feature_manipulations.py`

File description: Python module for feature manipulations-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `pca_projection (line 20)` | features, n_components | `ft_reduced` | Apply PCA projection to reduce dimensionality of features. |
| `jl_projection (line 38)` | features, n_samples, eps=0.1, jl_dim=None | `ft_reduced` | Apply Johnson-Lindenstrauss projection to reduce dimensionality of features. |
| `cca_projection (line 71)` | features1, features2, n_components | tuple of 2 values | Apply Canonical Correlation Analysis (CCA) to find linear combinations of two sets of features that are maximally correlated. |

### `pipeline/pipeline_phases/mi_statistics.py`

File description: Python module for mi statistics-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `assert_mi (line 13)` | own_mi, pid_mi | `True` | This function checks if the mutual information calculated by the PID method is equal to the mutual information calculated by the own method. |

### `pipeline/pipeline_phases/preprocessing_layer.py`

File description: Python module for preprocessing layer-related project logic.

No functions or methods defined in this file.

### `pipeline/pipeline_phases/report_results.py`

File description: Python module for report results-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `print_pid_mi (line 6)` | pid_results, mi_result | No explicit return; likely `None` / side effects. | This functions takes the pid results and the mutual information results and prints them in a nice format. |

### `pipeline/pipeline_phases/sources_target_features.py`

File description: Python module for sources target features-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `prepare_sources (line 21)` | model_name_1: str, model_name_2: str | Annotated: `dict[str, dict]` | Prepare sources for feature extraction. |
| `prepare_target (line 42)` | hdf_path: Path, pkl_info_path: Path, neural_data_path: Path | Annotated: `dict` | Prepare target for feature extraction. |
| `prepare_target_for_voxel (line 64)` | voxel_index: int, subj_id: str, hdf_path: Path, pkl_info_path: Path, neural_data_path: Path | Annotated: `dict` | Prepare target for feature extraction for a specific voxel. |
| `make_nsd_dataloader (line 90)` | model_context: dict, stim_dataset, image_ids: np.ndarray, batch_size: int | Annotated: `DataLoader` | Create a DataLoader for an ordered subset of NSD images. |
| `batching (line 113)` | model_context: dict, batch_start: int, batch_end: int, stim_dataset, subj_image_ids: np.ndarray, layer_name: str, batch_size_dataloader: int | Annotated: `np.ndarray` | Batch process a range of images for feature extraction. |
| `feature_extraction (line 144)` | layer_index: int, model_context: dict, subj_image_ids: np.ndarray, stim_dataset, batch_size_process: int, batch_size_dataloader: int=128 | Annotated: `np.ndarray` | Extract features from the models and the neural data. |

### `pipeline/pipeline_utils.py`

File description: Shared adapters and helpers for config-driven PID experiment runners.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `choose_random_sources (line 18)` | sources_list: list[str], size: int=2, replace: bool=False | Annotated: `np.ndarray` | Randomly select a source from the list of available sources. |
| `run_configured_pid_pipeline (line 34)` | config: dict[str, Any], function_registry: dict[str, PipelineFunction], choose_layer_kwargs: dict[str, Any] \| None=None | Annotated: `dict[str, Any]` | Run PIDPipeline from a config dictionary and function registry. |
| `pipeline_functions_from_config (line 67)` | function_config: dict[str, Any], function_registry: dict[str, PipelineFunction] | Annotated: `PIDPipelineFunctions` | Resolve configured function names into PIDPipelineFunctions. |
| `resolve_pipeline_function (line 94)` | function_config: dict[str, Any], function_registry: dict[str, PipelineFunction], step_name: str, required: bool | Annotated: `PipelineFunction \| None` | Resolve one configured pipeline step name from a registry. |
| `validate_pipeline_config_sections (line 123)` | config: dict[str, Any], required_sections: tuple[str, ...] | Annotated: `None` | Validate that required config sections exist. |
| `nsd_sources (line 139)` | model_name_1: str, model_name_2: str | Annotated: `dict[str, dict[str, Any]]` | Load two model contexts and expose them under X1 and X2. |
| `specific_layer_index (line 156)` | sources: dict[str, dict[str, Any]], X1_index: int, X2_index: int | Annotated: `dict[str, int]` | Select one configured layer index for each source. |
| `random_layer_selection_for_sources (line 180)` | sources: dict[str, dict[str, Any]], random_seed: int \| None=None | Annotated: `dict[str, int]` | Select a random valid layer index for each source. |
| `voxel_best_layer_for_sources (line 202)` | sources: dict[str, dict[str, Any]], X1_index: int, X2_index: int, X1_path_to_results: str \| Path, X2_path_to_results: str \| Path | Annotated: `dict[str, int]` | Select each source model's best layer for one voxel index. |
| `overall_best_layer_for_sources (line 236)` | sources: dict[str, dict[str, Any]], path_to_results: str \| Path | Annotated: `dict[str, int]` | Select each source model's overall best OTC layer from one CSV file. |
| `nsd_feature_extraction (line 267)` | source_context: dict[str, Any], layer_index: int, target_context: dict[str, Any], batch_size_process: int, batch_size_dataloader: int=128 | Annotated: `Any` | Extract features for one NSD source and selected layer. |
| `pca_each_source (line 299)` | target: Any, source_1: Any, source_2: Any, n_components_source_1: int, n_components_source_2: int, n_components_target: int | Annotated: `tuple[Any, Any]` | Apply PCA separately to source_1 and source_2. |
| `pid_calc_adapter (line 334)` | target: Any, source_1: Any, source_2: Any, method: str, config: dict[str, Any] \| None=None, rng_seed: int=56, **pid_kwargs: Any | Annotated: `dict[str, Any]` | Call pid_calc using the strict PIDPipeline array order. |
| `print_pid_mi_adapter (line 381)` | pid_results: dict[str, Any], context: dict[str, Any], **report_kwargs: Any | Annotated: `Any` | Print PID and MI outputs from pid_calc_adapter. |
| `_as_2d_tensor (line 399)` | value: Any | Annotated: `Any` | Convert samples to a 2D torch tensor. |
| `_random_layer_index_for_source (line 420)` | sources: dict[str, dict[str, Any]], source_name: str, rng: np.random.Generator | Annotated: `int` | Select one random layer index for one source context. |
| `_layer_index_values (line 450)` | sources: dict[str, dict[str, Any]], source_name: str, requested_index: int | Annotated: `list[int]` | Create valid layer-index values for one source. |
| `_model_name_for_source (line 468)` | sources: dict[str, dict[str, Any]], source_name: str | Annotated: `str` | Read a source model name from a source context. |
| `_overall_best_layer_model_names (line 485)` | path_to_results: str \| Path | Annotated: `list[str]` | Read model names from an overall best-layer CSV for diagnostics. |
| `source_context (line 519)` | sources: Any, source_name: str | Annotated: `Any` | Read one source context from the sources object. |
| `choose_one_layer (line 535)` | layer_func: Callable[..., Any], source_context_value: Any, layer_kwargs: dict[str, Any] | Annotated: `Any` | Choose one layer by adapting to the common layer-selection helper signatures. |

### `pipeline/toy_examples/pid_pipeline_toy_example.py`

File description: Tiny no-data-loading example for debugging the strict PID pipeline flow.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `toy_target_extraction (line 17)` | No inputs | Annotated: `dict[str, Any]` | Create tiny fake target data. |
| `toy_sources_extraction (line 30)` | model_1: str, model_2: str | Annotated: `dict[str, dict[str, Any]]` | Create tiny fake source contexts with two layers per source. |
| `toy_choose_layer (line 61)` | sources: dict[str, dict[str, Any]], layer_index: int=0 | Annotated: `dict[str, str]` | Choose one layer for each source by index. |
| `toy_feature_extraction (line 78)` | source_context: dict[str, Any], layer_name: str, target_context: dict[str, Any] | Annotated: `list[list[float]]` | Read fake features for one selected source layer. |
| `toy_preprocess (line 98)` | target: list[list[float]], source_1: list[list[float]], source_2: list[list[float]], scale: float=1.0 | Annotated: `tuple[list[list[float]], list[list[float]], list[list[float]]]` | Scale target and source values together as a visible preprocessing step. |
| `toy_feature_manipulation (line 123)` | source_1: list[list[float]], source_2: list[list[float]], keep_columns: int=1 | Annotated: `tuple[list[list[float]], list[list[float]]]` | Keep the first columns from both source feature matrices. |
| `toy_pid_calculation (line 145)` | target: list[list[float]], source_1: list[list[float]], source_2: list[list[float]], method_name: str='toy_pid' | Annotated: `dict[str, Any]` | Return a readable dummy PID result from tiny arrays. |
| `toy_pid_report (line 177)` | pid_results: dict[str, Any], context: dict[str, Any] | Annotated: `str` | Print a compact toy pipeline report. |
| `main (line 198)` | No inputs | Annotated: `dict[str, Any]` | Run the tiny toy PID pipeline. |

### `pipeline/toy_examples/voxel_experiment_smoke_example.py`

File description: No-data smoke example for debugging the config-driven voxel experiment runner.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `smoke_voxel_target (line 17)` | voxel_index: int, subj_id: str, n_images: int=3 | Annotated: `dict[str, Any]` | Create a tiny fake voxel target context. |
| `smoke_sources (line 38)` | model_name_1: str, model_name_2: str | Annotated: `dict[str, dict[str, Any]]` | Create tiny fake source contexts. |
| `smoke_choose_layer (line 69)` | sources: dict[str, dict[str, Any]], X1_index: int, X2_index: int | Annotated: `dict[str, int]` | Choose fake layer indexes from config. |
| `smoke_feature_extraction (line 85)` | source_context: dict[str, Any], layer_index: int, target_context: dict[str, Any], feature_shift: float=0.0 | Annotated: `list[list[float]]` | Read fake source features for one selected layer. |
| `smoke_feature_manipulation (line 111)` | source_1: list[list[float]], source_2: list[list[float]], keep_columns: int=1 | Annotated: `tuple[list[list[float]], list[list[float]]]` | Keep the first columns from both fake source matrices. |
| `smoke_pid_calculation (line 133)` | target: list[list[float]], source_1: list[list[float]], source_2: list[list[float]], method: str='smoke_pid' | Annotated: `dict[str, Any]` | Return a tiny deterministic PID-like result. |
| `smoke_report (line 167)` | pid_results: dict[str, Any], context: dict[str, Any] | Annotated: `str` | Print a compact smoke run report. |
| `register_smoke_functions (line 188)` | No inputs | Annotated: `None` | Register smoke wrapper functions for this example run. |
| `smoke_config (line 211)` | No inputs | Annotated: `dict[str, Any]` | Create a small YAML-shaped config for run_voxel_experiment. |
| `main (line 242)` | No inputs | Annotated: `dict[str, Any]` | Run the voxel experiment smoke example. |

### `pipeline/trash/old_middle_man_functions.py`

File description: Deprecated pipeline helpers moved out of the active pipeline.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `prepare_sources_step (line 19)` | prepare_sources: Callable[..., Any] \| None, model_1: str \| None, model_2: str \| None, source_kwargs: dict[str, Any], context: dict[str, Any] | Annotated: `Any` | Prepare source contexts when a source-preparation function is provided. |
| `prepare_target_step (line 46)` | prepare_target: Callable[..., Any] \| None, target_kwargs: dict[str, Any], context: dict[str, Any] | Annotated: `Any` | Prepare target context when a target-preparation function is provided. |
| `target_data_step (line 67)` | target_data: Any, context: dict[str, Any] | Annotated: `Any` | Choose the target samples for PID. |
| `choose_layers_step (line 86)` | choose_layer: Callable[..., Any] \| None, layer_kwargs: dict[str, Any], context: dict[str, Any] | Annotated: `dict[str, Any]` | Choose one layer for each source when a layer-selection function is provided. |
| `extract_or_use_features_step (line 115)` | extract_features: Callable[..., Any] \| None, source_1_features: Any, source_2_features: Any, feature_extraction_kwargs: dict[str, Any], context: dict[str, Any] | Annotated: `dict[str, Any]` | Extract source features or use precomputed source features. |
| `manipulate_features_step (line 168)` | manipulate_features: Callable[..., Any] \| None, feature_manipulation_kwargs: dict[str, Any], context: dict[str, Any] | Annotated: `dict[str, Any]` | Apply an optional feature-manipulation function to source features. |
| `calculate_pid_step (line 200)` | calculate_pid: Callable[..., Any] \| None, pid_kwargs: dict[str, Any], context: dict[str, Any] | Annotated: `Any` | Calculate PID when a PID function is provided. |
| `report_results_step (line 227)` | report: Callable[..., Any] \| None, report_kwargs: dict[str, Any], context: dict[str, Any] | Annotated: `Any` | Report PID results when a reporting function is provided. |
| `source_context (line 248)` | sources: Any, source_name: str | Annotated: `Any` | Read one source context from the sources object. |
| `source_step_kwargs (line 264)` | kwargs: dict[str, Any], source_name: str | Annotated: `dict[str, Any]` | Merge shared and source-specific keyword arguments for one source step. |
| `call_pid_function (line 282)` | pid_func: Callable[..., Any], target_data: Any, source_1_features: Any, source_2_features: Any, pid_kwargs: dict[str, Any] | Annotated: `Any` | Call a PID function using either local or existing PID_calc-style arguments. |
| `choose_layer_function (line 310)` | layer_func_name: str | `layer_funcs[layer_func_name]` | Resolve an old layer-selection helper name to a callable. |
| `choose_manipulation_function (line 338)` | manip_func_name: str | `manip_funcs[manip_func_name]` | Resolve an old feature-manipulation helper name to a callable. |
| `ica_projection (line 361)` | features | No explicit return; likely `None` / side effects. | Placeholder for ICA feature reduction. |
| `run_feature_reduction_smoke (line 374)` | config_path: Path \| str | Annotated: `dict` | Run the old feature-reduction smoke test using source features from a YAML config. |
| `extract_NSD_model_transform (line 441)` | model, stim_dataset, subj_image_ids | call `make_nsd_dataloader(...)` | Create a full-subject NSD DataLoader using a model's image transforms. |
| `features_pipeline (line 463)` | model1, model2, subj_id, hdf_path: Path, pkl_info_path: Path, neural_data_path: Path | Annotated: `dict` | Run the old source/target feature extraction pipeline. |
| `smoke_example_config (line 506)` | No inputs | dict (MODEL_1_NAME, MODEL_2_NAME, DEBUG_LAYER_1, DEBUG_LAYER_2, layer_func, manipulation_func, N_DEBUG_IMAGES, BATCH_SIZE_PROCESS...) | Load the old smoke-example configuration. |

### `pipeline/voxel_experiments/voxel_experiment.py`

File description: Thin config-driven voxel experiment runner that owns voxel target extraction and delegates shared pipeline adapters to `pipeline_utils`.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `run_voxel_experiment (line 23)` | config: dict[str, Any] | Annotated: `dict[str, Any]` | Run one voxel experiment from an already-loaded config dictionary. |
| `nsd_voxel_target (line 43)` | voxel_index: int, subj_id: str, hdf_path: str \| Path, pkl_info_path: str \| Path, neural_data_path: str \| Path, n_images: int \| None=None | Annotated: `dict[str, Any]` | Load one voxel target and expose it under the PIDPipeline target key. |
| `_choose_layer_kwargs (line 84)` | config: dict[str, Any] | Annotated: `dict[str, Any]` | Arrange voxel-specific kwargs for the configured choose_layer function. |
| `_validate_config (line 100)` | config: dict[str, Any] | Annotated: `None` | Validate the config sections needed to call the voxel runner. |

### `pipeline/voxel_experiments/voxel_run.py`

File description: Python module for voxel run-related project logic.

No functions or methods defined in this file.

## Partial_Information_Decomposition

PID calculation, mutual information helpers, bias correction, plotting, and PID-specific utilities.

### `Partial_Information_Decomposition/PID_calc.py`

File description: Main PID dispatcher and wrappers for Idep, Tilde, Delta, Thin fallback, and Flow-PID.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `pid_calc (line 28)` | config=None, sources=None, target=None, rng=torch.Generator().manual_seed(56), method=None, on_rvs: callable=None, covariance: torch.Tensor=None | tuple of 2 values | No docstring; infer behavior from name/signature before reuse. |
| `pid_idep_wrapper (line 53)` | config, sources=None, target=None, covariance=None, rng=None, on_rvs=None | tuple of 2 values | This function is a wrapper to PID calculated by Idep_multivariate_gauss class, which implements the Idep PID calculation for multivariate Gaussian variables. This wrapper allows us to use the same input format for bot... |
| `pid_tilde_wrapper (line 81)` | config: dict, sources: list, target: list, covariance: torch.Tensor, rng: torch.random.Generator, on_rvs: callable=None | tuple of 2 values | This function is a wrapper to PID calculated by BROJA and implemented by Venkatesh et al. 2023 Because Idep and BROJA have different input format, this wrapper converts the input format to fit the BROJA implementation... |
| `delta_wrapper (line 119)` | config, sources, target, covariance, rng, on_rvs | tuple of 2 values | This function is a wrapper to PID calculated by BROJA and implemented by Venkatesh et al. 2023 Because Idep and BROJA have different input format, this wrapper converts the input format to fit the BROJA implementation... |
| `_to_numpy_samples (line 175)` | data | `data` | Convert torch/numpy samples to the numpy format expected by flow-pid. |
| `flow_pid_wrapper (line 185)` | config: dict, sources: list, target: list, covariance: torch.Tensor, rng: torch.random.Generator, on_rvs: callable=None | tuple of 2 values | Wrapper for flow-pid. |

### `Partial_Information_Decomposition/PID_util.py`

File description: PID-specific covariance, table, plotting, and helper utilities.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `LinearRegression_fit (line 11)` | X, y | `model` | No docstring; infer behavior from name/signature before reuse. |
| `compute_ridge_cv_r2 (line 25)` | X, y, alphas=None | tuple of 2 values | Compute cross-validated R² using RidgeCV with efficient LOO cross-validation. |
| `cond_cov (line 61)` | sigma_1, sigma_2, sigma12, sigma21 | `cond_cov` | This function will compute the conditional covariance matrix of two Gaussian variables Sigma_1\|2 = Sigma_1 - Sigma12*inv(Sigma_2)*Sigma21 |
| `ledoit_wolf_cov_torch (line 75)` | X: torch.Tensor, assume_centered: bool=False | Annotated: `torch.Tensor` | Fit Ledoit-Wolf on X (N,פ) and return covariance as torch.Tensor on same device/dtype. |
| `create_cov_matrix (line 87)` | rvs: list=[], verbose=False, Sigma=None, dims: list=None, device='cpu', check_singular=True | `cov_dict` | This function will create the covariance matrix for the three variables M1,M2,T input: M1,M2,T are torch tensors of shape (N,p) rvs is a list of the three variables [M1,M2,T] N is the number of observations, p is the... |
| `reorder_cov_blocks (line 152)` | Sigma: torch.Tensor, dims: dict[str, int], old_order: list[str], new_order: list[str] | Annotated: `torch.Tensor` | Reorder covariance matrix blocks according to variable names. |
| `para_create_cov_matrix (line 193)` | dims, Sigmas=None, verbose=False | `cov_dict` | This function will create the covariance matrix for the three variables M1,M2,T |
| `old_para_create_cov_matrix (line 241)` | dims, Sigmas=None, verbose=False | `cov_dict` | This function will create the covariance matrix for the three variables M1,M2,T input: M1,M2,T are torch tensors of shape (N,p) rvs is a list of the three variables [M1,M2,T] N is the number of observations, p is the... |
| `whiten_block (line 290)` | Sigma_xx: torch.Tensor, Sigma_xy: torch.Tensor, Sigma_yy: torch.Tensor | Annotated: `torch.Tensor` | return Ux^{-T} @ Sigma_xy @ Uy^{-1} where Sigma_xx = Ux^T Ux, Sigma_yy = Uy^T Uy, and Ux,Uy are upper triangular. |
| `para_whiten_block (line 307)` | Sigma_xx: torch.Tensor, Sigma_xy: torch.Tensor, Sigma_yy: torch.Tensor | Annotated: `torch.Tensor` | Computes: Ux^{-T} @ Sigma_xy @ Uy^{-1} where Sigma_xx = Ux^T Ux, Sigma_yy = Uy^T Uy, and Ux,Uy are upper triangular. Supports batched inputs of shape (N, d, d). |
| `plot_cov_blocks (line 332)` | cov_dict, x0_dim, x1_dim, x2_dim, *, title='Covariance (block view)', cmap='Blues', vmin=None, vmax=None, fine_grid=False, show_colorbar=True | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `standardize (line 373)` | X: torch.Tensor, eps: float=1e-12 | Annotated: `torch.Tensor` | Standardize columns of X to zero mean and unit variance. |
| `assert_full_rank (line 385)` | X: torch.Tensor, jitter=0 | Annotated: `None` | Assert that the input matrix X is full rank. |
| `correlation_matrix (line 409)` | X | `corr_matrix`; call `np.array(...)` | Compute the correlation matrix of the columns of X. |
| `block_singularity_check (line 424)` | X, tol=1e-10 | tuple of 2 values; `singular_dict` | Check if a block is singular or ill-conditioned. |
| `singularity_report (line 445)` | X_M1, X_M2, y_real, tol=1e-10, return_printing_required=False | `report`; tuple of 2 values | Return min eigenvalue and singularity flag for blocks and combinations. |
| `diagnostic_plots (line 475)` | X_M1, X_M2, y_real, method, mixing_dimension | `cov / np.outer(sx, sy)` | No docstring; infer behavior from name/signature before reuse. |
| `diagnostic_plots.cross_correlation (line 476)` | X, Y | `cov / np.outer(sx, sy)` | No docstring; infer behavior from name/signature before reuse. |
| `vif_summary (line 510)` | X | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `std_scaling_summary (line 542)` | X | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `eigvenvalue_summary (line 561)` | X | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `_get_first (line 569)` | mapping, *keys | `mapping[key]` | No docstring; infer behavior from name/signature before reuse. |
| `compare_results (line 576)` | vp_results, pid_results, mi_results=None | No explicit return; likely `None` / side effects. | Compare Variance Partitioning and Partial Information Decomposition results. |
| `pid_comparison_table (line 622)` | results: dict, decimals: int=4, print_table: bool=True | `rows` | Print and return a compact table comparing PID and MI outputs. |
| `save_pid_comparison_table (line 651)` | results: dict, save_path: str, decimals: int=4, title: str='PID Method Comparison', config: dict=None | `save_path` | Save the PID comparison table as a clean matplotlib image. |
| `plot_mi_heatmap (line 696)` | csv_path, value_col, *, n_col='N', p_col='p', figsize=(7, 5), title=None, save_path=None, annotate=True, fmt='.3f', cmap='viridis' | No explicit return; likely `None` / side effects. | Plot a block heatmap from an averaged CSV. |
| `plot_all_mi_heatmaps (line 816)` | csv_path, title='Mutual Information Heatmaps', *, n_col='N', p_col='p', figsize=(16, 5), save_path=None, annotate=True, mean_fmt='.2f', std_fmt='.2f', log_scale=False, cmap='viridis', annotation_mode='pm', fontsize=9, aggfunc='mean' | No explicit return; likely `None` / side effects. | Plot theoretical, naive, and bias-corrected MI heatmaps in one figure. |
| `plot_block_heatmap (line 983)` | csv_path, save_path=None | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `anchored_oas_shrinkage (line 1043)` | Sigma_full: torch.Tensor, cov_loo_all: torch.Tensor, n_samples: int | tuple of 2 values | Calculates OAS parameters ONCE on the full matrix, and applies the EXACT SAME linear shrinkage to all LOO matrices. |
| `oas_cov_torch (line 1084)` | S: torch.Tensor, N: int | Annotated: `torch.Tensor` | Apply Oracle Approximating Shrinkage (OAS) to a covariance matrix. Requires ONLY the sample covariance matrix S and sample size N. |
| `residual_rvs (line 1115)` | rv_list: list, predictor_index=0 | list of 2 values | Given a list of random variables (Torch.Tensors), returns a list where we predict the second rv using the first rv and return the residuls. |

### `Partial_Information_Decomposition/__init__.py`

File description: Python module for   init  -related project logic.

No functions or methods defined in this file.

### `Partial_Information_Decomposition/bias_functions.py`

File description: Bias correction helpers for Gaussian MI/PID estimates and permutation bias estimates.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `logdet_wishart_bias (line 21)` | df: int, d: int | Annotated: `float` | Exact finite-sample bias for log\|S\| when S is the unbiased sample covariance from Gaussian data and (df) * S ~ Wishart_d(Sigma, df). |
| `mi_wishart_bias (line 41)` | dims: list, n_samples: int | `bias_mi`; dict (bias_mi_1_t, bias_mi_2_t, bias_tri_mi, bias_mi_12) | Bias correction for Gaussian mutual information estimates from unbiased sample covariance. |
| `permuteation_debiased (line 100)` | config, term='nume' | `value` | No docstring; infer behavior from name/signature before reuse. |
| `permutation_null_debias (line 119)` | config, func | dict (bias, perm_mean, perm_std, perm_se, perm_values, n_perm); dict (debiased, perm_mean, perm_std, perm_se, perm_values, n_perm) | Debias an MI-like estimator by subtracting its permutation null floor. |
| `unique_bias (line 187)` | config, functions_dict: dict=None | `bias_dict` | No docstring; infer behavior from name/signature before reuse. |
| `bias_func (line 207)` | config, model | dict (i, h); dict (k, j) | No docstring; infer behavior from name/signature before reuse. |

### `Partial_Information_Decomposition/heatmap_plot.py`

File description: Python module for heatmap plot-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `title_case_words (line 101)` | value | `value` | Convert names like: 'idep_gaussian' -> 'Idep Gaussian' 'm7_pid' -> 'M7 PID' |
| `make_full_title (line 132)` | base_title, pid_ver_name, component_name | `f'{base_title} — {pid_ver_name} — {component_name}'`; `f'{pid_ver_name} — {component_name}'` | Build the full plot title. |
| `find_column (line 149)` | df, component_key, stat_name | None; `lower_to_original[candidate_lower]`; `lower_to_original[prefix_lower]` | Find the correct column for a component and statistic. |
| `make_p_column (line 202)` | df, p_col='p' | `df` | Create a p-like column from dx1, dx2, dt if p does not already exist. |
| `display_p_label (line 230)` | v | call `str(...)`; `f'[{', '.join(map(str, v))}]'` | Pretty display for p values on the y-axis. |
| `sort_p_index (line 244)` | values | call `sorted(...)`; `v`; call `tuple(...)` | Sort p values numerically when they are tuples like: (dx1, dx2, dt) |
| `sort_p_index.key (line 250)` | v | `v`; call `tuple(...)` | No docstring; infer behavior from name/signature before reuse. |
| `optional_pivot_table (line 258)` | df, *, index, columns, values, aggfunc, reference_index, reference_columns | None; `mat` | Build a pivot table for an optional statistic and align it with the mean matrix. |
| `plot_single_component_heatmap (line 287)` | df, *, pid_ver, component_key, base_title=None, x_col='N', y_col='p', aggfunc='last', cmap='viridis', figsize=(9, 7), save_dir=None, show=True, mean_fmt='.3f', std_fmt='.3f' | None; tuple of 2 values | Plot one heatmap for one PID version and one component. |
| `plot_pid_and_mi_heatmaps_from_csv (line 508)` | csv_path, *, base_title=None, save_dir=None, pid_versions=None, components=('red', 'unq1', 'unq2', 'syn', 'mi_x1_t', 'mi_x2_t', 'mi_x1x2_t', 'mi_m7', 'mi_m8'), x_col='N', y_col='p', seed=None, aggfunc='last', cmap='viridis', figsize=(9, 7), show=True | `figures` | Read a checkpoint CSV and create heatmaps for PID components and mutual information values. |

### `Partial_Information_Decomposition/mi_functions.py`

File description: Mutual information calculations from covariance matrices and related MI helper code.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `mi_calculation_not_whiten (line 16)` | config | Annotated: `float` | Compute MI from covariance matrices using the formula: MI = 0.5 * (log\|deno_matrix\| - log\|nume_matrix\|) |
| `safe_logdet (line 73)` | A: torch.Tensor | Annotated: `float` | Compute log determinant and raise if matrix is not positive definite. |
| `np_safe_logdet (line 91)` | A, eps=1e-08 | `val` | Stable logdet for covariance matrices. |
| `calcualte_mi (line 103)` | config, sigma_dict, term='full' | dict (mi_tri, mi_bi_1, mi_bi_2, nume, deno); dict (term) | This function calculates the tri-variate mutual information using the covariance matrices and the formula MI = 0.5 * (log\|deno_matrix\| - log\|nume_matrix\|) |
| `calculate_mi_raw (line 129)` | device: torch.device, sigma: torch.Tensor, dims: list | dict (tri_mi, bi_mi_1_t, bi_mi_2_t) | This function calculates the tri-variate or bi-variate mutual information using the covariance matrices without any whitening - in raw mode (: |
| `para_calcualte_mi (line 184)` | config, sigma_dict, term='full', assumed_whitened=True | dict (mi_tri, mi_bi_1, mi_bi_2, nume, deno); dict (term) | This function calculates the tri-variate mutual information using the for multiple covariances matrices and the formula MI = 0.5 * (log\|deno_matrix\| - log\|nume_matrix\|) |
| `calculate_mi_lr (line 215)` | config, sigma_dict | dict (mi_tri, mi_bi_1, mi_bi_2, nume, deno_1, deno_2) | This function calculates the trivarite (X1;X2,T) mutual information using the covaraince matrix especially for functions that use linear regression. The function above uses matrices that ill-conditioned using linear r... |
| `mi_wrapper (line 252)` | config, sigma_dict, whiten_terms_dict, tri_variate=True | `mi` | This function is a wrapper for the mutual information calculation functions. It takes in the config and sigma_dict and calls the appropriate function based on the mi_type argument. |
| `pid_components (line 277)` | pid_config, print_results=False | `pid_dict` | Calculate PID components with the known components. |

### `Partial_Information_Decomposition/output_utils.py`

File description: Small output/path helpers shared by PID plotting and simulation code.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `safe_filename (line 4)` | name | call `str(...)` | Return the filename exactly as given. Nothing is deleted or replaced. |

## Partial_Information_Decomposition/Idep

Idep PID estimators and Gaussian implementation classes.

### `Partial_Information_Decomposition/Idep/Idep_multivariate_gauss.py`

File description: Python module for Idep multivariate gauss-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `Idep_multivariate_gauss class (line 29)` | class | class | Class with methods listed below. |
| `Idep_multivariate_gauss.__init__ (line 30)` | self, config, rng=None, sources: Optional[list]=None, targets: Optional[list]=None, cov_matrix: Optional[torch.tensor]=None, base_e: bool=True, bias_correction: bool=False | No explicit return; likely `None` / side effects. | Initialize the Idep multivariate gaussian class |
| `Idep_multivariate_gauss.create_model_M (line 126)` | self, block1: Optional[torch.tensor]=None, block2: Optional[torch.tensor]=None, block3: Optional[torch.tensor]=None | Annotated: `torch.tensor` | This function will create the dependency matrix for the given blocks |
| `Idep_multivariate_gauss.dependency_matrix (line 150)` | self, constraints: list, cov_matrix: Optional[torch.tensor]=None, cov_dict: Optional[dict]=None | Annotated: `dict` | This function will create the dependency matrix for the given constraint |
| `Idep_multivariate_gauss.compute_Idep (line 214)` | self | Annotated: `dict` | This function calcualtes the mutual information for a given covariance matrix - U models in the lattice |
| `Idep_multivariate_gauss.pid_values (line 261)` | self, unique_1, unique_2 | `self.PID_values` | This function will compute the PID values using the I_dep values input: unique_0, unique_1 are the unique informations for source 0 and source 1 output: a dictionary with the PID values keys: 'red', 'unq1', 'unq2', 'syn' |
| `Idep_multivariate_gauss.idep (line 309)` | self, cov_matrix: Optional[torch.tensor]=None | Annotated: `dict` | This function will compute the full Idep PID decomposition |

### `Partial_Information_Decomposition/Idep/Idep_simulations.py`

File description: Python module for Idep simulations-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `pid_simulation (line 22)` | config, rng, cov, pid_ver, true_values=None | dict (unq1, unq2, syn, red, mi_tri, mi_bi_1, mi_bi_2) | Run PID simulation with know ground truth PID values from the covariance matrix sample for the true covariance matrix, calculate the PID using the specified method, and return the results along with the ground truth P... |
| `trials_simulation (line 104)` | config, title | `output_csv` | No docstring; infer behavior from name/signature before reuse. |
| `main (line 165)` | config, single=True, multi=False, exp_name=None | `output_csv` | No docstring; infer behavior from name/signature before reuse. |

### `Partial_Information_Decomposition/Idep/Idep_univariabe_gauss.py`

File description: Python module for Idep univariabe gauss-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `Idep_univariate_gauss class (line 18)` | class | class | Class with methods listed below. |
| `Idep_univariate_gauss.__init__ (line 19)` | self, sources: Optional[list]=None, targets: Optional[list]=None, cov_matrix: Optional[torch.tensor]=None | No explicit return; likely `None` / side effects. | Initialize the Idep univariate gaussian class |
| `Idep_univariate_gauss.dependency_matrix (line 46)` | self, constraints: list, cov_matrix: Optional[torch.tensor]=None, cov_dict: Optional[dict]=None | Annotated: `dict` | This function will create the dependency matrix for the given constraint |
| `Idep_univariate_gauss.compute_Idep (line 116)` | self, unique: list=[0, 1] | Annotated: `dict` | This function calcualtes the mutual information for a given covariance matrix - U models in the lattice |
| `Idep_univariate_gauss.pid_values (line 170)` | self, unique_0, unique_1 | `self.PID_values` | This function will compute the PID values using the I_dep values input: unique_0, unique_1 are the unique informations for source 0 and source 1 output: a dictionary with the PID values keys: 'red', 'unq0', 'unq1', 'syn' |
| `Idep_univariate_gauss.idep (line 197)` | self, cov_matrix: Optional[torch.tensor]=None | Annotated: `dict` | This function will compute the full Idep PID decomposition |
| `test_idep_gauss_q0_example (line 221)` | p=0.3, r=0.5, tol=1e-08 | No explicit return; likely `None` / side effects. | Test Example 1 from the paper: q = corr(X0, Y) = 0 p = corr(X0, X1) != 0 r = corr(X1, Y) != 0 |
| `check_idep_gauss_r0_example (line 271)` | p=0.3, q=0.5, tol=1e-08 | No explicit return; likely `None` / side effects. | Example 2 from the paper: r = corr(X1, Y) = 0 p = corr(X0, X1) != 0 q = corr(X0, Y) != 0 |
| `check_idep_gauss_p0_example (line 323)` | q=0.3, r=0.5, tol=1e-08 | No explicit return; likely `None` / side effects. | Example 3 from the paper: p = corr(X0, X1) = 0 q = corr(X0, Y) != 0 r = corr(X1, Y) != 0 |
| `tests (line 375)` | No inputs | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |

### `Partial_Information_Decomposition/Idep/parallel_Idep_multivariate_gauss.py`

File description: Python module for parallel Idep multivariate gauss-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `para_Idep_multivariate_gauss class (line 23)` | class | class | Class with methods listed below. |
| `para_Idep_multivariate_gauss.__init__ (line 24)` | self, N=None, df=None, device='cuda', sources: Optional[list]=None, targets: Optional[list]=None, cov_matrix: Optional[torch.tensor]=None, dims: Optional[list]=None, bias_correction: bool=False | No explicit return; likely `None` / side effects. | Initialize the Idep multivariate gaussian class |
| `para_Idep_multivariate_gauss.compute_Idep (line 110)` | self | Annotated: `dict` | This function calcualtes the mutual information for a given covariance matrix - U models in the lattice |
| `para_Idep_multivariate_gauss.pid_values (line 167)` | self, unique_1, unique_2 | `self.PID_values` | This function will compute the PID values using the I_dep values input: unique_0, unique_1 are the unique informations for source 0 and source 1 output: a dictionary with the PID values keys: 'red', 'unq1', 'unq2', 'syn' |
| `para_Idep_multivariate_gauss.idep (line 195)` | self, cov_matrix: Optional[torch.tensor]=None | Annotated: `dict` | This function will compute the full Idep PID decomposition |

## Partial_Information_Decomposition/Idep/Idep_Simulations

Simulation, covariance, shrinkage, and analysis helpers for Idep experiments.

### `Partial_Information_Decomposition/Idep/Idep_Simulations/Covariance_utils.py`

File description: Python module for Covariance utils-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `on_covariance (line 5)` | config, data | dict (cov) | This will call an intermidate function on the covariance |

### `Partial_Information_Decomposition/Idep/Idep_Simulations/Simulation_utils.py`

File description: Python module for Simulation utils-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `mean_std_csv_results (line 23)` | results_dict | tuple of 2 values | Helper: Compute mean results across seeds |
| `m7_m8_mean_std_csv_results (line 31)` | results_dict | tuple of 2 values | Helper: Compute mean results across seeds |
| `N_P_variation_simulation (line 41)` | config, simulation_func=None, mean_std_func=m7_m8_mean_std_csv_results | `all_results` | Helper: Run simulations across different N and p values and then create a heatamp of the results. |
| `to_python_scalar (line 92)` | value | `value`; call `value.numpy().tolist(...)`; call `value.item(...)`; call `value.tolist(...)` | Convert torch/numpy scalar values into normal Python scalars. |
| `flatten_pid_results (line 115)` | pid_results | `flat` | Flatten nested PID result dictionaries. |
| `get_pid_ver_csv_path (line 157)` | output_folder, pid_ver, csv_title='pid_results' | `output_folder / f'{csv_title_safe}_{pid_ver_safe}.csv'` | Return the CSV path for a specific pid_ver. |
| `append_row_to_csv (line 179)` | row, output_folder, csv_title='pid_results' | `output_csv` | Append one row to the CSV file corresponding to row["pid_ver"]. |
| `already_exists_in_csv (line 213)` | output_folder, N, p, pid_ver, seed, csv_title='pid_results' | call `mask.any(...)`; `False` | Check whether this exact simulation setting already exists in the CSV file specific to pid_ver. |
| `sample_data_from_cov (line 247)` | config, true_cov: torch.tensor, rng: np.random.Generator | Annotated: `np.ndarray` | Sample multivariate Gaussian data from the specified covariance. and return it's covariance matrix. This is a helper function for the m7_whiten bias simulation. |
| `build_m8_terms (line 271)` | config, cov_dict, whiten: bool='whiten_ver', para=False | dict (P, Q, R, Sigma) | Build the covariance matrix for M7 using the specified covariance dictionary. |
| `build_m7_terms (line 320)` | config, cov_dict, whiten: bool='whiten_ver', para=False | dict (P, Q, R, Sigma) | Build the covariance matrix for M7 using the specified covariance dictionary. |
| `extract_num_den_matrices (line 404)` | config: dict, matrix: torch.tensor | tuple of 3 values | Extract the numerator and denominator covariance matrices for M7/M8 from the full covariance matrix. assumes whitening |
| `mi_bias_calc (line 417)` | config: dict | `bias_dict` | No docstring; infer behavior from name/signature before reuse. |
| `para_nume_logdet (line 442)` | config, Sigmas: torch.Tensor | Annotated: `float` | Helper function to compute log determinant of the numerator covariance matrix. |
| `para_unique_bias_calc (line 451)` | config: dict | dict (i, k, h, j) | Helper function to compute bias for the unique information estimator. |
| `plot_heatmap_mean_std (line 500)` | results, x_col='N', y_col='p', mean_col='mean', std_col='std', emp_bias_col='emp_bias', ground_truth_col='ground_truth', var_col='var', mse_col='mse', title=None, cmap='viridis', figsize=(8, 6), save_path=None, mean_fmt='.3f', std_fmt='.3f' | `v`; call `str(...)`; call `tuple(...)` | Create a heatmap where: x-axis = N y-axis = p color = mean text = mean ± std |
| `plot_heatmap_mean_std.normalize_y (line 536)` | v | `v`; call `tuple(...)` | No docstring; infer behavior from name/signature before reuse. |
| `plot_heatmap_mean_std.display_label (line 544)` | v | call `str(...)` | No docstring; infer behavior from name/signature before reuse. |
| `corrected_statistic (line 622)` | statistics: np.ndarray, bias_correction: float | Annotated: `np.ndarray` | Apply bias correction to the raw statistics. |
| `plot_nodes_as_alpha (line 629)` | node_dict, title=None, save_path=None | No explicit return; likely `None` / side effects. | Helper function to plot the bias-corrected statistics as a function of alpha. |
| `save_nodes_results_csv (line 655)` | i_result, j_result, k_result, h_result, save_path | `df` | No docstring; infer behavior from name/signature before reuse. |
| `_to_scalar (line 676)` | value | None; call `float(...)`; call `value.item(...)` | No docstring; infer behavior from name/signature before reuse. |
| `_build_pid_rows_from_node (line 684)` | node_rows, known_component | tuple of 2 values | Build PID rows from node summaries by filling missing components. |
| `plot_pid_trajectory_vs_p_over_N (line 733)` | results, ground_truth=None, save_path=None, title='PID components vs p/N', p_col='p', n_col='N', components=None, figsize=(10, 6), dpi=300, descending_x=True | None; `traj`; `p`; call `np.sum(...)`; ... | Plot PID components and mutual information terms as a function of p/N. |
| `plot_pid_trajectory_vs_p_over_N.get_total_p (line 790)` | p | `p`; call `np.sum(...)`; call `sum(...)` | No docstring; infer behavior from name/signature before reuse. |
| `plot_pid_trajectory_vs_p_over_N.normalize_dims (line 797)` | p | None; call `tuple(...)`; `p` | No docstring; infer behavior from name/signature before reuse. |
| `CCA_reduction (line 879)` | device, rv_list: list, n_components: int=None | dict (X0, X1) | Will implement Canonical Correlation Analysis. For feature reduction. |
| `make_pre_config (line 905)` | exp, MI_config, mi0_config, above0__M7_mi_config, above0__M8_mi_config, n_p_config, unk_cfg, de_config=None | `config` | No docstring; infer behavior from name/signature before reuse. |

### `Partial_Information_Decomposition/Idep/Idep_Simulations/logdet_m7_m8.py`

File description: Python module for logdet m7 m8-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `simulate_m7_m8_log_det (line 18)` | data: list, sim_config: dict, rng: torch.Generator \| None=None | dict (M8, M7) | Inputs: data: list containing the true covariance matrices for M7 and M8 models, in the form [m7_true_cov, m8_true_cov] |
| `calculate_bias (line 200)` | config: dict, m8: bool, m7: bool, m7_wishart: bool, bias_correction: bool=True | Annotated: `list[dict]` | Run the specified simulation function over combinations of N and p values, calculating mean and std of results. |
| `simulation_wrapper (line 234)` | config: dict | Annotated: `dict` | Run the logdet bias simulation for M7 and M8 models, returning a summary of results. |
| `sort_m7_m8_results (line 257)` | results_list | tuple of 2 values | Helper: Sort results list by N and p values for sperate by m7 and m8. |

### `Partial_Information_Decomposition/Idep/Idep_Simulations/mi_m7_m8.py`

File description: Python module for mi m7 m8-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `simulate_m7_m8_mi (line 23)` | data: list, sim_config: dict, rng: torch.Generator \| None=None | dict (M8_mi, M8_nume, M8_deno, M7_mi, M7_nume, M7_deno) | Run MI simulation under the same covariance construction used in the logdet experiments. |
| `calculate_bias (line 282)` | config: dict, m8: bool=False, m8_nume: bool=False, m8_deno: bool=False, m7: bool=False, m7_deno: bool=False, m7_nume: bool=False, bias_correction: bool=True | Annotated: `dict` | Run the specified simulation function over combinations of N and p values, calculating mean and std of results. |
| `sort_m7_m8_results (line 335)` | results_list | tuple of 2 values | Helper: Sort results list by N and p values for sperate by m7 and m8. |
| `simulation_wrapper (line 358)` | config: dict | Annotated: `dict` | Run the logdet bias simulation for M7 and M8 models, returning a summary of results. |

### `Partial_Information_Decomposition/Idep/Idep_Simulations/min_test.py`

File description: Python module for min test-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `find_minimum (line 18)` | config: dict, rng: torch.Generator \| None=None | Annotated: `dict` | Run a single simulation for the M7_whiten and M8_Whiten models. |
| `plot_minimums (line 82)` | config, results_dict: dict, title: str, save_path: str \| None=None | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |

### `Partial_Information_Decomposition/Idep/Idep_Simulations/premutations_bias_corr.py`

File description: Python module for premutations bias corr-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `permutation_null_debias (line 24)` | X: ArrayLike \| tuple[ArrayLike, ...], Y: ArrayLike \| tuple[ArrayLike, ...], func: Callable[..., float], *, n_perm: int=20, random_state: int \| np.random.Generator \| None=None, **func_kwargs: Any | Annotated: `dict[str, Any]` | No docstring; infer behavior from name/signature before reuse. |
| `safe_logdet (line 85)` | A: np.ndarray, eps: float=1e-08 | Annotated: `float` | No docstring; infer behavior from name/signature before reuse. |
| `sample_cov (line 97)` | X: np.ndarray | Annotated: `np.ndarray` | No docstring; infer behavior from name/signature before reuse. |
| `gaussian_mi_logdet (line 101)` | X: np.ndarray, Y: np.ndarray, eps: float=1e-08 | Annotated: `float` | Plug-in Gaussian MI estimator: |
| `gaussian_mi_from_cov (line 121)` | Sigma: np.ndarray, dx: int | Annotated: `float` | True Gaussian MI from the population covariance. |
| `random_orthonormal_matrix (line 140)` | n: int, k: int, rng: np.random.Generator | Annotated: `np.ndarray` | Return an n x k matrix with orthonormal columns. |
| `make_population_cov (line 149)` | dx: int, dy: int, canonical_corrs: list[float], rng: np.random.Generator | Annotated: `np.ndarray` | Construct a population covariance: |
| `sample_multivariate_gaussian (line 187)` | n: int, Sigma: np.ndarray, dx: int, rng: np.random.Generator | Annotated: `tuple[np.ndarray, np.ndarray]` | Sample (X,Y) from the joint Gaussian. |
| `run_multivariate_logdet_permutation_sim (line 210)` | *, n: int=100, dx: int=15, dy: int=20, canonical_corrs: list[float]=[0.6, 0.4, 0.2], n_trials: int=100, n_perm: int=30, seed: int=0 | Annotated: `tuple[pd.DataFrame, pd.DataFrame]` | No docstring; infer behavior from name/signature before reuse. |

### `Partial_Information_Decomposition/Idep/Idep_Simulations/shrinkaging.py`

File description: Python module for shrinkaging-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `ledoit_wolf_cov (line 10)` | X | `lw.covariance_` | No docstring; infer behavior from name/signature before reuse. |
| `oracle_shrinkage_cov (line 14)` | X, assume_centered=False, return_shrinkage=False | `oas.covariance_`; tuple of 2 values | Oracle Approximating Shrinkage covariance estimator. |
| `shrunk_cov (line 41)` | X, alpha=0.1 | `sc.covariance_`; `sc_list` | No docstring; infer behavior from name/signature before reuse. |
| `shrinkage_covariance (line 54)` | X, method='ledoit_wolf', alpha=0.1 | call `ledoit_wolf_cov(...)`; call `shrunk_cov(...)` | No docstring; infer behavior from name/signature before reuse. |
| `custom_shrunk_cov (line 63)` | X, alpha=0.1, target=None, assume_centered=False, ddof=0 | `Sigma_hat` | Shrink sample covariance toward a user-supplied symmetric target matrix. |
| `shrinkage_m7_m8_simulation (line 108)` | config: dict, evluation_func: callable=None, data=None | dict (M8, M7) | This function takes true covriances and return a smaple with shrinkage covariance estimation for both M7 and M8 models. It also returns the true covariances for both models. The function can be used to evaluate the pe... |
| `evaluate_shrinkage (line 137)` | config: dict, results_dict: dict | `evaluation_results` | Will evaluate the preformance of the shrinkage covriance according to some evaluation function (e.g. Frobenius norm between the true covariance and the estimated covariance) |
| `_to_torch_float64 (line 162)` | X: TensorLike, device: str=None | Annotated: `torch.Tensor` | Convert input to torch.float64 tensor. |
| `_check_same_shape (line 176)` | A: torch.Tensor, B: torch.Tensor | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `_check_spd (line 183)` | A: torch.Tensor, name: str='matrix', eps: float=1e-12 | No explicit return; likely `None` / side effects. | Check positive definiteness via eigenvalues. |
| `covariance_frobenius_error (line 193)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None | Annotated: `float` | \|\|Sigma_hat - Sigma_true\|\|_F |
| `covariance_relative_frobenius_error (line 208)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None, eps: float=1e-12 | Annotated: `float` | \|\|Sigma_hat - Sigma_true\|\|_F / \|\|Sigma_true\|\|_F |
| `covariance_operator_error (line 226)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None | Annotated: `float` | Spectral/operator norm error: \|\|Sigma_hat - Sigma_true\|\|_2 |
| `precision_frobenius_error (line 242)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None, check_spd: bool=True | Annotated: `float` | \|\|Sigma_hat^{-1} - Sigma_true^{-1}\|\|_F |
| `precision_relative_frobenius_error (line 264)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None, check_spd: bool=True, eps: float=1e-12 | Annotated: `float` | \|\|Sigma_hat^{-1} - Sigma_true^{-1}\|\|_F / \|\|Sigma_true^{-1}\|\|_F |
| `logdet_error (line 289)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None, check_spd: bool=True | Annotated: `float` | \|log\|Sigma_hat\| - log\|Sigma_true\|\| |
| `gaussian_kl_true_to_hat (line 317)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None, check_spd: bool=True | Annotated: `float` | KL( N(0, Sigma_true) \|\| N(0, Sigma_hat) ) |
| `gaussian_kl_hat_to_true (line 349)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None, check_spd: bool=True | Annotated: `float` | KL( N(0, Sigma_hat) \|\| N(0, Sigma_true) ) |
| `gaussian_symmetric_kl (line 366)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None, check_spd: bool=True | Annotated: `float` | Symmetric KL = KL(true \|\| hat) + KL(hat \|\| true) |
| `eigenvalue_l2_error (line 390)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None | Annotated: `float` | \|\|eig(Sigma_hat) - eig(Sigma_true)\|\|_2 Uses sorted eigenvalues from eigvalsh. |
| `eigenvalue_relative_l2_error (line 408)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None, eps: float=1e-12 | Annotated: `float` | \|\|eig(Sigma_hat) - eig(Sigma_true)\|\|_2 / \|\|eig(Sigma_true)\|\|_2 |
| `evaluate_covariance_estimator (line 429)` | Sigma_true: TensorLike, Sigma_hat: TensorLike, device: str=None, check_spd: bool=True | Annotated: `Dict[str, float]` | Return all standard covariance-estimation distances in one dictionary. |

### `Partial_Information_Decomposition/Idep/Idep_Simulations/simulation_wrapper.py`

File description: Python module for simulation wrapper-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `_build_whitened_blocks_from_cov (line 16)` | S, n0, n1, n2 | tuple of 4 values | Given a covariance matrix S in block order [X0, X1, Y], return the whitened blocks P, Q, R using the same helpers as the main code. |
| `make_random_true_cov (line 30)` | config: dict, rng: torch.Generator \| None=None | tuple of 2 values | Construct a whitened Gaussian M8 covariance and its corresponding M7 covariance. |
| `simulation (line 198)` | config, functions_dict: dict, seed=None | `results_dict` | Run a simulation over combinations of N and p values, computing the specified statistic and bias correction. |
| `create_cov_m8 (line 272)` | config, P, Q, R | `m8_true_cov` | Helper to create M8 covariance from P,Q,R blocks for logging purposes |
| `create_m7_cov (line 293)` | config: dict, cov_m8, whitening_normalize: bool=True | call `torch.cat(...)` | Takes covariance of M8 and creates M7 out of M8 |

### `Partial_Information_Decomposition/Idep/Idep_Simulations/unique_m7_m8.py`

File description: Python module for unique m7 m8-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `simulate_m7_m8_idep (line 25)` | data: list, sim_config: dict, rng: torch.Generator \| None=None, intermediate_func: callable=None | dict (i, k, h, j) | Run MI simulation under the same covariance construction used in the logdet experiments. |
| `sort_m7_m8_results (line 311)` | results_list | list of 4 values | Helper: Sort results list by N and p values for sperate by m7 and m8. |
| `simulation_wrapper (line 329)` | config: dict | Annotated: `dict` | Run the logdet bias simulation for M7 and M8 models, returning a summary of results. |
| `_save_config_yaml (line 348)` | config, save_path, exp_name | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `_render_heatmaps (line 354)` | i_result, j_result, k_result, h_result, save_path, exp_name | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `_render_pid_trajectories (line 361)` | config, i_result, j_result, k_result, h_result, save_path, exp_name | None | No docstring; infer behavior from name/signature before reuse. |
| `run (line 388)` | main_func, exp_name, config, save_path, plot_heatmaps: bool=True, plot_graphs: bool=False | `nodes_results_list` | No docstring; infer behavior from name/signature before reuse. |

## Partial_Information_Decomposition/Idep/non_parametric_bias_corr

Bootstrap, jackknife, and resampling utilities for non-parametric bias correction.

### `Partial_Information_Decomposition/Idep/non_parametric_bias_corr/bootstrap.py`

File description: Python module for bootstrap-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `_get_bootstrap_count (line 25)` | config: dict | Annotated: `int` | Infer the number of bootstrap replicates from config. |
| `_to_tensor_list (line 41)` | rvs_list: list, device: str \| torch.device | Annotated: `list[torch.Tensor]` | No docstring; infer behavior from name/signature before reuse. |
| `_estimate_fitted_model_cov (line 51)` | config: dict | Annotated: `torch.Tensor` | Return the fitted covariance used by the parametric bootstrap. |
| `bootstrap_func (line 82)` | config: dict, cov_bootstrap: torch.Tensor, calculate_statistic_func: callable | `bias_dict` | Estimate parametric-bootstrap bias for a statistic. |
| `bootstrap_resample (line 135)` | config: dict | Annotated: `list` | Generate parametric-bootstrap covariance estimates. |
| `bootstrap_whiten (line 175)` | config: dict, cov_dict: dict | Annotated: `torch.Tensor` | Project batched covariance estimates onto the M7/M8 whitened model space. |

### `Partial_Information_Decomposition/Idep/non_parametric_bias_corr/jackknife.py`

File description: Python module for jackknife-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `jackkinfe_func (line 14)` | config, cov_loo, calculate_statistic_func | `bias` | Calculate the jackknife bias correction for a given statistic calculated on leave-one-out covariance matrices. |
| `jackknife_resample (line 36)` | config: dict | Annotated: `list` | Compute the full covnarice matrix across smaples and the covariance matrix of the left out ovbesrvation. Using the formula for covariance matrix Σ(-j)=N-2/(S(-j)-(1/N)*s(-j)s(-j)T) Where S(-j)=S-ZjZjT and s(-j)=s-Zj |
| `jackknife_whiten (line 99)` | config, m7_cov_dict | `cov_whiten` | No docstring; infer behavior from name/signature before reuse. |

### `Partial_Information_Decomposition/Idep/non_parametric_bias_corr/resampling_wrapper.py`

File description: Python module for resampling wrapper-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `resampleing (line 20)` | resample_inputs: dict, rng | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `calculate_statistic (line 34)` | config: dict, calc_func: callable, population: dict, rng: np.random.Generator | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `calculate_bias (line 42)` | config: dict, statistic_dict: dict, bias_func: callable | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `bias_resampling (line 51)` | config: dict, calc_func: callable=None | Annotated: `dict` | This function will calculate the statistics value and it's and will return a dictionary with the following keys: |

## encoding_model

Encoding model, commonality analysis, regression, and prediction pipeline utilities.

### `encoding_model/__init__.py`

File description: Python module for   init  -related project logic.

No functions or methods defined in this file.

### `encoding_model/algoanut_data.py`

File description: Python module for algoanut data-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `argObj class (line 26)` | class | class | Class with methods listed below. |
| `argObj.__init__ (line 27)` | self, data_dir, parent_submission_dir, subj | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `load_data_algonauts (line 41)` | paths_dict, args=None, subj=1, plot_fmri=False | tuple of 2 values | Load fMRI data and image file lists for a given subject. |

### `encoding_model/commonality.py`

File description: Shared commonality analysis utilities.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `_ensure_2d (line 13)` | features | `features` | Raise value error if features are not 2D: (n_samples, n_features). If features are 1D, raise an error with instructions to reshape. |
| `_score_only (line 22)` | score_result | `score_result`; `score_result[1]` | No docstring; infer behavior from name/signature before reuse. |
| `commonality_analysis (line 28)` | features_X1, features_X2, target, method='standard', alphas=None, scale_by_target_variance=False | dict (R²_X1, R²_X2, R²_X12, unique_X1, unique_X2, common, unexplained) | Decompose predictive power into unique, common, and unexplained components. |

### `encoding_model/encoding_utils.py`

File description: Python module for encoding utils-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `ImageDataset class (line 23)` | class | class | Class with methods listed below. |
| `ImageDataset.__init__ (line 24)` | self, imgs_paths, idxs, transform | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `ImageDataset.__len__ (line 31)` | self | call `len(...)` | No docstring; infer behavior from name/signature before reuse. |
| `ImageDataset.__getitem__ (line 34)` | self, idx | `img` | No docstring; infer behavior from name/signature before reuse. |
| `plot_fmri (line 44)` | path, args, hemi, title='' | No explicit return; likely `None` / side effects. | Plot fMRI data on a brain surface and save the figure. |
| `fmri_response_image (line 76)` | path, args, hemisphere, img_idx, train_img_dir, train_img_list, lh_fmri, rh_fmri | No explicit return; likely `None` / side effects. | This function outputs the fmri response that matches the image shown. accoring to the NSD dataset structure. |
| `split_dataset (line 128)` | train_img_list, test_img_list, rand_seed=5, train_p=90 | tuple of 2 values; tuple of 3 values | No docstring; infer behavior from name/signature before reuse. |
| `fmri_data_loader (line 153)` | lh_fmri, rh_fmri, train_img_list, test_img_list, train_img_dir, test_img_dir, batch_size=500, train_p=90 | tuple of 3 values | No docstring; infer behavior from name/signature before reuse. |
| `map_correlation_to_rois (line 203)` | args, lh_correlation, rh_correlation, hemisphere | No explicit return; likely `None` / side effects. | Map correlation values to ROIs. |
| `roi_fmri_data (line 240)` | args, lh_fmri, rh_fmri | tuple of 3 values | Map fMRI data to ROIs. |
| `get_specific_roi_fmri (line 294)` | args, lh_fmri, rh_fmri, roi_name | tuple of 2 values | Get fMRI data for a specific ROI. |
| `visualize_encdoing_accuaracy (line 312)` | args, lh_correlation, rh_correlation, correlation_path, plot=True | tuple of 3 values | Visualize encoding accuracy with a bar graph and return ROI correlation values and ROI names for left and right hemispheres for a given subject. |
| `save_corellation (line 396)` | roi_names, lh_correlation, rh_correlation, correlation_path, experiment_name | No explicit return; likely `None` / side effects. | Save correlation values to .npy files. |
| `save_model (line 425)` | folder_path, model_name, save_dict, reg_lh: Optional[ndarray]=None, reg_rh: Optional[ndarray]=None, features_val_pred_lh: Optional[List]=None, features_val_pred_rh: Optional[List]=None, features_train: Optional[ndarray]=None, features_val_trained: Optional[ndarray]=None, predict_array: Optional[ndarray]=None, roi_names: Optional[List]=None, lh_correlation: Optional[ndarray]=None, rh_correlation: Optional[ndarray]=None | `models_folder` | Save the trained encoding model. with its corellation values and roi names and figs` |

### `encoding_model/fmri_model.py`

File description: Python module for fmri model-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `encoding_model class (line 14)` | class | class | Class with methods listed below. |
| `encoding_model.__init__ (line 15)` | self, device, model: str='alexnet', model_layer: str='features.2', model_path: str='pytorch/vision:v0.10.0', features: Optional[np.ndarray]=None, n_features=None | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `encoding_model.fit_pca (line 40)` | self, dataloader, batch_size=100, ncomponents=100 | `pca` | No docstring; infer behavior from name/signature before reuse. |
| `encoding_model.extract_features (line 51)` | self, dataloader, pca | call `np.vstack(...)` | No docstring; infer behavior from name/signature before reuse. |
| `encoding_model.train (line 61)` | self, train_data_loader, lh_fmri_train, rh_fmri_train, features_train: Optional[np.ndarray]=None, alphas: Optional[np.ndarray]=None | tuple of 2 values | No docstring; infer behavior from name/signature before reuse. |
| `encoding_model.validate (line 82)` | self, reg_lh, reg_rh, lh_fmri_val, rh_fmri_val, features_val | tuple of 2 values | "This funciton validates the encoding model on the validation set and returns the correlation scores for each hemisphere. |
| `encoding_model.run_model (line 110)` | self, train_imgs_dataloader, val_imgs_dataloader, lh_fmri_train, rh_fmri_train, lh_fmri_val, rh_fmri_val, batch_size=100, ncomponents=None | tuple of 4 values | This function runs the entire encoding model pipeline: feature extraction, training, validation without testing. |

### `encoding_model/grid_search.py`

File description: Python module for grid search-related project logic.

No functions or methods defined in this file.

### `encoding_model/pred_pipeline.py`

File description: Python module for pred pipeline-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `pipeline (line 20)` | data_dir, parent_submission_dir, subj, args, layer_name='features.2', model=None, features=None, only_validate=False, train_p=80, data_fmri=None, data_imgs=None | `diction` | Main pipeline to run the encoding model on Algonauts data for a given subject. |
| `trained_model (line 99)` | layer_name, model, model_name, train_p | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `just_validate (line 128)` | layer_name, model | `models_folder` | No docstring; infer behavior from name/signature before reuse. |

### `encoding_model/regression_metrics.py`

File description: Shared regression scoring helpers for encoding and toy examples.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `compute_ols_cv_r2 (line 6)` | X, y, return_model=False | `ridge_cv.best_score_`; tuple of 2 values | Compute cross-validated R2 using leave-one-out cross-validation. |
| `compute_ridge_cv_r2 (line 32)` | X, y, alphas=None, return_model=False | `ridge_cv.best_score_`; tuple of 2 values | Compute cross-validated R2 using RidgeCV with efficient LOO cross-validation. |
| `compute_r2 (line 64)` | X, y, return_model=False | `score`; tuple of 2 values | Compute in-sample R2 for OLS regression. |
| `compute_lasso_cv_r2 (line 87)` | X, y | tuple of 2 values | Compute in-sample R2 after fitting multi-output LassoCV. |

### `encoding_model/suppresion_model.py`

File description: Python module for suppresion model-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `real_model_func (line 52)` | model, layer_name, model_path, batch_size, ncomponents, train_data_loader | tuple of 5 values | Create and train the real encoding model using fMRI data and image features. |
| `train_save_or_load (line 81)` | folder_path=None, model_name=None, path_to_load=None | `trained_real_model` | Load a trained encoding model from disk. |
| `main (line 100)` | dict, suppression_strength=0.5, rng_seed=0 | No explicit return; likely `None` / side effects. | Run the 2x3 factorial experiment design. |
| `test_run (line 164)` | run_name, save_dir, features, fmri_dict, rng_seeds, suppression_method, suppression_strength=[0.5], n_samples=[1000], n_features=[100], snr=[1.0], mixing_dimension=[None] | None | No docstring; infer behavior from name/signature before reuse. |

### `encoding_model/suppression_core.py`

File description: Python module for suppression core-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `create_predictions (line 12)` | reg_lh, reg_rh, features | tuple of 2 values | Create fMRI predictions using trained regression models. |
| `create_encoder (line 28)` | rng, features, target, n_features | tuple of 2 values | Create and train a linear regression encoder. mostly usable when the number of model features is larger than the number of samples. there for we randomly select a subset of features to use for training. |
| `permutate_models (line 59)` | rng, features, suppression_strength | tuple of 2 values | No docstring; infer behavior from name/signature before reuse. |
| `noise_component (line 80)` | rng, features, suppression_strength, permutation | tuple of 2 values | Create suppresion model using only noise component. |
| `create_supression_model (line 106)` | rng, signal, suppresion_method, features, suppression_strength=0.5, snr=1.0, mixing_dimension=None | tuple of 3 values | Create suppression model features X_M1 and X_M2 based on the given parameters. |
| `run_all_methods (line 146)` | rng_seed, suppresion_method, mixing_dimension, snr, suppression_strength, models_and_features_dict=None | `methods_outputs` | No docstring; infer behavior from name/signature before reuse. |
| `suppression_analysis_pipeline (line 164)` | features, reg_lh=None, reg_rh=None, hemisphere='both', suppression_strength=0.5, snr=1.0, mixing_dimension=None, suppresion_method='permutate', analysis_methods=['standard', 'ols_cv', 'ridge_cv'], rng_seed=None, alphas=None | `pipeline_results` | Complete pipeline that takes model features, creates predictions via regression, generates suppression models, and performs commonality analysis. |
| `grid_search_suppression_analysis (line 306)` | features, reg_lh=None, reg_rh=None, suppression_strength_list=None, snr_list=None, mixing_dimension_list=None, rng_seed_list=None, hemisphere='both', suppresion_method='permutate', output_dir='./grid_search_results', grid_name='NoName', verbose=True | dict (results_df, results_by_seed, file_paths, output_dir) | Perform a grid search over suppression analysis parameters using ridge regression. |

## data

Data-specific loading/parsing scripts.

### `data/FBA-1.py`

File description: Python module for FBA-1-related project logic.

No functions or methods defined in this file.

### `data/OTC.py`

File description: Python module for OTC-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `load_OTC (line 8)` | subject_id: int, path_to_data: str | Annotated: `dict` | Load OTC fMRI data for a given subject. (This function assumes data files are zarr files stored in the specified path.) |
| `main (line 30)` | No inputs | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |

### `data/V1.py`

File description: Python module for V1-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `load_roi_data (line 20)` | args, roi_name='V1v' | tuple of 4 values | No docstring; infer behavior from name/signature before reuse. |
| `roi_encoding_model (line 38)` | train_data_loader, val_data_loader, lh_roi_fmri_train, rh_roi_fmri_train, lh_roi_fmri_val, rh_roi_fmri_val, layer_name='features.2', model=None, features=None, batch_size=500, ncomponents=None | `output_dict` | No docstring; infer behavior from name/signature before reuse. |
| `main (line 67)` | No inputs | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |

## library_wrappers

CLI and Python wrappers around external PID/R implementations.

### `library_wrappers/Delta_PID.py`

File description: Wrapper for the gpid Gaussian delta PID definition.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `parse_sizes (line 41)` | value: str | Annotated: `tuple[int, int, int]` | No docstring; infer behavior from name/signature before reuse. |
| `parse_args (line 48)` | No inputs | Annotated: `argparse.Namespace` | No docstring; infer behavior from name/signature before reuse. |
| `simple_example_args (line 64)` | No inputs | Annotated: `argparse.Namespace` | Small debug example: source1 and source2 are noisy copies of one target. |
| `main (line 75)` | No inputs | Annotated: `int` | No docstring; infer behavior from name/signature before reuse. |

### `library_wrappers/Flow_PID.py`

File description: Wrapper for warrenzha/flow-pid's normalizing-flow PID estimator.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `parse_args (line 36)` | No inputs | Annotated: `argparse.Namespace` | Parse CLI arguments for Flow-PID on raw sample matrices. |
| `simple_example_args (line 77)` | No inputs | Annotated: `argparse.Namespace` | Small debug example: train Flow-PID on samples from the shared Gaussian case. |
| `load_flow_pid (line 99)` | No inputs | `flow_pid_module.flow_pid` | Load and return flow-pid's original ``flow_pid`` function. |
| `read_samples (line 131)` | path: Path, expected_columns: int \| None=None | Annotated: `np.ndarray` | Read a two-dimensional sample CSV with rows as observations. |
| `split_combined_samples (line 145)` | samples: np.ndarray, sizes: tuple[int, int, int] | Annotated: `tuple[np.ndarray, np.ndarray, np.ndarray]` | Split [source1, source2, target] samples into flow-pid's (target, source1, source2). |
| `load_input_arrays (line 157)` | args: argparse.Namespace | Annotated: `tuple[np.ndarray, np.ndarray, np.ndarray]` | Load target/M, source1/X, and source2/Y sample arrays. |
| `validate_training_args (line 183)` | args: argparse.Namespace | Annotated: `None` | Validate Flow-PID training hyperparameters. |
| `main (line 195)` | No inputs | Annotated: `int` | Load raw samples, train flow-pid's estimator, and save PID components. |

### `library_wrappers/IG_R.py`

File description: Small Python wrapper for JWKay/PID/IGFuns.R.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `parse_correlation (line 149)` | value: str | Annotated: `float` | No docstring; infer behavior from name/signature before reuse. |
| `parse_args (line 159)` | No inputs | Annotated: `argparse.Namespace` | No docstring; infer behavior from name/signature before reuse. |
| `simple_example_args (line 182)` | No inputs | Annotated: `argparse.Namespace` | Small debug example: run IG on the shared 1D Gaussian covariance. |
| `apply_example_defaults (line 203)` | args: argparse.Namespace | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `set_evil_twin_inputs (line 224)` | args: argparse.Namespace | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `set_simple_gaussian_inputs (line 237)` | args: argparse.Namespace, matrix_csv: Path | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `ig_source_candidates (line 250)` | pid_repo: Path \| None | Annotated: `list[Path]` | No docstring; infer behavior from name/signature before reuse. |
| `find_ig_source (line 254)` | args: argparse.Namespace | Annotated: `Path` | No docstring; infer behavior from name/signature before reuse. |
| `require_shape (line 270)` | path: Path, expected: tuple[int, int], label: str | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `validate_inputs (line 278)` | args: argparse.Namespace | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `run_r (line 314)` | args: argparse.Namespace | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `write_scalar_csv (line 376)` | path: Path, value: float | Annotated: `Path` | No docstring; infer behavior from name/signature before reuse. |
| `add_standard_table (line 381)` | result: dict | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `write_result (line 401)` | result: dict, output: Path \| None | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `print_result (line 411)` | result: dict | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `main (line 418)` | No inputs | Annotated: `int` | No docstring; infer behavior from name/signature before reuse. |

### `library_wrappers/Idep_R.py`

File description: Small Python wrapper for JWKay/PID/IdepGauss.R.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `parse_args (line 153)` | No inputs | Annotated: `argparse.Namespace` | No docstring; infer behavior from name/signature before reuse. |
| `csv_shape (line 171)` | path: Path | Annotated: `tuple[int, int]` | No docstring; infer behavior from name/signature before reuse. |
| `validate_matrix_args (line 190)` | args: argparse.Namespace | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `absolute_path (line 203)` | value: str | Annotated: `Path` | No docstring; infer behavior from name/signature before reuse. |
| `main (line 210)` | No inputs | Annotated: `int` | No docstring; infer behavior from name/signature before reuse. |

### `library_wrappers/Thin_PID.py`

File description: Wrapper for warrenzha/flow-pid's exact Gaussian Thin-PID definition.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `parse_args (line 31)` | No inputs | Annotated: `argparse.Namespace` | Parse CLI arguments for running Thin-PID on a covariance/correlation CSV. |
| `simple_example_args (line 56)` | No inputs | Annotated: `argparse.Namespace` | Small debug example: source1 and source2 are noisy copies of one target. |
| `load_exact_gauss_thin_pid (line 69)` | No inputs | `thin_pid.exact_gauss_thin_pid` | Load and return flow-pid's original ``exact_gauss_thin_pid`` function. |
| `main (line 99)` | No inputs | Annotated: `int` | Load input, run flow-pid Thin-PID, and save the result CSV. |

### `library_wrappers/Tilde_PID.py`

File description: Wrapper for the gpid Gaussian tilde PID definition.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `parse_sizes (line 42)` | value: str | Annotated: `tuple[int, int, int]` | No docstring; infer behavior from name/signature before reuse. |
| `parse_args (line 49)` | No inputs | Annotated: `argparse.Namespace` | No docstring; infer behavior from name/signature before reuse. |
| `simple_example_args (line 66)` | No inputs | Annotated: `argparse.Namespace` | Small debug example: source1 and source2 are noisy copies of one target. |
| `main (line 78)` | No inputs | Annotated: `int` | No docstring; infer behavior from name/signature before reuse. |

### `library_wrappers/check_evil_twin_all.py`

File description: Run the IG and Idep evil-twin checks and combine their PID tables.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `parse_args (line 39)` | No inputs | Annotated: `argparse.Namespace` | No docstring; infer behavior from name/signature before reuse. |
| `run (line 49)` | command: list[str], label: str, verbose: bool | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `require_file (line 70)` | path: Path, label: str | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `load_ig_rows (line 75)` | path: Path | Annotated: `list[dict[str, str]]` | No docstring; infer behavior from name/signature before reuse. |
| `load_idep_rows (line 80)` | path: Path | Annotated: `list[dict[str, str]]` | No docstring; infer behavior from name/signature before reuse. |
| `load_single_row_csv (line 107)` | path: Path | Annotated: `list[dict[str, str]]` | No docstring; infer behavior from name/signature before reuse. |
| `normalize_rows (line 115)` | rows: list[dict] | Annotated: `list[dict[str, str]]` | No docstring; infer behavior from name/signature before reuse. |
| `write_csv (line 130)` | rows: list[dict[str, str]], path: Path | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `write_svg (line 138)` | rows: list[dict[str, str]], path: Path | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `write_svg.esc (line 161)` | value: object | Annotated: `str` | No docstring; infer behavior from name/signature before reuse. |
| `print_table (line 205)` | rows: list[dict[str, str]] | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `main (line 220)` | No inputs | Annotated: `int` | No docstring; infer behavior from name/signature before reuse. |

### `library_wrappers/compare_gpid_canonical.py`

File description: Compare GPID canonical examples by direct GPID calls and PID_calc wrappers.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `canonical_examples (line 38)` | No inputs | Annotated: `list[dict[str, object]]` | Build the canonical covariance examples from external/gpid/scripts. |
| `canonical_examples.add (line 51)` | desc: str, case_id: int, cov: np.ndarray, sigma: float=np.nan, rho: float=np.nan | Annotated: `None` | Append one canonical example to the local case list. |
| `unpack_gpid (line 92)` | values: tuple[float, ...] | Annotated: `dict[str, float]` | Convert one GPID return tuple into named components. |
| `unpack_wrapper (line 106)` | pid: dict[str, float], mi: dict[str, float] | Annotated: `dict[str, float]` | Convert one PID_calc wrapper output into named components. |
| `pid_calc (line 120)` | method: str, cov: np.ndarray, dm: int, dx: int, dy: int | Annotated: `dict[str, float]` | Run one canonical covariance through the matching PID_calc wrapper. |
| `compare (line 140)` | No inputs | Annotated: `list[dict[str, object]]` | Compare direct GPID calls with PID_calc wrappers for all canonical examples. |
| `print_summary (line 174)` | rows: list[dict[str, object]] | Annotated: `None` | Print the maximum absolute difference per method and component. |
| `main (line 193)` | No inputs | Annotated: `int` | Run the canonical GPID comparison command. |

### `library_wrappers/r_idep_client.py`

File description: Programmatic Python client for JWKay/PID/IdepGauss.R.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `RIdePResult class (line 93)` | class | class | Class with methods listed below. |
| `_to_2d_float_rows (line 100)` | matrix: Any | Annotated: `list[list[float]]` | No docstring; infer behavior from name/signature before reuse. |
| `_write_matrix_csv (line 117)` | path: Path, rows: Sequence[Sequence[float]] | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `_read_result_csv (line 123)` | path: Path, stdout: str, stderr: str, *, bits_to_nats: bool | Annotated: `RIdePResult` | No docstring; infer behavior from name/signature before reuse. |
| `run_idep_from_covariance (line 148)` | sigma: Any, sizes: Sequence[int], *, rscript: str \| Path \| None=None, idep_url: str=DEFAULT_IDEP_URL, local_idep: str \| Path \| None='IdepGauss.R', bits_to_nats: bool=True, keep_temp: bool=False | Annotated: `RIdePResult` | Run R idepGM(sizes, sigma) and return named Idep/MMI atoms. |
| `run_idep_for_cases (line 220)` | case_covariances: Mapping[str, Any], sizes: Sequence[int], **kwargs: Any | Annotated: `dict[str, RIdePResult]` | Run R Idep/MMI for several named covariance matrices. |
| `atoms_as_ordered_values (line 232)` | values: Mapping[str, float] | Annotated: `list[float]` | No docstring; infer behavior from name/signature before reuse. |
| `_resolve_local_idep (line 236)` | local_idep: str \| Path \| None | Annotated: `str` | No docstring; infer behavior from name/signature before reuse. |

### `library_wrappers/wrapper_utils.py`

File description: Shared utilities for the Python PID covariance wrappers.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `add_gpid_src_to_path (line 55)` | repo_root: Path \| None=None | Annotated: `Path` | Put external/gpid/src on sys.path for script-style wrapper imports. |
| `parse_sizes (line 66)` | value: str | Annotated: `tuple[int, int, int]` | Parse source1,source2,target dimensions and reject invalid values. |
| `read_matrix (line 77)` | path: Path | Annotated: `np.ndarray` | Read a square covariance or correlation matrix from a CSV file. |
| `validate_covariance (line 87)` | matrix: np.ndarray, expected_shape: tuple[int, int] | Annotated: `None` | Validate shape and basic covariance/correlation symmetry. |
| `source_source_target_to_target_source_source (line 95)` | matrix: np.ndarray, sizes: tuple[int, int, int] | Annotated: `np.ndarray` | Reorder a [source1, source2, target] matrix to [target, source1, source2]. |
| `write_simple_gaussian_covariance (line 104)` | path: Path | Annotated: `None` | Write the shared simple [source1, source2, target] Gaussian covariance example. |
| `simple_gaussian_samples (line 110)` | num_samples: int, seed: int | Annotated: `tuple[np.ndarray, np.ndarray, np.ndarray]` | Generate raw samples from the shared simple Gaussian example. |
| `covariance_example_context (line 130)` | args: argparse.Namespace | None | Temporarily attach the shared simple covariance example to wrapper args. |
| `pid_result_row (line 145)` | values: tuple[float, ...], case: str, pid_definition: str, *, include_union_objective: bool=False | Annotated: `dict[str, object]` | Convert a Gaussian PID tuple into the local one-row CSV schema. |
| `write_pid_row (line 172)` | row: dict[str, object], path: Path, columns: list[str] | Annotated: `None` | Write one standardized PID result row to a CSV file. |
| `print_pid_result (line 181)` | row: dict[str, object], columns: list[str] | Annotated: `None` | Print one PID result in a compact debug table. |
| `load_module (line 213)` | module_name: str, path: Path | Annotated: `types.ModuleType` | Load one source file as a module without importing package __init__ files. |
| `run_covariance_pid_wrapper (line 224)` | args: argparse.Namespace, solver: Callable[..., tuple[float, ...]] \| None=None, *, pid_definition: str, columns: list[str], solver_loader: Callable[[], Callable[..., tuple[float, ...]]] \| None=None, solver_kwargs: dict[str, object] \| None=None, include_union_objective: bool=False, verbose_history: bool=False, read_message: bool=False, call_message: str \| None=None, written_message: str \| None=None | Annotated: `int` | Run the common covariance-wrapper flow and write a one-row PID CSV. |

## Simulations/Encoder_simulation

Encoder-based simulation scripts for unique/shared information examples.

### `Simulations/Encoder_simulation/Both_Unique_Encodrs.py`

File description: Python module for Both Unique Encodrs-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `get_run_config (line 20)` | No inputs | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `creature_featurs (line 46)` | rng, snr, unique_ratio, features, signal, redundant_dim=None | tuple of 3 values | No docstring; infer behavior from name/signature before reuse. |
| `run_single_seed (line 76)` | seed: int, config: dict, features: torch.Tensor, fmri_dict: dict | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `main (line 105)` | No inputs | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |

### `Simulations/Encoder_simulation/both_unique.py`

File description: Python module for both unique-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `get_run_config (line 20)` | No inputs | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `half_permute (line 36)` | rng, features, snr=10 | tuple of 2 values | No docstring; infer behavior from name/signature before reuse. |
| `orthogonal_vectors (line 57)` | rng, n, p, features, noise=None, singal=None, unique_ratio=None, function=None | tuple of 3 values | No docstring; infer behavior from name/signature before reuse. |
| `feature_creation (line 98)` | rng, unique_ratio, unique_method='orthogonal', n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', show_diagnostic_plots=False | tuple of 3 values | Creates dummy predictors and a target |
| `test_both_unique (line 139)` | rng, unique_ratio, n=1024, p=100, snr=10.0, method='standard' | tuple of 3 values | No docstring; infer behavior from name/signature before reuse. |
| `run_single_seed (line 150)` | seed: int, config: dict | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `main (line 164)` | No inputs | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |

### `Simulations/Encoder_simulation/turned_off_unqiue.py`

File description: Python module for turned off unqiue-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `get_run_config (line 28)` | No inputs | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `feature_creation (line 48)` | rng, r_str, u1_str, u2_str, unique_method='orthogonal', n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', show_diagnostic_plots=False | tuple of 3 values | Creates dummy predictors and a target |
| `test (line 86)` | rng, r_str, u1_str, u2_str, n=1024, p=100, snr=10.0, method='standard' | tuple of 4 values | No docstring; infer behavior from name/signature before reuse. |
| `extract_betas (line 101)` | ca_results | dict (X1_betas, X2_betas, X12_betas, X12_1_betas, X12_2_betas) | No docstring; infer behavior from name/signature before reuse. |
| `run_single_seed (line 120)` | seed: int, config: dict | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `test_regularization_term (line 140)` | seed: int, config: dict | `alpha_results` | No docstring; infer behavior from name/signature before reuse. |
| `save_term_results_csv (line 169)` | x_axis: str, term_results: dict, output_csv_path: str \| Path | Annotated: `Path` | No docstring; infer behavior from name/signature before reuse. |
| `test_u2str (line 195)` | seed: int, config: dict, final_ratio: float | `results` | No docstring; infer behavior from name/signature before reuse. |
| `plot_keys_vs_alpha (line 226)` | csv_path: Union[str, Path], keys: Sequence[str], *, x_col: str='alpha', sort_alpha: bool=False, logx: bool=False, figsize: tuple[float, float]=(8, 4.5), marker: Optional[str]=None, save_path: Optional[Union[str, Path]]=None | Annotated: `None` | Plot selected columns (keys) vs x_col from a CSV file. |
| `main (line 290)` | No inputs | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |

## Simulations/Theoretical_Examples/Covariance

Theoretical covariance examples, sampling, and result utilities.

### `Simulations/Theoretical_Examples/Covariance/cov_functions.py`

File description: Python module for cov functions-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `make_random_true_cov (line 20)` | config: dict, rng: torch.Generator \| None=None | Annotated: `np.ndarray` | Construct a generic positive-definite Gaussian covariance. |
| `rectangular_identity (line 85)` | d_a: int, d_b: int, dtype: torch.dtype=torch.float64, device: str \| torch.device='cpu' | Annotated: `torch.Tensor` | Create a rectangular identity-like matrix of shape (d_a, d_b). |
| `make_direct_true_cov_from_config (line 109)` | config: dict, dtype: torch.dtype=torch.float64, eps: float=1e-10 | Annotated: `torch.Tensor` | Create an interpretable covariance matrix for [X1, X2, T] directly from a merged config dictionary. |
| `sample_from_cov (line 183)` | config, true_cov: torch.Tensor, n_samples: int, rng: torch.Generator | Annotated: `torch.Tensor` | Sample from a Gaussian distribution with the given covariance. |
| `change_covariance_order (line 212)` | cov: torch.Tensor, new_order: list[int], dims: list[int] | Annotated: `torch.Tensor` | Permute the covariance matrix to change the order of variables. |

### `Simulations/Theoretical_Examples/Covariance/sample_simulation.py`

File description: Python module for sample simulation-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `load_config (line 25)` | config_path: str \| Path=DEFAULT_CONFIG_PATH | Annotated: `dict` | Load configuration from YAML file. |
| `csv_save (line 31)` | config: dict, experiment_name: str, method: str, theoretical_values: tuple, sampled_values: dict | Annotated: `Path` | Save theoretical and sampled PID/MI component values to one method CSV. |
| `simulation (line 68)` | config: dict, methods: list, experiment_name: str \| None=None | Annotated: `dict` | Run theoretical-covariance PID calculations and sampled trial summaries. |

### `Simulations/Theoretical_Examples/Covariance/save_results.py`

File description: Python module for save results-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `save_sample_simulation_results_table (line 15)` | results: dict, config: dict, save_path: str \| Path, decimals: int=4, title: str='PID Method Comparison', dpi: int=200 | Annotated: `Path` | Save sample-simulation PID/MI summaries as a styled table image. |

## Simulations/Theoretical_Examples/RVs_Story

Random-variable story examples, truth helpers, batching, and Flow-PID grid-search tooling.

### `Simulations/Theoretical_Examples/RVs_Story/flow_pid_grid_search.py`

File description: Python module for flow pid grid search-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `load_functions (line 60)` | path, names, namespace | `{name: namespace[name] for name in names}` | Load only selected function definitions from a file, avoiding top-level imports. |
| `load_example_and_truth (line 70)` | example_name | tuple of 2 values | No docstring; infer behavior from name/signature before reuse. |
| `grid_items (line 92)` | grid | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `standardize_train_val (line 98)` | train, val | tuple of 2 values | No docstring; infer behavior from name/signature before reuse. |
| `make_folds (line 107)` | n, k_folds, seed | `[fold for fold in np.array_split(indices, k_folds) if len(fold)]` | No docstring; infer behavior from name/signature before reuse. |
| `true_synergy (line 115)` | truth_func, x1, x2, t | call `float(...)` | No docstring; infer behavior from name/signature before reuse. |
| `write_csv (line 122)` | path, rows, fieldnames | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `run_grid_search (line 130)` | config, example_name, k_folds, grid, results_dir, device | `best` | No docstring; infer behavior from name/signature before reuse. |
| `parse_args (line 252)` | No inputs | call `parser.parse_args(...)` | No docstring; infer behavior from name/signature before reuse. |
| `main (line 267)` | No inputs | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |

### `Simulations/Theoretical_Examples/RVs_Story/story_batch_utils.py`

File description: Seed-loop and CSV helpers for RVs_Story example batches.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `_as_float (line 26)` | value | Annotated: `float` | Convert scalar numeric values from tensors or Python numbers to float. |
| `csv_path (line 38)` | results_dir: Path, example: str, method: str | Annotated: `Path` | Build the per-example, per-method seed CSV path. |
| `csv_has_seed (line 54)` | path: Path, seed: int | Annotated: `bool` | Check whether a seed is already present in a seed CSV. |
| `seed_is_done (line 70)` | seed: int, example_names: list[str], results_dir: Path, methods=PID_METHODS | Annotated: `bool` | Check whether all expected example/method CSV rows exist for a seed. |
| `loop_examples (line 89)` | config: dict, functions_to_run: list[Callable], example_names: list[str], main_func: Callable, save_image: bool=True | Annotated: `dict` | Run a list of RV examples once with one config. |
| `save_seed_csvs (line 121)` | seed: int, all_results: dict, results_dir: Path | Annotated: `None` | Save one seed of PID results into method-specific CSV files. |
| `mean_results_from_csvs (line 150)` | results_dir: Path, example: str, seeds: list[int] | Annotated: `dict` | Average saved seed CSVs back into a PID result dictionary. |
| `loop_examples_over_seeds (line 178)` | config: dict, functions_to_run: list[Callable], example_names: list[str], main_func: Callable, num_seeds: int \| None=None, seeds: list[int] \| None=None | Annotated: `dict` | Run examples over seeds, save seed CSVs, then save averaged figures. |

### `Simulations/Theoretical_Examples/RVs_Story/story_math_utils.py`

File description: Small adapters around shared MI and bias helpers for RVs_Story truth rows.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `calculate_story_mi_values (line 19)` | sources: list[torch.Tensor], target: list[torch.Tensor] | Annotated: `tuple[dict, dict]` | Calculate raw Gaussian MI values and legacy Wishart bias values. |

### `Simulations/Theoretical_Examples/RVs_Story/story_pid_utils.py`

File description: Shared PID execution helpers for the RVs_Story examples.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `truth_pid_suppression (line 29)` | sources: list[torch.Tensor], target: list[torch.Tensor], covariance=None | Annotated: `tuple[dict, dict]` | Compute the Gaussian truth row for suppression-style examples. |
| `truth_pid_equal_unique (line 52)` | sources: list[torch.Tensor], target: list[torch.Tensor], covariance=None | Annotated: `tuple[dict, dict]` | Compute the Gaussian truth row for equal-unique regular examples. |
| `run_pid_story (line 80)` | config: dict, function_to_run: Callable, truth_func: Callable \| None=None, methods: tuple[str, ...]=('tilde', 'delta', 'flow') | Annotated: `dict` | Run one RVs_Story generator through selected PID methods. |
| `load_story_config (line 132)` | config_path: str \| Path \| None=None | Annotated: `dict` | Load the RVs_Story YAML configuration. |
| `save_single_example (line 146)` | config: dict, function_to_run: Callable, output_name: str, truth_func: Callable \| None=None | Annotated: `dict` | Run one example and save its PID comparison figure. |

## Simulations/Theoretical_Examples/RVs_Story/regular_examples

Regular theoretical examples, including equal-unique source examples.

### `Simulations/Theoretical_Examples/RVs_Story/regular_examples/All_above_zero.py`

File description: Regular examples with balanced unique information.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `all_above_zero_weighted (line 15)` | rng, n, p, noise_std, unique1_weight=5.0, unique2_weight=5.0, redundant_weight=1.0, shared_noise_weight=1.0 | tuple of 3 values | Gaussian example where all PID components should be above zero, but unique information is emphasized. |
| `con_all_above_zero_weighted (line 71)` | rng, n, p, noise_std, unique1_weight=5.0, unique2_weight=5.0, redundant_weight=1.0, shared_noise_weight=1.0 | tuple of 3 values | Concatenated Gaussian example where all PID components should be above zero, but unique information is emphasized. |

### `Simulations/Theoretical_Examples/RVs_Story/regular_examples/all_reg_examples.py`

File description: Run all regular RVs_Story examples across seeds.

No functions or methods defined in this file.

### `Simulations/Theoretical_Examples/RVs_Story/regular_examples/core_model.py`

File description: Compatibility wrapper for regular RVs_Story examples.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `true_mi_pid (line 13)` | sources, target, covariance=None | call `truth_pid_equal_unique(...)` | Return the equal-unique true PID row. |
| `main_func (line 27)` | config, function_to_run | call `run_pid_story(...)` | Run a regular RV generator through the shared PID story runner. |

### `Simulations/Theoretical_Examples/RVs_Story/regular_examples/equal_unique.py`

File description: Regular examples with balanced unique information.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `equal_unique (line 15)` | rng, n, p, noise_std | tuple of 3 values | Generate a Gaussian example with equal unique information in both sources. |
| `equal_unique2 (line 39)` | rng, n, p, noise_std, snr=1 | tuple of 3 values | Generate a higher-dimensional equal-unique example from real features. |

### `Simulations/Theoretical_Examples/RVs_Story/regular_examples/no_mi.py`

File description: Python module for no mi-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `zero_MI (line 10)` | rng, n, p, noise_std | tuple of 3 values | Generate a Gaussian example with equal unique information in both sources. |

## Simulations/Theoretical_Examples/RVs_Story/suppresion_examples

Suppression/suppressor-variable theoretical examples.

### `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/all_supp_examples.py`

File description: Run all suppression RVs_Story examples across seeds.

No functions or methods defined in this file.

### `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/core_model.py`

File description: Compatibility wrapper for suppression RVs_Story examples.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `true_mi_pid (line 13)` | sources, target, covariance=None | call `truth_pid_suppression(...)` | Return the suppression-style true PID row. |
| `main_func (line 27)` | config, function_to_run | call `run_pid_story(...)` | Run a suppression RV generator through the shared PID story runner. |

### `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/full_suppresion.py`

File description: Full suppression Gaussian RV example.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `full_suppresion (line 15)` | rng, n, p, noise_std | tuple of 3 values | Generate the full suppression example. |

### `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/unq12_zero.py`

File description: Suppression example where both sources have zero unique information.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `unq12_zero (line 15)` | rng, n, p, noise_std | tuple of 3 values | Generate an example with zero source-1 and source-2 unique information. |

### `Simulations/Theoretical_Examples/RVs_Story/suppresion_examples/unq2_zero.py`

File description: Suppression example where source 2 has zero unique information.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `unq2_zero (line 15)` | rng, n, p, noise_std | tuple of 3 values | Generate an example with zero source-2 unique information. |

## Simulations/Theoretical_Examples/RVs_Story/Non-gaussian

Non-Gaussian random-variable story examples.

### `Simulations/Theoretical_Examples/RVs_Story/Non-gaussian/core_model.py`

File description: Compatibility wrapper for non-Gaussian RVs_Story examples.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `main_func (line 13)` | config, function_to_run | call `run_pid_story(...)` | Run a non-Gaussian RV generator without a Gaussian truth row. |

### `Simulations/Theoretical_Examples/RVs_Story/Non-gaussian/t-dist_rvs.py`

File description: Student-t non-Gaussian RV examples.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `standardized_t (line 13)` | rng, df, size | `rng.standard_t(df=df, size=size) / np.sqrt(df / (df - 2))` | Sample a variance-one Student-t random variable. |
| `unq2_zero_t (line 29)` | rng, n, p, noise_std, df=5 | tuple of 3 values | Generate a Student-t version of the zero-source-2-unique example. |
| `unq12_zero (line 53)` | rng, n, p, noise_std, df=5 | tuple of 3 values | Generate a Student-t version of the zero-both-unique example. |

## Simulations/evil_twin

Evil-twin covariance examples, PID sweeps, and related checks.

### `Simulations/evil_twin/__init__.py`

File description: Sonic/Shadow evil-twin covariance examples.

No functions or methods defined in this file.

### `Simulations/evil_twin/covariance_example.py`

File description: Torch Sonic/Shadow evil-twin covariance example.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `empirical_covariance_matrix_torch (line 6)` | X1, X2, T, correction=1 | call `torch.cov(...)` | Compute the empirical covariance matrix of (X1, X2, T). |
| `check_evil_twin_covariances_torch (line 24)` | data, atol=0.05, rtol=0.05, verbose=True | dict (Sigma_sonic, Sigma_shadow, difference, max_abs_difference, are_equal) | Compare the empirical covariance matrices of Sonic and Shadow. |
| `evil_twin_example_torch (line 70)` | generator, n, p, device='cpu', dtype=torch.float64 | dict (sonic, shadow); `scale * torch.randn(n, p, generator=generator, device=device, dtype=dtype)` | Generate the Sonic and Shadow evil-twin Gaussian examples. |
| `evil_twin_example_torch.randn_scaled (line 88)` | var_total | `scale * torch.randn(n, p, generator=generator, device=device, dtype=dtype)` | No docstring; infer behavior from name/signature before reuse. |
| `run_covariance_comparison (line 126)` | n=1000, p=300, seed=0, device='cpu', dtype=torch.float64, atol=0.05, rtol=0.05, verbose=True | `result` | Generate Sonic/Shadow samples and compare their empirical covariances. |

### `Simulations/evil_twin/evil_twin_pid_batch_utils.py`

File description: Seed-loop and CSV helpers for evil-twin PID_calc sweeps.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `summary_csv_path (line 39)` | output_dir: Path, prefix: str='evil_twin_pid' | Annotated: `Path` | Build the output CSV path for the mean summary table. |
| `method_csv_path (line 53)` | output_dir: Path, method: str, prefix: str='evil_twin_pid' | Annotated: `Path` | Build the output CSV path for one PID method. |
| `append_rows_to_csv (line 68)` | path: Path, rows: list[dict] | Annotated: `Path` | Append rows to a CSV file, creating the header when needed. |
| `write_rows_to_csv (line 87)` | path: Path, rows: list[dict], fieldnames: list[str] | Annotated: `Path` | Write a complete CSV table, replacing any existing file. |
| `make_pid_config (line 106)` | n: int, p: int, device: str, flow_epochs: int, flow_verbose: bool | Annotated: `dict` | Create the config dictionary expected by PID_calc wrappers. |
| `result_row (line 131)` | seed: int, twin: str, method: str, n: int, p: int, pid: dict, mi: dict | Annotated: `dict` | Create a successful result row for one twin and PID method. |
| `error_row (line 164)` | seed: int, twin: str, method: str, n: int, p: int, error: Exception | Annotated: `dict` | Create an error row for one failed twin and PID method. |
| `run_seed (line 193)` | seed: int, config: dict, methods: tuple[str, ...], output_dir: Path, csv_prefix: str | Annotated: `dict` | Run all requested PID methods for one evil-twin seed and save CSV rows. |
| `summary_fieldnames (line 238)` | twins: tuple[str, ...]=SUMMARY_TWINS | Annotated: `list[str]` | Build ordered column names for the mean summary table. |
| `mean_summary_rows (line 254)` | seed_results: dict, methods: tuple[str, ...], twins: tuple[str, ...]=SUMMARY_TWINS | Annotated: `list[dict]` | Calculate mean PID and MI values across seeds for each method. |
| `save_summary_csv (line 288)` | output_dir: Path, prefix: str, rows: list[dict] | Annotated: `Path` | Save the mean summary table to a CSV file. |
| `summary_image_path (line 302)` | output_dir: Path, prefix: str, twin: str | Annotated: `Path` | Build the output image path for one twin's mean summary table. |
| `summary_rows_to_pid_results (line 317)` | rows: list[dict], twin: str | Annotated: `dict` | Convert mean summary rows to the PID result shape used by RVs_Story tables. |
| `save_summary_table_images (line 346)` | rows: list[dict], output_dir: Path, prefix: str, config: dict, twins: tuple[str, ...]=SUMMARY_TWINS | Annotated: `list[Path]` | Save RVs_Story-style PID comparison images for the mean summary. |
| `format_summary_value (line 391)` | value, decimals: int | Annotated: `str` | Format one summary table cell for terminal output. |
| `format_summary_table (line 408)` | rows: list[dict], decimals: int=6 | Annotated: `str` | Format summary rows as an aligned plain-text table. |
| `run_evil_twin_pid_sweep (line 439)` | seeds: list[int], n: int=1000, p: int=1, methods: tuple[str, ...]=DEFAULT_METHODS, output_dir: Path \| str=Path('simulation_results/evil_twin_pid'), device: str='cpu', flow_epochs: int=250, flow_verbose: bool=False, csv_prefix: str='evil_twin_pid' | Annotated: `dict` | Run PID_calc methods on Sonic and Shadow across multiple seeds. |

### `Simulations/evil_twin/run_pid_calc_methods.py`

File description: Run PID_calc methods on the evil-twin covariance example across seeds.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `parse_args (line 16)` | No inputs | Annotated: `argparse.Namespace` | Parse command-line arguments for the evil-twin PID sweep. |
| `main (line 38)` | No inputs | Annotated: `dict` | Run the command-line evil-twin PID sweep. |

## source_conwell_code

Source Conwell analysis scripts kept inside the project tree.

### `source_conwell_code/__init__.py`

File description: Python module for   init  -related project logic.

No functions or methods defined in this file.

## source_conwell_code/pressures

Source Conwell pressure analysis scripts.

### `source_conwell_code/pressures/__init__.py`

File description: Python module for   init  -related project logic.

No functions or methods defined in this file.

### `source_conwell_code/pressures/main_analysis.py`

File description: Python module for main analysis-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `get_splithalf_xy (line 13)` | feature_map, response_data, scale=True | `data_splits` | No docstring; infer behavior from name/signature before reuse. |
| `permute_benchmark (line 26)` | benchmark | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `run_benchmarking (line 30)` | benchmark, model_option, precomputed_feature_maps=None, layers_to_retain=None, metrics=['crsa', 'srpr', 'wrsa'], alpha_values=np.logspace(-1, 5, 7).tolist(), regression_means=True, precompute_rdms=True | `results` | No docstring; infer behavior from name/signature before reuse. |
| `get_results_max (line 166)` | results, average_over=None, average_when='after' | `results` | No docstring; infer behavior from name/signature before reuse. |

### `source_conwell_code/pressures/ridge_gcv_mod.py`

File description: Python module for ridge gcv mod-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `pearson_r_score (line 19)` | y_true, y_pred, multioutput=None | `pearsonr_vec(y_true_, y_pred_)[0]` | No docstring; infer behavior from name/signature before reuse. |
| `_RidgeGCVMod class (line 24)` | class | class | Ridge regression with built-in Leave-one-out Cross-Validation. |
| `_RidgeGCVMod.__init__ (line 27)` | self, alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, copy_X=True, gcv_mode=None, store_cv_values=False, is_clf=False, alpha_per_target=False | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `_RidgeGCVMod.fit (line 50)` | self, X, y, sample_weight=None | `self` | No docstring; infer behavior from name/signature before reuse. |
| `_BaseRidgeCVMod class (line 174)` | class | class | Class with methods listed below. |
| `_BaseRidgeCVMod.fit (line 175)` | self, X, y, sample_weight=None | `self` | No docstring; infer behavior from name/signature before reuse. |
| `RidgeCVMod class (line 226)` | class | class | Ridge regression with built-in cross-validation. |

## source_conwell_code/pressures/brain_data

Source Conwell brain-data benchmark and parsing helpers.

### `source_conwell_code/pressures/brain_data/__init__.py`

File description: Python module for   init  -related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `average_rdms (line 7)` | rdm_array | `1 - fisherz_inv(fisherz(np.stack([1 - rdm for rdm in rdm_array])).mean(axis=0, keepdims...`; call `np.arctanh(...)`; call `np.tanh(...)` | No docstring; infer behavior from name/signature before reuse. |
| `average_rdms.fisherz (line 8)` | r, eps=1e-05 | call `np.arctanh(...)` | No docstring; infer behavior from name/signature before reuse. |
| `average_rdms.fisherz_inv (line 11)` | z | call `np.tanh(...)` | No docstring; infer behavior from name/signature before reuse. |
| `NSDBenchmark class (line 17)` | class | class | Class with methods listed below. |
| `NSDBenchmark.__init__ (line 18)` | self, image_set='shared1000', voxel_set='OTC-only', train_test_split=False, clean_rdms_only=True, anatomical_roi_subset=None, functional_roi_subset=None | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `NSDBenchmark.get_sample_stimulus (line 80)` | self, image_index=None | call `Image.open(...)` | No docstring; infer behavior from name/signature before reuse. |
| `NSDBenchmark.get_rdm_indices (line 90)` | self, roi_subset=None, row_number=False | `rdm_indices` | No docstring; infer behavior from name/signature before reuse. |
| `NSDBenchmark.get_rdms (line 111)` | self, roi_subset=None, include_group_average=False | `brain_rdms` | No docstring; infer behavior from name/signature before reuse. |
| `NSDBenchmark.get_splithalf_rdms (line 136)` | self | `split_rdms` | No docstring; infer behavior from name/signature before reuse. |

### `source_conwell_code/pressures/brain_data/benchmark.py`

File description: Python module for benchmark-related project logic.

No functions or methods defined in this file.

### `source_conwell_code/pressures/brain_data/nsd_parser.py`

File description: Python module for nsd parser-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `get_nsd_path (line 21)` | config=None, key='NSD_PATH' | call `config.get(...)` | No docstring; infer behavior from name/signature before reuse. |
| `_nsd_path_prompt (line 42)` | No inputs | None; `user_input` | No docstring; infer behavior from name/signature before reuse. |
| `_check_space (line 69)` | space | No explicit return; likely `None` / side effects. | No docstring; infer behavior from name/signature before reuse. |
| `get_subj_dims (line 73)` | subj, space='func1pt8mm' | `nib.load(fn).get_fdata().shape` | No docstring; infer behavior from name/signature before reuse. |
| `load_voxel_info (line 81)` | subj, space | `roi_dfs` | No docstring; infer behavior from name/signature before reuse. |
| `load_NSD_voxel_metadata (line 210)` | subjs, roi_group, space, voxels_to_include=None, savedir=None, overwrite=False | `voxel_metadata` | No docstring; infer behavior from name/signature before reuse. |
| `load_NSD_brain_data (line 356)` | subjs, space, roi_group, voxel_metadata, annotations, savedir, output=False | `brain_data` | No docstring; infer behavior from name/signature before reuse. |
| `load_NSD_benchmark_ROI_metadata (line 455)` | subjs, space, ncsnr_threshold=0.2, t_threshold=1, savedir=None, overwrite=False, **kwargs | `voxel_metadata`; `excl_idx`; call `np.isin(...)`; call `np.logical_not(...)` | No docstring; infer behavior from name/signature before reuse. |
| `load_NSD_benchmark_ROI_metadata.get_idx_by_roi_group (line 464)` | metadata, group_label, roi_names, logic='include', t_threshold=None | `excl_idx`; call `np.isin(...)`; call `np.logical_not(...)` | No docstring; infer behavior from name/signature before reuse. |

## supression_effect

Suppression-effect model experiments and encoder definitions.

### `supression_effect/Suppresed_Encoder.py`

File description: Python module for Suppresed Encoder-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `_append_project_root_to_path (line 8)` | No inputs | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `get_run_config (line 37)` | No inputs | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `load_model_and_fmri (line 63)` | config: dict | tuple of 2 values | No docstring; infer behavior from name/signature before reuse. |
| `prepare_inputs (line 70)` | config: dict, real_features: np.ndarray, fmri_dict: dict | tuple of 2 values | No docstring; infer behavior from name/signature before reuse. |
| `run_suppression_pipeline (line 94)` | config: dict, selected_features: np.ndarray, encoder, verbose: bool=True | tuple of 3 values | No docstring; infer behavior from name/signature before reuse. |
| `save_all_seed_runs_results (line 129)` | seed_rows: list[dict], config: dict | Annotated: `Path` | No docstring; infer behavior from name/signature before reuse. |
| `run_single_seed (line 149)` | seed: int, config: dict, real_features: np.ndarray, fmri_dict: dict | Annotated: `dict` | No docstring; infer behavior from name/signature before reuse. |
| `run_encoding_multi_seed_experiment (line 168)` | config: dict | Annotated: `tuple[dict, list[dict]]` | No docstring; infer behavior from name/signature before reuse. |
| `save_seed_summary (line 181)` | summary: dict, config: dict | Annotated: `Path` | No docstring; infer behavior from name/signature before reuse. |
| `print_results (line 185)` | outputs: dict, mi: dict, pid: dict | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |
| `main (line 199)` | No inputs | Annotated: `None` | No docstring; infer behavior from name/signature before reuse. |

### `supression_effect/__init__.py`

File description: Python module for   init  -related project logic.

No functions or methods defined in this file.

### `supression_effect/gauss_univariate.py`

File description: Python module for gauss univariate-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `gauss_simple_example (line 23)` | N=1000, P=1, rng_seed=1, noise_seed=1, simple_example=True, snr=1.0, method='ridge_cv', mixing_dimension=None | tuple of 2 values | Run a simple Gaussian example with no mixing deminsion each experiement has different random seeds. |
| `check_supression_effect (line 76)` | vp_results, pid_results | None | Check for suppression effect in the results. |
| `main (line 122)` | No inputs | No explicit return; likely `None` / side effects. | Main function to run the Gaussian simple example and compare results. |

### `supression_effect/supp_gauss_multivariate.py`

File description: Python module for supp gauss multivariate-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `crossfit_residualize (line 46)` | Y_raw, X_raw, n_splits=5, seed=0 | `residuals` | Residualize Y_raw against X_raw using cross-fitted linear regression. Returns residuals Y - E[Y\|X] predicted out-of-fold. |
| `test_suppression (line 64)` | N=1000, P=1, suppression_strength=0.5, rng_seed=1, mode='simple', snr=1.0, method='ridge_cv', mixing_dimension=None | tuple of 3 values | Run a simple Gaussian example with no mixing deminsion each experiement has different random seeds. |
| `plot_pid_results (line 125)` | mi_results=None, pid_results=None, sub_title=None | No explicit return; likely `None` / side effects. | Plot bar chart for PID results. |
| `get_seed_sweep_config (line 173)` | No inputs | Annotated: `dict` | Configuration for fixed-parameter suppression simulations across seeds. |
| `run_single_seed_fixed (line 195)` | seed: int, config: dict | Annotated: `dict` | Run one suppression experiment seed with all other parameters fixed. |
| `run_fixed_params_across_seeds (line 210)` | config: dict \| None=None | Annotated: `tuple[dict, list[dict]]` | Sweep over seeds while keeping all other simulation parameters fixed, then save the per-seed results and mean/std summary. |
| `main (line 238)` | No inputs | No explicit return; likely `None` / side effects. | Main function to run the Gaussian simple example and compare results. |

## toy_examples

Small toy scripts for covariance, PID, and suppression demonstrations.

### `toy_examples/Theortical_cov_toy_example.py`

File description: Python module for Theortical cov toy example-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `make_spd_matrix (line 15)` | p, rng, eig_min=0.5, eig_max=2.0 | `Sigma` | Create a random symmetric positive definite covariance matrix. |
| `check_spd (line 30)` | Sigma, name='Sigma', tol=1e-10 | No explicit return; likely `None` / side effects. | Check whether a matrix is symmetric positive definite. |
| `theoretical_covariance_multivariate (line 52)` | Sigma_R, Sigma_U, Sigma_N, Sigma_eps, order=('X1', 'X2', 'Y') | tuple of 2 values | Theoretical covariance for the multivariate generative process: |
| `simulate_multivariate_process (line 123)` | n, Sigma_R, Sigma_U, Sigma_N, Sigma_eps, rng=None | tuple of 3 values | Simulate: |
| `extract_covariance_blocks (line 161)` | Sigma_full, p | `blocks` | Extract covariance blocks assuming order [X1, X2, Y]. |
| `whiten_theoretical_covariance_blocks (line 187)` | Sigma_full, p, device='cpu', dtype=torch.float64 | dict (Sigma_X1X1, Sigma_X2X2, Sigma_YY, Sigma_X1X2, Sigma_X1Y, Sigma_X2Y, P_X1X2, Q_X1Y...) | Given the full covariance matrix of [X1, X2, Y], compute the whitened cross-covariance blocks: |
| `validate_multivariate_covariance (line 281)` | n=1000000, p=5, seed=123, device='cpu' | dict (Sigma_R, Sigma_U, Sigma_N, Sigma_eps, Sigma_theoretical, Sigma_empirical, cov_blocks, abs_err...) | Validate the theoretical covariance by simulation, then compute whitened covariance blocks. |

### `toy_examples/simple_problems.py`

File description: Python module for simple problems-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `bern_pair_as_channels (line 8)` | p1=0.5, p2=0.5 | tuple of 3 values | Convert two independent Bernoulli(p1), Bernoulli(p2) into PXgS, PYgS, PS format compatible with computeQUI_numpy. |
| `dist_from_tensor (line 59)` | Q, names=('S', 'X', 'Y') | `d` | No docstring; infer behavior from name/signature before reuse. |

### `toy_examples/suppression_pipeline_example.py`

File description: Example script demonstrating the use of the suppression analysis pipeline.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `example_with_synthetic_data (line 24)` | No inputs | `results` | Example using synthetic data to demonstrate the pipeline. |
| `example_with_pretrained_models (line 109)` | No inputs | `results` | Example showing how to use the pipeline with pre-trained models. |

### `toy_examples/suppression_toy_runner.py`

File description: Shared runners for suppression toy examples.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `generate_correlated_features (line 12)` | n, p, rho, rng | call `rng.multivariate_normal(...)` | Generate samples with an AR(1)-style covariance structure. |
| `_apply_mixing (line 19)` | rng, X_M1, X_M2, mixing_dimension | tuple of 2 values | No docstring; infer behavior from name/signature before reuse. |
| `_split_signal_sources (line 28)` | rng, n, p, snr | tuple of 3 values | No docstring; infer behavior from name/signature before reuse. |
| `_feature_correlation_sources (line 50)` | rng, n, p, snr, rho | tuple of 3 values | No docstring; infer behavior from name/signature before reuse. |
| `build_toy_sources (line 69)` | rng, n, p, snr, experiment_kind, rho=1 | call `_split_signal_sources(...)`; call `_feature_correlation_sources(...)` | No docstring; infer behavior from name/signature before reuse. |
| `run_toy_experiment (line 80)` | rng, n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', experiment_kind=SPLIT_SIGNAL, rho=1 | `decomp` | Run one toy suppression/commonality experiment. |
| `run_all_toy_methods (line 128)` | rng_seed, n, p, mixing_dimension, snr, experiment_kind, methods=DEFAULT_METHODS, report_negative_common=False, rho=1 | `results` | Run all requested commonality methods with a fixed seed. |
| `run_default_factorial_scenarios (line 172)` | experiment_kind, n=1000, p=100, seed=42, report_negative_common=False | No explicit return; likely `None` / side effects. | Run the standard low/high SNR by mixing-dimension toy scenarios. |

### `toy_examples/toy_example.py`

File description: Python module for toy example-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `unq2_zero_with_red_unq1_syn (line 15)` | rng, n, p, noise_std=0.9 | tuple of 3 values | Continuous Gaussian-like example where theoretically: |
| `run_experiment (line 91)` | rng, suppresion_strength, mode='permuted', n=1024, p=100, mixing_dimension=None, snr=10.0, method='standard', show_diagnostic_plots=False | tuple of 2 values | Run commonality analysis experiment. |
| `run_ridge_toy_method (line 175)` | rng_seed, n, p, mixing_dimension, snr | `decomp_dict` | Run the ridge-CV analysis for this specialized toy example. |
| `main (line 196)` | No inputs | No explicit return; likely `None` / side effects. | Run the 2x3 factorial experiment design. |
| `plot_components (line 240)` | comp_dict, title='Variance Decomposition' | No explicit return; likely `None` / side effects. | Plot bar chart for component dictionary. |
| `gauss_simple_example (line 259)` | No inputs | No explicit return; likely `None` / side effects. | Run a simple Gaussian example with no mixing deminsion each experiement has different random seeds. |

### `toy_examples/toy_example_feature_correlation.py`

File description: Python module for toy example feature correlation-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `main (line 16)` | No inputs | No explicit return; likely `None` / side effects. | Run the 2x3 factorial experiment design. |

### `toy_examples/toy_example_new.py`

File description: Python module for toy example new-related project logic.

| Function / Method | Inputs | Outputs | What it does |
|---|---|---|---|
| `main (line 16)` | No inputs | No explicit return; likely `None` / side effects. | Run the 2x3 factorial experiment design. |
