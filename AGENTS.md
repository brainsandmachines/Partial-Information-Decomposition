### Instrucitons 

1. You are not allowed to make changes to other files that are not specefied without asking for permissions. If you want to make a change explain what is the change why would change it and then ask for permission. That includes adding or removing stuff.  

2. On every function you create you should make a description of the funciton what inputs it expects and their types and what it will output.  


3. When you are importing something put it in the top of the file and not inside a function.

4. Each time you write a code that changes the shape or returns a tensor/array write a comment that say from what shape to what shape it changed. In that format: old_shape -> new_shape


### Research Explenation 
# Master's research context for Codex or Claude

Last updated: 12 October 2026.

Use this guide to understand my research when helping with analysis, code, literature, Notion notes, and Overleaf writing. The main project is an empirical neuroscience study about what different deep neural networks explain about cortical responses. Eigen-PID is a theoretical contribution that enables this study.

## 1. Researcher and central question

My name is Ohad Sheelo. My background is in neuroscience and mathematics, and my master's research is in computational neuroscience at Ben-Gurion University of the Negev, supervised by Tal Golan and Ari Pakman.

The project is titled:

**Do Different Deep Learning Models Explain Distinct Components of the Cortical Neural Code?**

The central question is:

**When different DNNs achieve similar overall neural predictivity, do they explain the same information about neural responses, or does each capture information that the other does not?**

My objective is to quantify the overlap and complementarity of the information provided by different models about visual cortical responses. Comparing individual prediction scores is insufficient to answer this question.

## 2. Scientific motivation

Intermediate representations of DNNs trained on visual tasks can predict responses in the primate visual cortex through linear encoding models. Deeper layers often correspond more closely to later stages of the ventral visual pathway. This makes DNNs useful computational models for studying visual representations, particularly in higher-order visual cortex, whose computational principles remain incompletely understood.

However, models with different architectures, learning objectives, and training data can achieve similar neural predictivity. For example, large-scale comparisons have found comparable predictivity in human occipitotemporal cortex across some CNN, Vision Transformer, vision-only, and vision-language models.

A similar prediction score does not tell us whether two models succeed on the same aspects of the neural response. Two models could explain largely overlapping information, or each could capture something that the other misses.

This connects the project to the debate over whether representations shared with the brain are largely universal across networks or retain contributions associated with model-specific inductive biases.

The intended scientific contribution is to move from asking **how well each model predicts the brain** to asking **how the information about the brain is distributed across models**.

## 3. Planned comparisons and what they can tell us

I plan to compare controlled pairs of models differing along particular sources of inductive bias:

| Comparison | Scientific question |
| --- | --- |
| CNN versus Vision Transformer | Do different architectures capture distinct information about cortical responses? |
| Supervised versus self-supervised learning | Does the learning objective affect which information about neural responses a model captures? |
| Vision-only versus vision-language training | Does language alignment provide information about cortical responses beyond that captured through visual learning alone? |

A motivating example is comparing CLIP with a matched vision-only model to ask whether language alignment contributes unique information. Exact model pairs and matching criteria still need to be specified; do not assume that a comparison isolates one factor if architecture, training data, or other properties also differ.

The analysis should report all four PID components:

- **Redundancy:** information about the neural target available from either model.
- **Unique information from model 1:** information about the target supplied by model 1 beyond what model 2 provides.
- **Unique information from model 2:** the corresponding contribution of model 2.
- **Synergy:** information about the target available only from the models jointly.

The unique components are the primary quantities of interest. They address whether models contribute distinct information even when their overall prediction scores are similar.

Possible outcomes have different implications:

- Substantial unique information would support complementarity between the selected model predictions and motivate investigating which properties account for it.
- Predominantly redundant information would suggest that these models largely capture the same information about the chosen neural target.
- Synergy would identify a benefit from considering both predictions together, which needs to be distinguished from information unique to either model.

These are interpretations to evaluate, not predetermined findings. Low unique information for the tested pairs would not establish universality across all networks or all cortical representations. Likewise, positive unique information does not by itself identify a biological mechanism or prove that one architectural or training difference caused it.

## 4. Data, variables, and empirical pipeline

The current empirical focus is human fMRI responses from the **Natural Scenes Dataset (NSD)**, initially emphasizing **occipitotemporal cortex (OTC)**.

NSD contains high-resolution 7T fMRI measurements from eight participants viewing natural images. Each participant viewed approximately 9,000-10,000 distinct images, including a designated set of 1,000 images shared across participants. The intended design uses participant-specific images for encoding-model fitting and selection, and reserves the shared images for held-out evaluation.

### What the PID variables represent

| Symbol | Meaning in the empirical study |
| --- | --- |
| \(T\) | Measured cortical responses after PCA reduction |
| \(X_1\) | Predictions of that neural target from DNN model 1 |
| \(X_2\) | Predictions of that neural target from DNN model 2 |
| \(n\) | Number of stimulus observations used for estimation |
| \(d_T,d_1,d_2\) | Dimensions of the target and the two sources |

Each observation corresponds to the same stimulus across the neural target and both predictions.

The PID sources are **encoding-model predictions of the neural target**, rather than the original DNN activations. DNN features are first mapped to the neural response space through ridge regression. Interpret the resulting PID as information in these predictions about the selected neural target; it does not automatically describe all information in the original model representations or all cortical activity.

My notation varies across tasks. The neural target may also be called \(Y\). In Eigen-PID derivations, the target is \(M\) and the sources are \(X,Y\). Always identify variable roles before translating notation.

### Intended analysis sequence

1. **Prepare neural responses.** Assemble stimulus-aligned cortical responses for each participant and the selected region.
2. **Reduce the neural target.** Apply PCA to obtain a lower-dimensional representation while retaining most of the neural variance. The choice of components remains an important methodological question.
3. **Extract DNN features.** Pass the NSD images through each model and collect activations from intermediate layers.
4. **Fit encoding models.** Use ridge regression to predict the PCA-reduced neural responses from each layer's features.
5. **Select layers.** Select the most predictive layer using cross-validated correlation within the training data.
6. **Generate held-out predictions.** Predict responses to the designated shared images using the selected encoding models.
7. **Construct each model-pair analysis.** Use the two held-out predictions as sources and the corresponding measured PCA-reduced responses as the target.
8. **Estimate information.** Fit a joint Gaussian model through its covariance and estimate pairwise MI, joint MI, and BROJA PID using Eigen-PID.
9. **Correct finite-sample bias.** Apply suitable Gaussian MI and PID bias corrections and assess their behavior.
10. **Compare information profiles.** Examine redundancy, both unique components, synergy, and MI across model pairs and participants, alongside neural-predictivity scores.

These steps describe the intended design. Do not claim that every step is implemented, validated, or finalized.

Using the same evaluation stimuli across participants supports comparison of their information estimates, but the statistical procedure for those comparisons still needs to be specified.

## 5. Why BROJA PID is the chosen framework

Mutual information quantifies the relationship between predictions and neural responses, but individual or joint MI alone does not separate redundancy, uniqueness, and synergy.

For two sources and a target:

$$
I(T;X_1,X_2)=R+U_1+U_2+S,
\qquad
I(T;X_1)=R+U_1,
\qquad
I(T;X_2)=R+U_2.
$$

**BROJA is the selected PID definition. Do not use MMI as the default or silently substitute another definition.**

For the observed distribution \(P\), define the compatible couplings:

$$
\Delta(P)=\{Q:Q(T,X_1)=P(T,X_1),\ Q(T,X_2)=P(T,X_2)\}.
$$

BROJA unique information is:

$$
U_1^{\mathrm{BROJA}}
=\inf_{Q\in\Delta(P)}I_Q(T;X_1\mid X_2),
\qquad
U_2^{\mathrm{BROJA}}
=\inf_{Q\in\Delta(P)}I_Q(T;X_2\mid X_1).
$$

The optimization preserves each source's relationship with the target while allowing the dependence between sources to vary. Use a minimum when attainment is established under the applicable assumptions.

### Unique information versus suppression

Commonality analysis, or variance partitioning, can assign a positive unique explained-variance contribution to a suppressor. Consider mutually independent Gaussian variables \(T,N,\varepsilon\), with:

$$
X_1=T+N+\varepsilon,
\qquad
X_2=N.
$$

\(X_2\) alone carries no information about \(T\), but improves prediction when combined with \(X_1\) by allowing nuisance variation to be removed. BROJA assigns \(U_2=0\) and attributes this joint benefit to synergy.

This distinction matters for my research: an improvement in joint prediction should not automatically be described as unique information supplied by a model.

Blackwell comparison is another motivation for BROJA. If one source's channel from the target can be obtained by garbling the other's, BROJA assigns zero unique information to the inferior source.

I previously investigated James's dependency-constraint PID, \(I_{\mathrm{dep}}\), using its Gaussian analytical formulation. Suppression examples and the Evil Twin construction motivated a move to BROJA because of the desired interpretation of uniqueness and the Blackwell property. \(I_{\mathrm{dep}}\) is historical work, not the current default.

## 6. Eigen-PID's role in the project

We developed **Eigen-PID**, a closed-form computation of BROJA PID for multivariate Gaussian systems. It enables the empirical analysis by making repeated PID evaluations much faster, including evaluations needed for resampling-based bias correction.

Eigen-PID computes the selected BROJA definition; it is not a separate PID definition. Its theoretical development is a distinct contribution from the empirical question about DNNs and cortical responses.

Using target \(M\) and sources \(X,Y\), define:

$$
\Gamma_X=\operatorname{Cov}(M\mid X),
\qquad
\Gamma_Y=\operatorname{Cov}(M\mid Y).
$$

Under nondegenerate jointly Gaussian target-source marginals, these posterior covariances are positive definite. Solve the generalized eigenproblem:

$$
\Gamma_Yv_i=\lambda_i\Gamma_Xv_i.
$$

The unique-information terms, in bits, are:

$$
U_X^{\mathrm{BROJA}}=\frac12\sum_i[\log_2\lambda_i]_+,
\qquad
U_Y^{\mathrm{BROJA}}=\frac12\sum_i[-\log_2\lambda_i]_+.
$$

Eigenvalues above one contribute to unique information from \(X\), while those below one contribute to unique information from \(Y\). In multivariate systems, both sources can therefore carry positive unique information, even when their pairwise MI values are equal.

The method also computes the BROJA union information \(I_\cup\), the minimum joint MI over compatible couplings. The other atoms follow from:

$$
R=I(M;X)+I(M;Y)-I_\cup,
\qquad
S=I_P(M;X,Y)-I_\cup.
$$

Union and unique information depend on the target-source marginals; synergy additionally depends on the observed joint distribution. Keep the optimal compatible coupling separate from the observed covariance used to calculate joint MI and synergy.

The closed-form solution removes iterative optimization and has shown speedups exceeding two orders of magnitude in high-dimensional comparisons with numerical methods. Its exact population solution does not remove finite-sample estimation bias.

We also completed a separate feasibility demonstration using left and right V1 as sources and hV4 as a target in NSD. Those results concern relationships between brain regions, not the planned comparisons between DNN predictions. The current focus has returned to the DNN study.

## 7. Current methodological questions

The main practical challenge is estimating information reliably from a relatively small evaluation set compared with the dimensions of neural responses and model features.

- **Neural dimensionality:** choose PCA components that provide a useful neural target without retaining excessive noise. A variance-retention threshold is not automatically a validated choice of predictable dimensions.
- **Encoding-model selection:** specify layer selection, ridge hyperparameters, and validation procedures, keeping all selection within training data.
- **Covariance estimation:** check rank, positive definiteness where required, and conditioning. Reducing each variable block below the sample count does not guarantee that the full joint covariance is nonsingular.
- **Finite-sample bias:** distinguish raw estimates, corrected estimates, and uncertainty. Relevant approaches include analytical Gaussian entropy/MI correction and Lorenz et al.'s merged permutation and parametric-bootstrap PID correction. Their suitability for the DNN pipeline must be checked.
- **Gaussian modeling:** interpret the estimates under the chosen Gaussian model and assess departures from its assumptions where relevant.
- **Model comparability:** determine which pairs provide informative comparisons of architecture, learning objective, or language alignment.
- **Statistical inference:** specify how model-pair effects and participant variation will be assessed.

Fit learned preprocessing, including PCA and scaling, on training data and apply it to held-out data. Keep the shared evaluation images out of fitting, layer selection, and hyperparameter selection.

Exact model pairs, PCA component counts, covariance regularization, and the final statistical procedure are not fixed by this guide. Inspect current work before treating any choice as settled.

For toy examples and validation, establish expected outcomes through independent mathematical reasoning where possible. Do not use the algorithm being tested as its own ground truth.

## 8. How Codex should help me

Keep the central research question in view when helping with individual tasks. Connect methodological choices and interpretations to whether model predictions provide overlapping or distinct information about cortical responses.

### Scientific discussion and writing

- Help me understand literature, check derivations, inspect code, organize analyses, and interpret results.
- Assume familiarity with undergraduate neuroscience and mathematics. Explain unfamiliar methods precisely without unnecessary introductory material.
- Use clear, formal academic English and concrete wording. Keep answers concise unless I request a detailed explanation.
- For writing feedback, suggest replacements and identify grammar or spelling mistakes. Do not alter existing text unless I explicitly request an edit.
- Distinguish established results, observations, hypotheses, historical attempts, and proposed next steps.
- Use reliable primary sources for technical claims when the task requires literature support. Do not invent references, experimental results, parameter choices, or implementation details.
- Preserve BROJA as the current framework unless I explicitly ask to reconsider it.

### Notion research notes

Help me read, organize, and edit my Notion pages, including progress entries, literature notes, derivations, experimental plans, and records of results.

Read the current page and relevant section before editing. For feedback requests, provide suggestions. When I explicitly ask to edit or update a page, apply the scoped changes through an available authorized connection while preserving unrelated material, links, equations, and citations.

Keep completed observations separate from interpretations and next steps. Use dates for new progress entries when appropriate.

If live access is unavailable, provide a concrete replacement for the requested section and clearly state that the live page was not updated. After an applied edit, report which page and section changed.

#### Notion model catalogue

The Notion [Plan page](https://app.notion.com/p/3f738f2015f4801789eddaffb0f1baf6) contains the Models database used to organize candidate DNNs. Treat that database as the single source of truth. Its linked views group the same records by architecture and by learning objective; they are not separate tables or copied data. Changes made through any linked view update the underlying record and therefore appear in every view.

When asked to add models, first check whether each model already has a record, then add or update it once in the existing Models database (`collection://3f738f20-15f4-8096-837f-000b26cf1002`). Do not create manually duplicated mini-tables. Populate only information that is known or verified, using these fields:

- **Model:** commonly used model name.
- **Architecture:** main architecture family and variant.
- **Learning Objective:** training signal or pretraining objective.
- **Training Modalities:** modalities used during training.
- **Cross-modal Alignment:** modalities explicitly aligned during training; use `None` for single-modality models.
- **Primary Task:** main task for which the model was trained or designed.
- **Training Dataset:** dataset or dataset combination used for training.
- **Exact Checkpoint:** precise pretrained-weights identifier, which is distinct from the architecture and training dataset.
- **Source / Paper:** official paper, repository, or documentation supporting the entry.

The current ResNet-50, ViT-B/16, and DINOv2 ViT-B/14 records demonstrate how the linked views work. Do not treat them as finalized experimental selections merely because they are in the catalogue.

### Overleaf and LaTeX

Help me edit proposals, manuscripts, supplementary material, equations, references, figures, captions, and formatting, and resolve compilation errors.

Read the current source and surrounding context. Follow the project's notation, macros, bibliography style, and structure. Preserve labels and cross-references, and verify that the referenced results support the text.

For feedback requests, suggest changes. When I explicitly request a fix or rewrite, apply the scoped edit without changing unrelated material or scientific claims.

If working through a local Git checkout, inspect repository status and preserve uncommitted work. Follow my requested scope for committing and pushing. Report whether an edit is local or synchronized with Overleaf.

Compile and inspect affected output when tools are available and the change affects syntax or layout. If compilation or live access is unavailable, provide the usable source change and explain what was checked.

### Code and analysis

Inspect the actual implementation before assuming that a proposed method is already in use. Follow repository instructions and the requested scope. Verify that stimulus alignment, preprocessing, training/evaluation separation, variable roles, and estimator assumptions match the intended analysis.

The immediate goal is to build and validate the NSD model-comparison analysis, then use its information profiles to answer whether different DNN encoding models explain distinct components of cortical responses.

## 9. Repository structure

Search `FUNCTION_REGISTRY.md` before writing or changing a function; it lists every callable under `pid/`, `pipeline/`, and `simulations/` with signatures and conventions. Inspect the source before reuse.

### Top-level layout

| Path | Role |
| --- | --- |
| `pid/` | Estimator layer. `calculation.py` (`pid_calc`) wraps the installed `eigen_pid` package (`eigen_pid`, `debias_eigen_pid`); it is not implemented in this repository. `mutual_information.py` computes Gaussian MI with optional Wishart bias correction, `bias.py` holds the log-determinant bias terms, and `utils.py` contains covariance construction, shrinkage, comparison tables, and plotting helpers. |
| `pipeline/` | Real-data NSD pipeline (see below). |
| `simulations/` | Validation on synthetic Gaussian data with known ground truth (see below). |
| `tests/` | Pytest suite for the PID wrapper, pipeline, commonality analysis, PCA checkpointing, missMDA wrapper, and experiment runners. |
| `cluster/` | SLURM `sbatch` scripts, one per entry point; `instructions.txt` records the `pid_env` Conda environment and interactive-session commands. |
| `docs/` | `data_paths.txt` (NSD OTC beta locations on the cluster), `model_errors.txt`, and `real_data_pid_pipeline_plan.pdf`, the plan used by the QA-reviewer skill. |
| `external/` | Git submodules: `DeepDive` (DNN loading and feature extraction), `mayas_project` (NSD processing, OTC ROI, and per-model best-layer CSVs under `results_shared/encoding/best_layers/`), and `PID` (R code for Gaussian PIDs from the historical Idep work). |
| `source_conwell_code/` | Local, git-ignored copy of Conwell et al.'s `pressures` code. |
| `.codex/skills/` | `function-registry-reuse` and `real-data-pid-pipeline-qa-reviewer`. |

### `pipeline/`

- `pid_pipeline.py`: `PIDPipeline` runs these stages in order: `target_extraction`, `sources_extraction`, `choose_layer`, `feature_extraction`, optional `preprocess`, optional `feature_manipulation`, `pid_calculation`, and `pid_report`.
- `utils.py`: resolves stage names from a YAML `functions:` block into callables. It also holds the NSD adapters, such as `nsd_sources`, `nsd_feature_extraction`, `pca_each_source`, and `pid_calc_adapter`.
- `phases/`: stage implementations, including stimulus/target loading and feature extraction (`sources_target_features.py`), layer selection (`choosing_layer.py`), scaling and ridge preprocessing (`preprocessing_layer.py`), and PCA, ridge prediction, and other feature transforms (`feature_manipulations.py`).
- `pca/`: neural-target preparation. `subject_pca.py` fits a subject PCA on unique-image responses and projects the shared images. `missmda_ncp.py` selects the component count, and `find_alpha.py` finds one ridge alpha per target PC for each model's best layer.
- `analysis/pairwise/`: the main model-pair analyses. `prediction.py` runs prediction-level PID, which is the intended design. `representation.py` runs PID on PCA-reduced features instead.
- `analysis/pca/`: PID and MI as functions of the number of target PCs, and permutation analysis.
- `analysis/predictivity/`: ridge prediction correlation for each target PC.
- `experiments/`: single-pair runners for full OTC and single voxels.
- `examples/`: toy and smoke runs that need no data.
- `plotting.py`: pairwise PID heatmaps.

Most runners read a YAML config of the same name in the same folder, for example `prediction.py` reads `prediction.yaml`.

Main prediction-level flow:

1. `pipeline/pca/subject_pca.py` saves the fitted target PCA model.
2. The best layer for each model is read from the Maya's-project CSV through `overall_best_layer`.
3. `pipeline/pca/find_alpha.py` saves per-PC ridge alphas for each model.
4. `pipeline/analysis/pairwise/prediction.py` runs every unfinished unordered model pair. For each pair it extracts features, fits ridge on the unique images, predicts the shared images, and runs Eigen-PID. Results go to a resumable CSV, and the script then plots the heatmaps.

### `simulations/`

- `suppression_examples.py`, `commonality.py`: suppression generators and variance partitioning, for the VP-versus-PID comparison.
- `random_variables.py`: generators in which all atoms are positive, or both unique terms are equal.
- `evil_twin/`: Sonic/Shadow construction, with identical covariance and different generative stories.
- `pca/rank/`: PCA component-count selection on data with a known rank.
- `pca/ridge/`: checks whether the PCA and ridge routes preserve PID, using exact per-trial ground truth (`GROUND_TRUTH_PROOFS.md`).

### Conventions and cautions

- **Variable order differs by layer.** `pid_calc` uses `[T, X1, X2]`, the simulation generators and covariances use `[X1, X2, T]`, and the Eigen-PID derivation uses `[M, X, Y]`. Check the order at every boundary.
- **Units.** PID and MI outputs are in bits.
- **Ignored outputs.** Generated artifacts are git-ignored: `pipeline/pca/saved_pcs/`, `pipeline/pca/results/`, `simulation_results/`, and logs.
- **Absolute paths.** Several YAML configs, `__main__` blocks, and `sbatch` scripts use cluster paths under `/groups/golan_neurogroup/...` and checkout paths under `/home/ohadshee/Desktop/Partial-Information-Decomposition`. Verify these paths before running elsewhere.
