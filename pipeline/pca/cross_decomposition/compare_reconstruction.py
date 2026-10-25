"""Run a paired OTC comparison using the existing GSN configuration and loader."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd
import scipy
import yaml
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.pca.choosing_minimal_best_pca.choosing_minimal_best_pca import (
    load_repeat_responses,
    selection_results_to_table,
)
from pipeline.pca.cross_decomposition.cross_decomposition import (
    select_cross_decomposition_components,
    validation_design,
)
from pipeline.testing.gsn.gsn_pca import select_gsn_pca_components

DEFAULT_CONFIG_PATH = Path(__file__).with_suffix(".yaml")


def save_comparison(
    results: dict[str, dict[str, Any]], output_dir: Path, subj_id: str,
    image_set: str,
) -> pd.DataFrame:
    """Save paired curves, descriptive summaries, differences, and a PDF figure.

    Inputs: results: dict keyed 'gsn_pca'/'cross_decomposition', each containing
        GSN-compatible candidate (K,) and fold-error (folds,K) arrays; output_dir:
        Path to a fresh run directory; subj_id/image_set: str labels.
    Output: summary DataFrame (two rows). Writes method CSVs, summary.csv,
        paired_differences.csv and reconstruction_comparison.pdf. Differences
        are cross minus GSN within each fold, not an unpaired SE calculation.
    """
    gsn, cross = results["gsn_pca"], results["cross_decomposition"]
    candidates = gsn["candidate_n_components"]
    if not np.array_equal(candidates, cross["candidate_n_components"]):
        raise ValueError("Both methods must use exactly the same candidate counts.")
    if (gsn["fold_normalized_reconstruction_errors"].shape
            != cross["fold_normalized_reconstruction_errors"].shape):
        raise ValueError("Both methods must use the same fold count.")
    figure = Figure(figsize=(10, 7), layout="constrained")
    # Two axes -> array (2,).
    curve_axis, difference_axis = figure.subplots(2, 1, sharex=True)
    rows = []
    for method, selection in results.items():
        table = selection_results_to_table(selection, subj_id)
        table.insert(1, "image_set", image_set)
        # (n_folds,K) -> (K,).
        table["reconstruction_error_standard_deviation"] = selection[
            "fold_normalized_reconstruction_errors"
        ].std(axis=0, ddof=1)
        table.to_csv(output_dir / f"{method}.csv", index=False)
        means = selection["mean_normalized_reconstruction_errors"]
        errors = selection["reconstruction_error_standard_errors"]
        best = int(np.argmin(means))
        selected = int(selection["selected_n_components"])
        rows.append({
            "method": method, "subj_id": subj_id, "image_set": image_set,
            "best_n_components": int(candidates[best]),
            "selected_n_components_one_se": selected,
            "minimum_cv_normalized_mse": float(means[best]),
            "standard_error_at_best": float(errors[best]),
            "standard_deviation_at_best": float(table["reconstruction_error_standard_deviation"].iloc[best]),
            "one_se_choice_cv_normalized_mse": float(means[selected - 1]),
            "score_role": "component_selection_validation_not_final_test",
        })
        label = f"{method}: best K={candidates[best]}, one-SE K={selected}"
        line, = curve_axis.plot(candidates, means, label=label)
        curve_axis.fill_between(candidates, means - errors, means + errors,
                                color=line.get_color(), alpha=0.2)
        curve_axis.scatter(candidates[best], means[best], color=line.get_color())
    # (folds,K) - (folds,K) -> (folds,K).
    differences = (cross["fold_normalized_reconstruction_errors"]
                   - gsn["fold_normalized_reconstruction_errors"])
    # (folds,K) -> (K,).
    difference_mean = differences.mean(axis=0)
    difference_sd = differences.std(axis=0, ddof=1)
    difference_se = difference_sd / np.sqrt(differences.shape[0])
    difference_table = pd.DataFrame({
        "candidate_n_components": candidates,
        "mean_cross_minus_gsn": difference_mean,
        "paired_fold_standard_deviation": difference_sd,
        "paired_fold_standard_error": difference_se,
    })
    for index, row in enumerate(differences, start=1):
        difference_table[f"fold_cross_minus_gsn_{index}"] = row
    difference_table.to_csv(output_dir / "paired_differences.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    curve_axis.axhline(1, color="grey", linestyle=":", label="Training-mean baseline (K=0)")
    curve_axis.set_ylabel("Validation normalized MSE")
    curve_axis.set_title(f"{subj_id} OTC / {image_set}: selection curves (shading: fold SE)")
    curve_axis.legend(fontsize="small")
    difference_axis.plot(candidates, difference_mean, color="black")
    difference_axis.fill_between(candidates, difference_mean - difference_se,
                                 difference_mean + difference_se, color="grey", alpha=0.3)
    difference_axis.axhline(0, color="grey", linestyle=":")
    difference_axis.set_ylabel("Cross − GSN normalized MSE")
    difference_axis.set_xlabel("Number of reconstruction components K")
    figure.savefig(output_dir / "reconstruction_comparison.pdf")
    figure.clear()
    return summary


def run_comparison(config_path: str | Path = DEFAULT_CONFIG_PATH) -> list[Path]:
    """Load the configured OTC data once per subject and run both selectors.

    Inputs: config_path: str/Path to comparison YAML; relative paths in both
        configurations resolve from the repository root. The source GSN YAML
        supplies participants, data paths, image_set, and all selection kwargs.
    Output: list[Path] of newly created subject run directories. Saves resolved
        configuration, image-fold membership, repeat directions/weights, method
        CSVs, paired differences, summary, and PDF. Does not save projected data.
        This is an expensive real-data entry point; tests inject tiny arrays.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    source_path = REPO_ROOT / config["gsn_config_path"]
    with source_path.open() as stream:
        source = yaml.safe_load(stream)
    if source["functions"]["selection_function"] != "gsn_independent_repeat":
        raise ValueError("The source YAML must select gsn_independent_repeat.")
    options = {"max_components": None, "n_folds": 5, "seed": 0,
               "repeat_split_weighting": "input_repeat_count", **source["selection_kwargs"]}
    if options["seed"] is None:
        raise ValueError("An integer seed is required for paired reproducible folds.")
    solver = config.get("svd_solver", "auto")
    output_root = REPO_ROOT / config["output_dir"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_paths = []
    for subject in source["subjects"]:
        output_dir = output_root / f"{subject}_{source['image_set']}_seed{options['seed']}_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
        data = load_repeat_responses(
            subject, image_set=source["image_set"],
            neural_data_dir=REPO_ROOT / source["neural_data_dir"],
            pkl_info_path=REPO_ROOT / source["pkl_info_path"],
            nonfinite_report_dir=output_dir / "nonfinite",
        )
        responses, image_ids = data["responses"], data["image_ids"]
        if np.unique(image_ids).size != len(image_ids):
            raise ValueError("Duplicate stimulus IDs would leak images across folds.")
        folds, splits, weights = validation_design(
            len(image_ids), responses.shape[2], options["n_folds"],
            options["seed"], options["repeat_split_weighting"],
        )
        # n_images -> (n_images,); each row belongs to one validation fold.
        fold_ids = np.empty(len(image_ids), dtype=int)
        for fold_index, indices in enumerate(folds, start=1):
            fold_ids[indices] = fold_index
        pd.DataFrame({"image_id": image_ids, "validation_fold": fold_ids}).to_csv(
            output_dir / "image_folds.csv", index=False,
        )
        pd.DataFrame([
            {"direction": index + 1, "input_repeat_indices": str(a),
             "target_repeat_indices": str(b), "weight": float(weight)}
            for index, ((a, b), weight) in enumerate(zip(splits, weights))
        ]).to_csv(output_dir / "repeat_splits.csv", index=False)
        metadata = {
            "comparison_config_path": str(config_path.resolve()),
            "gsn_config_path": str(source_path.resolve()), "source_config": source,
            "selection_kwargs": options, "svd_solver": solver,
            "response_shape": list(responses.shape), "region": "OTC",
            "projection_mean": "pooled_training_images_and_repeats",
            "svd_centering": "separate_training_repeat_group_means",
            "scaling": "none", "score_role": "selection_validation_not_final_test",
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "direction_error_rows": "fold-major; within fold follow repeat_splits.csv",
        }
        with (output_dir / "run_config.yaml").open("w") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)
        print(f"{subject}: cross-decomposition, {options}, solver={solver}", flush=True)
        cross = select_cross_decomposition_components(responses, **options, solver=solver)
        print(f"{subject}: unchanged GSN+PCA selector on the same data and folds", flush=True)
        # GSN may use NumPy's legacy global RNG internally; preserve caller state.
        random_state = np.random.get_state()
        try:
            np.random.seed(options["seed"])
            gsn = select_gsn_pca_components(responses, **options)
        finally:
            np.random.set_state(random_state)
        summary = save_comparison(
            {"gsn_pca": gsn, "cross_decomposition": cross}, output_dir,
            subject, source["image_set"],
        )
        print(summary.to_string(index=False))
        print(f"Saved comparison to {output_dir}", flush=True)
        output_paths.append(output_dir)
    return output_paths


if __name__ == "__main__":
    run_comparison()
