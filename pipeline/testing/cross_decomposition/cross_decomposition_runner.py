"""Orchestrate cross-decomposition rank experiments and save their outputs."""

import argparse
from pathlib import Path
from typing import Any

from pipeline.testing.cross_decomposition.cross_decomposition_inputs import (
    DEFAULT_CONFIG_PATH,
    assert_caching_disabled,
    load_config,
    load_cross_decomposition_inputs,
)

from lib.spectra._computation import _get_folds

from pipeline.testing.cross_decomposition.cross_decomposition_criteria import (
    calculate_authors_spectra,
    calculate_reconstruction_criterion,
    verify_ground_truth,
)
from pipeline.testing.cross_decomposition.cross_decomposition_outputs import (
    save_outputs,
    summarize_criteria,
)


def run_cross_decomposition_criteria(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the configured rank calculations on synthetic or OTC responses.

    Inputs:
        config_path: str or Path to the experiment YAML.
        smoke: bool selecting the small permutation count from the YAML.

    Output:
        results: dict[str, Any] containing data-source metadata, the authors'
            binned and unbinned spectra, rank-significance tables, fold labels,
            and a summary. Synthetic runs additionally contain verified ground
            truth and the separate reconstruction-rank criterion.
    """

    config = load_config(config_path)
    data_config = config["data"]
    cross_decomposition_config = config["cross_decomposition"]
    permutation_config = config["permutation"]
    n_permutations = (
        config["smoke_n_permutations"]
        if smoke
        else permutation_config["n_permutations"]
    )
    assert_caching_disabled()

    inputs = load_cross_decomposition_inputs(data_config)
    x = inputs["x"]
    y = inputs["y"]
    true_rank = inputs["true_rank"]
    ground_truth = None
    if inputs["synthetic_data"] is not None:
        ground_truth = verify_ground_truth(inputs["synthetic_data"], true_rank)

    spectra_results = calculate_authors_spectra(
        x,
        y,
        cross_decomposition_config,
        permutation_config,
        n_permutations,
    )
    # Scalar presentation count -> (presentations,)
    folds = _get_folds(
        n_stimuli=x.sizes["presentation"],
        n_folds=cross_decomposition_config["n_folds"],
        seed=cross_decomposition_config["seed"],
    )
    reconstruction = None
    if (
        inputs["data_metadata"]["source"] == "synthetic"
        and config["reconstruction"]["enabled_for_synthetic"]
    ):
        reconstruction = calculate_reconstruction_criterion(
            x,
            y,
            folds,
            randomized=cross_decomposition_config["randomized"],
            seed=cross_decomposition_config["seed"],
        )
    summary = summarize_criteria(
        spectra_results["bin_table"],
        spectra_results["rank_table"],
        true_rank,
        None if reconstruction is None else reconstruction["k_reconstruction"],
    )
    return {
        "config": config,
        "n_permutations": n_permutations,
        "data_metadata": inputs["data_metadata"],
        "image_ids": inputs["image_ids"],
        "synthetic_data": inputs["synthetic_data"],
        "ground_truth": ground_truth,
        "true_rank": true_rank,
        **spectra_results,
        "reconstruction": reconstruction,
        "folds": folds,
        "summary": summary,
    }


def main(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the rank experiment, save deliverables, and print its summary.

    Inputs:
        config_path: str or Path to the experiment YAML.
        smoke: bool selecting the small permutation count and smoke output path.

    Output:
        results: dict[str, Any] from ``run_cross_decomposition_criteria`` with
            an added ``output_dir`` Path.
    """

    results = run_cross_decomposition_criteria(config_path, smoke=smoke)
    results["output_dir"] = save_outputs(results, smoke=smoke)
    print(results["summary"][["quantity", "result"]].to_string(index=False))
    print(f"Outputs: {results['output_dir']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    main(arguments.config, smoke=arguments.smoke)
