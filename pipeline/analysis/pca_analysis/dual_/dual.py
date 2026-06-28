"""Run the OTC PID pipeline across ordered pairs of model names."""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path
from typing import Any
import argparse
import yaml
import pandas as pd

repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.full_OTC import otc_experiment
from pipeline.plotting.pairwise_pid_heatmaps import plot_pairwise_pid_matrices


model_1_names = ['nf_resnet50_classification','hardcorenas_f_classification']
model_2_names = ['nf_resnet50_classification','hardcorenas_f_classification']
'''[,'eca_nfnet_l0_classification',
        'resnet50_classification','semnasnet_100_classification','cspresnet50_classification',
        'mobilenetv3_large_100_classification','ghostnet_100_classification','convnext_base_classification','xcit_nano_12_p8_224_classification'
        ,'xcit_nano_12_p16_224_classification','swin_large_patch4_window7_224_classification','jx_nest_tiny_classification',''
        'pit_ti_224_classification','vit_base_patch32_224_classification','vit_base_patch16_224_classification',
        'tnt_s_patch16_224_classification','crossvit_base_240_classification','deit_base_patch16_224_classification',
        'levit_128_classification','coat_lite_tiny_classification','visformer_small_classification',
        'convit_base_classification','ViT-B_32_clip','RN50_clip','RN101_clip','ViT-L_14_clip',
        'ResNet50-SimCLR_selfsupervised','ResNet50-DeepClusterV2-2x224_selfsupervised','ResNet50-SwAV-BS4096-2x224_selfsupervised',
        'ResNet50-PIRL_selfsupervised','ResNet50-ClusterFit-16K-RotNet_selfsupervised','ResNet50-MoCoV2-BS256_selfsupervised',
        'dino_resnet50_selfsupervised','dino_vitb16_selfsupervised']'''

def run_pairwise_pid_pipeline(
    model_1_names: list[str],
    model_2_names: list[str],
    otc_config: dict[str, Any],
    csv_path: str | Path,
) -> Path:
    """Run OTC PID once per unordered model pair and checkpoint results to CSV.

    Inputs:
        model_1_names: list[str], model names to use as source X1.
        model_2_names: list[str], model names to use as source X2.
        otc_config: dict[str, Any], already-loaded OTC pipeline configuration.
        csv_path: str or Path, output CSV used for checkpointing and resuming.

    Output:
        output_path: Path, path to the CSV containing existing and newly
            calculated pair results.

    The function scans the Cartesian product of the two model lists but treats
    (A, B) and (B, A) as the same completed pair. The first unseen orientation
    is evaluated and stored; its unq1 and unq2 values cover both unique-
    information directions. Self-pairs and pairs already present in either CSV
    orientation are skipped. Each successful row is appended immediately.
    Pipeline or save errors are raised after preserving rows completed earlier
    in the run. After all pairs finish, heatmaps are saved in a sibling
    directory named ``<csv_stem>_figures``.
    """

    columns = [
        "model_1",
        "model_2",
        "layer_1",
        "layer_2",
        "subj_id",
        "n_samples",
        "n_components_source_1",
        "n_components_source_2",
        "n_components_target",
        "pid_method",
        "rng_seed",
        "bias_correction",
        "red",
        "unq1",
        "unq2",
        "syn",
        "bi_mi_1",
        "bi_mi_2",
        "tri_mi",
    ]
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        try:
            existing_results = pd.read_csv(output_path)
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"Existing CSV has no header: {output_path}") from error
        if list(existing_results.columns) != columns:
            raise ValueError(
                f"Existing CSV has an incompatible schema: {output_path}. "
                f"Expected columns: {columns}"
            )
    else:
        existing_results = pd.DataFrame(columns=columns)
        existing_results.to_csv(output_path, index=False)

    completed_pairs = {
        frozenset((model_1, model_2))
        for model_1, model_2 in zip(
            existing_results["model_1"].astype(str),
            existing_results["model_2"].astype(str),
        )
    }
    config = deepcopy(otc_config)

    for model_1 in model_1_names:
        for model_2 in model_2_names:
            pair = frozenset((model_1, model_2))
            if model_1 == model_2 or pair in completed_pairs:
                continue

            config["sources_kwargs"]["model_name_1"] = model_1
            config["sources_kwargs"]["model_name_2"] = model_2
            results = otc_experiment.run_otc_experiment(config)
            pid_results = results["pid_results"]
            pid = pid_results["pid"]
            mi = pid_results["mi"]
            selected_layers = results["selected_layers"]
            feature_kwargs = config.get("feature_manipulation_kwargs", {})
            pid_kwargs = config.get("pid_kwargs", {})
            pid_config = pid_kwargs.get("config") or {}

            row = {
                "model_1": model_1,
                "model_2": model_2,
                "layer_1": selected_layers["X1"],
                "layer_2": selected_layers["X2"],
                "subj_id": config.get("target_kwargs", {}).get("subj_id"),
                "n_samples": len(results["target"]),
                "n_components_source_1": feature_kwargs.get("n_components_source_1"),
                "n_components_source_2": feature_kwargs.get("n_components_source_2"),
                "n_components_target": feature_kwargs.get("n_components_target"),
                "pid_method": pid_results.get("method", pid_kwargs.get("method")),
                "rng_seed": pid_kwargs.get("rng_seed"),
                "bias_correction": pid_config.get("bias_correction"),
                "red": pid["red"],
                "unq1": pid["unq1"],
                "unq2": pid["unq2"],
                "syn": pid["syn"],
                "bi_mi_1": mi["bi_mi_1"],
                "bi_mi_2": mi["bi_mi_2"],
                "tri_mi": mi["tri_mi"],
            }
            pd.DataFrame([row], columns=columns).to_csv(
                output_path,
                mode="a",
                header=False,
                index=False,
            )
            completed_pairs.add(pair)

    return output_path




if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Run OTC PID across ordered pairs of model names."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the YAML config file for the OTC pipeline.",
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the output CSV file for checkpointing results.",
    )
    args = parser.parse_args()


    config_path = Path('/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/pipeline/full_OTC/otc_config.yaml')
    csv_path = Path('/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/pipeline/analysis/pca_analysis/dual_')
    plot_path = csv_path / f"{csv_path.stem}_figures.png"
    with open(args.config_path, "r") as f:
        otc_config = yaml.safe_load(f)


    run_pairwise_pid_pipeline(
        model_1_names=model_1_names,
        model_2_names=model_2_names,
        otc_config=otc_config,
        csv_path=csv_path,
    )

    plot_pairwise_pid_matrices(
        csv_path=csv_path,
        output_dir=plot_path,
    )
