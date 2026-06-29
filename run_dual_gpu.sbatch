#!/bin/bash
#SBATCH --job-name=dual_pid_gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --output=slurm-%j-dual-gpu.out
#SBATCH --error=slurm-%j-dual-gpu.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate pid_env

REPO_ROOT="/home/ohadshee/Desktop/Partial-Information-Decomposition"
CONFIG_PATH="$REPO_ROOT/pipeline/full_OTC/otc_config.yaml"
OUTPUT_CSV="$REPO_ROOT/pipeline/analysis/pca_analysis/all_models_pairwise/pairwise_pid_results.csv"

cd "$REPO_ROOT"
python "$REPO_ROOT/pipeline/analysis/pca_analysis/all_models_pairwise/pair_wise_comp.py" "$CONFIG_PATH" "$OUTPUT_CSV"
