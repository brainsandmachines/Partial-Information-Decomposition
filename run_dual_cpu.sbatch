#!/bin/bash
#SBATCH --job-name=dual_pid_cpu
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j-dual-cpu.out
#SBATCH --error=slurm-%j-dual-cpu.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate pid_env

REPO_ROOT="/home/ohadshee/Desktop/Partial-Information-Decomposition"


cd "$REPO_ROOT"
python "$REPO_ROOT/pipeline/analysis/pca_analysis/all_models_pairwise/pair_wise_comp.py" 
