#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --job-name=ridge_pairwise
#SBATCH --output=/home/ohadshee/Desktop/Partial-Information-Decomposition/logs/ridge_pipeline_logs/%x-%j.out
#SBATCH --error=/home/ohadshee/Desktop/Partial-Information-Decomposition/logs/ridge_pipeline_logs/%x-%j.err

set -euo pipefail

# Move to the project directory.
cd /home/ohadshee/Desktop/Partial-Information-Decomposition

# Load Conda into the sbatch shell.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh. Please check where Conda is installed." >&2
    exit 1
fi

conda activate pid_env

echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not set}"

python -u pipeline/analysis/pca_analysis/all_models_pairwise/ridge_pair_wise_comp.py
