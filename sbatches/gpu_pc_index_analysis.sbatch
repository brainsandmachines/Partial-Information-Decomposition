#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --job-name=GPU_PC_INDEX_ANALYSIS
#SBATCH --output=//home/ohadshee/Desktop/Partial-Information-Decomposition/logs/PC_index_analysis_logs/%x-%j.out
#SBATCH --error=/home/ohadshee/Desktop/Partial-Information-Decomposition/logs/PC_index_analysis_logs/%x-%j.err

set -euo pipefail

# Move to the project directory
cd /home/ohadshee/Desktop/Partial-Information-Decomposition

# Load Conda into the sbatch shell
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

# Activate the environment
conda activate pid_env

# Useful debugging info in the .out file
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"


# Run the analysis script on GPU with unbuffered output
python -u pipeline/analysis/ridge_analysis/pcIndex_predictions.py