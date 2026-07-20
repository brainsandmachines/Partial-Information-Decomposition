#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --job-name=PC_INDEX_ANALYSIS
#SBATCH --output=//home/ohadshee/Desktop/Partial-Information-Decomposition/logs/PC_index_analysis_logs/%x-%j.out
#SBATCH --error=/home/ohadshee/Desktop/Partial-Information-Decomposition/logs/PC_index_analysis_logs/%x-%j.err

# Activate the Python environment
source activate pid_env


# Run the analysis script on CPU with unbuffered output
python -u pipeline/analysis/ridge_analysis/pcIndex_predictions.py