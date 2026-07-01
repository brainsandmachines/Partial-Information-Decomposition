#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --job-name=pca_analysis

# Activate the Python environment
source activate pid_env

# Run the analysis script
python pipeline/analysis/pca_analysis/pca_as_function.py
