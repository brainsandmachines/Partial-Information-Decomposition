#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --job-name=pca_unq1_search

# Activate the Python environment
source activate pid_env

# Run the analysis script
python pipeline/analysis/pca_analysis/otc_unique_search_pca.py
