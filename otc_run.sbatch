#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --job-name=pca_unq1_search

# Activate the Python environment
source activate pid_env

# Run the analysis script
python pipeline/full_OTC/otc_run.py
