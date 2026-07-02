#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --job-name=pca_unq1_search
#SBATCH --output=/home/ohadshee/Desktop/Partial-Information-Decomposition/otc_run/%x-%j.out
#SBATCH --error=/home/ohadshee/Desktop/Partial-Information-Decomposition/otc_run/%x-%j.err

# Activate the Python environment
source activate pid_env

# Run the analysis script
python pipeline/full_OTC/otc_run.py
