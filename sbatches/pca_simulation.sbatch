#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --job-name=pca_simulation
#SBATCH --output=/home/ohadshee/Desktop/Partial-Information-Decomposition/pca_simulation/%x-%j.out
#SBATCH --error=/home/ohadshee/Desktop/Partial-Information-Decomposition/pca_simulation/%x-%j.err

# Activate the Python environment
source activate pid_env

# Run the analysis script
python Simulations/PCA_rank/run_simulation.py
