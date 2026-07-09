#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --gpus=1
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --job-name=pair_wise
#SBATCH --output=/home/ohadshee/Desktop/Partial-Information-Decomposition/pair_wise_logs/%x-%j.out
#SBATCH --error=/home/ohadshee/Desktop/Partial-Information-Decomposition/pair_wise_logs/%x-%j.err

# Activate the Python environment
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate /home/ohadshee/.conda/envs/pid_env

# Run the analysis script
python pipeline/analysis/pca_analysis/all_models_pairwise/pair_wise_comp.py
