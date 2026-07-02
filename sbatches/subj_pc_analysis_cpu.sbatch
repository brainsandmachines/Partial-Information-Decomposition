#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
##SBATCH --qos=golan-neuro
#SBATCH --job-name=subj_pc_analysis_cpu
#SBATCH --output=/home/ohadshee/Desktop/Partial-Information-Decomposition/sbatches/%x-%j.out
#SBATCH --error=/home/ohadshee/Desktop/Partial-Information-Decomposition/sbatches/%x-%j.err

# Activate the Python environment
source activate pid_env

# Run the analysis script on CPU
python pipeline/subj_PCs/subj_pc_analysis.py