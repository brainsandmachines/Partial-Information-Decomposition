#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=pca_simulation

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate /home/ohadshee/.conda/envs/pid_env
export PYTHONNOUSERSITE=1

which python
python -c "import sys; print(sys.executable)"

python -u Simulations/PCA_rank/run_simulation.py
echo "Python finished at $(date)"