#!/bin/bash
#SBATCH --job-name=rank_selection
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/rank_selection/%x-%A_%a.out
#SBATCH --error=logs/rank_selection/%x-%A_%a.err

# One SLURM array task = one task of the config (see `run --list`).
# Submit from the repository root (logs/rank_selection must exist), e.g.
#   sbatch --array=0-41 --export=ALL,CONFIG=pipeline/rank_selection/configs/otc.yaml pipeline/rank_selection/slurm/run.sbatch
# or use slurm/submit.sh, which also queues the final --summarize job.
# Environment variables:
#   CONFIG     path of the YAML config (required)
#   ENV        conda environment (default pid_env; cross_decomposition for the authors' method)
#   ONLY       optional space-separated method names (same as run --only)
#   CHUNK      tasks per array element (default 1); arrays are limited to 1001 elements
#   SUMMARIZE  1 to rebuild summary.csv and the overview plots instead of running a task

set -euo pipefail
: "${CONFIG:?Set CONFIG=<path to the YAML config>}"
cd "$SLURM_SUBMIT_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh." >&2
    exit 1
fi
conda activate "${ENV:-pid_env}"

echo "Host: $(hostname) | Job: ${SLURM_JOB_ID} | Array task: ${SLURM_ARRAY_TASK_ID:-none} | Env: ${CONDA_DEFAULT_ENV}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

if [ "${SUMMARIZE:-0}" = "1" ]; then
    python -u -m pipeline.rank_selection.run --config "$CONFIG" --summarize
else
    python -u -m pipeline.rank_selection.run --config "$CONFIG" --task-index "$SLURM_ARRAY_TASK_ID" --chunk-size "${CHUNK:-1}" ${ONLY:+--only $ONLY}
fi
