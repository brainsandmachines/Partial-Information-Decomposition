#!/bin/bash
# Submit every task of a config as a SLURM array, then one --summarize job that
# runs after the array finishes. Run from the repository root with the conda
# environment active (it is used to list the tasks):
#   bash pipeline/rank_selection/slurm/submit.sh <config.yaml> [conda env] [extra sbatch options...]
# Set ONLY="method ..." to submit only some methods. Example (authors' method on a GPU):
#   ONLY=cross_decomposition bash pipeline/rank_selection/slurm/submit.sh pipeline/rank_selection/configs/otc.yaml cross_decomposition --gpus=1 --partition=rtx6000 --qos=golan-neuro

set -euo pipefail
CONFIG="$1"; ENV="${2:-pid_env}"; shift $(( $# >= 2 ? 2 : 1 ))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p logs/rank_selection

N=$(python -m pipeline.rank_selection.run --config "$CONFIG" --list ${ONLY:+--only $ONLY} | tail -n 1 | awk '{print $1}')
echo "Config $CONFIG: $N tasks, environment $ENV"
ARRAY_ID=$(sbatch --parsable --array=0-$((N - 1)) --export=ALL,CONFIG="$CONFIG",ENV="$ENV",ONLY="${ONLY:-}" "$@" "$SCRIPT_DIR/run.sbatch")
SUMMARY_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_ID" --array=0 --time=1:00:00 \
    --export=ALL,CONFIG="$CONFIG",ENV="$ENV",SUMMARIZE=1 "$SCRIPT_DIR/run.sbatch")
echo "Submitted array job $ARRAY_ID and summary job $SUMMARY_ID."
