#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_EXECUTABLE="/home/ohadshee/anaconda3/envs/PID_env/bin/python"

export FIGURE5B_OUTPUT_BASENAME="figure5b-eigenvalues_pid_n512_d84_164"
export FIGURE5B_NTRIALS="512"
export FIGURE5B_N_REPETITIONS="100"
export FIGURE5B_BIAS_ITERATIONS="20"
export FIGURE5B_DIMENSIONS="84,92,100,108,116,124,132,140,148,156,164"
export MPLCONFIGDIR="/tmp/lorenz-figure5b-matplotlib"
export XDG_CACHE_HOME="/tmp/lorenz-figure5b-cache"

cd "${PROJECT_ROOT}"
exec "${PYTHON_EXECUTABLE}" -u "${SCRIPT_DIR}/figure5B_recreatation.py"
