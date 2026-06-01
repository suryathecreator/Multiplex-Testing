#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p slurm_logs final_eval_outputs
sbatch "$@" scripts/run_six_hour_multiplex_repro_h200.sh
