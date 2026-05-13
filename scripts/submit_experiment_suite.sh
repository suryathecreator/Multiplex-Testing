#!/bin/bash
set -euo pipefail

REPO_ROOT="/gscratch/scrubbed/suryadv/repos/Multiplex-Testing"
STAMP="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

PASSK_OUTPUT_DIR="${PASSK_OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/passk-memory-${STAMP}}"
ABLATION_OUTPUT_DIR="${ABLATION_OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/branch-ablation-${STAMP}}"
HYPERPARAM_OUTPUT_DIR="${HYPERPARAM_OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/hyperparam-sweep-${STAMP}}"

mkdir -p "$PASSK_OUTPUT_DIR" "$ABLATION_OUTPUT_DIR" "$HYPERPARAM_OUTPUT_DIR"

submit_job() {
  local name="$1"
  shift
  export \
    EXPERIMENT_MODE \
    OUTPUT_DIR \
    RUN_TAG \
    MAX_K \
    METHODS \
    REASONING_PREFIX_TOKEN_VALUES \
    BRANCH_ABLATION_REASONING_PREFIX_TOKENS \
    BRANCH_ABLATION_GROUP_SIZES \
    HYPERPARAM_REASONING_PREFIX_TOKENS \
    HYPERPARAM_TOP_P_VALUES \
    HYPERPARAM_TEMPERATURE_VALUES
  sbatch \
    --job-name="$name" \
    --export=ALL \
    "$@" \
    "${REPO_ROOT}/run.sh"
}

echo "[submit] passk_output_dir=${PASSK_OUTPUT_DIR}"
echo "[submit] ablation_output_dir=${ABLATION_OUTPUT_DIR}"
echo "[submit] hyperparam_output_dir=${HYPERPARAM_OUTPUT_DIR}"

EXPERIMENT_MODE=passk_sweep \
OUTPUT_DIR="$PASSK_OUTPUT_DIR" \
RUN_TAG="${STAMP}-passk-low" \
MAX_K=64 \
METHODS=baseline,shared_trace \
REASONING_PREFIX_TOKEN_VALUES=256,512,1024 \
submit_job "passk-low"

EXPERIMENT_MODE=passk_sweep \
OUTPUT_DIR="$PASSK_OUTPUT_DIR" \
RUN_TAG="${STAMP}-passk-high" \
MAX_K=64 \
METHODS=baseline,shared_trace \
REASONING_PREFIX_TOKEN_VALUES=2048,4096,6144 \
submit_job "passk-high"

EXPERIMENT_MODE=branch_ablation \
OUTPUT_DIR="$ABLATION_OUTPUT_DIR" \
RUN_TAG="${STAMP}-ablation" \
MAX_K=16 \
METHODS=baseline,shared_trace \
BRANCH_ABLATION_REASONING_PREFIX_TOKENS=1024 \
BRANCH_ABLATION_GROUP_SIZES=2,4,8,16 \
submit_job "passk-ablation"

EXPERIMENT_MODE=hyperparam_sweep \
OUTPUT_DIR="$HYPERPARAM_OUTPUT_DIR" \
RUN_TAG="${STAMP}-hyperparam" \
MAX_K=8 \
METHODS=baseline,shared_trace \
HYPERPARAM_REASONING_PREFIX_TOKENS=1024 \
HYPERPARAM_TOP_P_VALUES=0.75,0.85,0.90,0.95,1.00 \
HYPERPARAM_TEMPERATURE_VALUES=0.4,0.6,0.8,1.0,1.2 \
submit_job "passk-hyper"
