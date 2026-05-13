#!/bin/bash
set -euo pipefail

REPO_ROOT="/gscratch/scrubbed/suryadv/repos/Multiplex-Testing"
cd "$REPO_ROOT"

# Defaults match the currently running ablation job from the 20260424-101750 run.
STAMP="${RUN_TAG:-20260424-101750}"
PASSK_DIR="${REPO_ROOT}/final_eval_outputs/passk-memory-raivn-${STAMP}"
HYPER_DIR="${REPO_ROOT}/final_eval_outputs/passk-hyperparam-raivn-${STAMP}"

LOW_JOB_ID="${LOW_JOB_ID:-34812575}"
HIGH_JOB_ID="${HIGH_JOB_ID:-34812576}"
HYPER_JOB_ID="${HYPER_JOB_ID:-34812577}"

HYPER_MAX_PROMPTS="${HYPER_MAX_PROMPTS:-8}"
HYPER_TOP_P_VALUES="${HYPER_TOP_P_VALUES:-0.75,0.85,0.90,0.95,1.00}"
HYPER_TEMPERATURE_VALUES="${HYPER_TEMPERATURE_VALUES:-0.4,0.6,0.8,1.0,1.2}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"

mkdir -p "$PASSK_DIR" "$HYPER_DIR"

scancel "$LOW_JOB_ID" "$HIGH_JOB_ID" "$HYPER_JOB_ID" 2>/dev/null || true

SBATCH_COMMON=(
  --account=raivn-ckpt
  --partition=ckpt-all
  --gres="$SLURM_GPU_GRES"
  --constraint="$SLURM_GPU_CONSTRAINT"
  --cpus-per-task=4
  --export=ALL
)

submit_job() {
  local job_name="$1"
  local time_limit="$2"
  shift 2
  env "$@" sbatch --parsable \
    --job-name="$job_name" \
    --time="$time_limit" \
    "${SBATCH_COMMON[@]}" \
    "${REPO_ROOT}/run.sh"
}

echo "[resubmit] stamp=${STAMP}"
echo "[resubmit] passk_dir=${PASSK_DIR}"
echo "[resubmit] hyper_dir=${HYPER_DIR}"
echo "[resubmit] cancelling old pending jobs: ${LOW_JOB_ID} ${HIGH_JOB_ID} ${HYPER_JOB_ID}"
echo "[resubmit] submitting passk-low/passk-high/passk-hyper with no dependency"

submit_job passk-low 96:00:00 \
  "EXPERIMENT_MODE=passk_sweep" \
  "RUN_TAG=raivn-passk-low-${STAMP}" \
  "OUTPUT_DIR=${PASSK_DIR}" \
  "MAX_K=64" \
  "METHODS=baseline,shared_trace" \
  "REASONING_PREFIX_TOKEN_VALUES=256,512,1024"

submit_job passk-high 96:00:00 \
  "EXPERIMENT_MODE=passk_sweep" \
  "RUN_TAG=raivn-passk-high-${STAMP}" \
  "OUTPUT_DIR=${PASSK_DIR}" \
  "MAX_K=64" \
  "METHODS=baseline,shared_trace" \
  "REASONING_PREFIX_TOKEN_VALUES=2048,4096,6144"

submit_job passk-hyper 24:00:00 \
  "EXPERIMENT_MODE=hyperparam_sweep" \
  "RUN_TAG=raivn-hyper-${STAMP}" \
  "OUTPUT_DIR=${HYPER_DIR}" \
  "MAX_K=8" \
  "MAX_PROMPTS=${HYPER_MAX_PROMPTS}" \
  "METHODS=baseline,shared_trace" \
  "HYPERPARAM_REASONING_PREFIX_TOKENS=1024" \
  "HYPERPARAM_TOP_P_VALUES=${HYPER_TOP_P_VALUES}" \
  "HYPERPARAM_TEMPERATURE_VALUES=${HYPER_TEMPERATURE_VALUES}"
