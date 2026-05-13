#!/bin/bash
set -euo pipefail

REPO_ROOT="/gscratch/scrubbed/suryadv/repos/Multiplex-Testing"
cd "$REPO_ROOT"

STAMP="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
PASSK_DIR="${REPO_ROOT}/final_eval_outputs/passk-memory-raivn-${STAMP}"
ABLATION_DIR="${REPO_ROOT}/final_eval_outputs/passk-ablation-raivn-${STAMP}"
HYPER_DIR="${REPO_ROOT}/final_eval_outputs/passk-hyperparam-raivn-${STAMP}"
HYPER_MAX_PROMPTS="${HYPER_MAX_PROMPTS:-8}"
HYPER_TOP_P_VALUES="${HYPER_TOP_P_VALUES:-0.75,0.85,0.90,0.95,1.00}"
HYPER_TEMPERATURE_VALUES="${HYPER_TEMPERATURE_VALUES:-0.4,0.6,0.8,1.0,1.2}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_PREP_GPU_GRES="${SLURM_PREP_GPU_GRES:-gpu:1}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"

for job_name in passk-low passk-high passk-ablation passk-hyper passk-runtime-prep; do
  scancel -u "$USER" --name="$job_name" 2>/dev/null || true
done

PREP_ID="$(
  sbatch --parsable \
    --job-name=passk-runtime-prep \
    --account=raivn-ckpt \
    --partition=ckpt-all \
    --gres="$SLURM_PREP_GPU_GRES" \
    --constraint="$SLURM_GPU_CONSTRAINT" \
    --cpus-per-task=4 \
    --time=02:00:00 \
    --export=ALL \
    "${REPO_ROOT}/scripts/prep_runtime.sh"
)"
PREP_ID="${PREP_ID%%;*}"

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
  local dependency="$2"
  local time_limit="$3"
  shift 3
  local job_id
  job_id="$(
    env "$@" sbatch --parsable \
      --job-name="$job_name" \
      --dependency="$dependency" \
      --time="$time_limit" \
      "${SBATCH_COMMON[@]}" \
      "${REPO_ROOT}/run.sh"
  )"
  job_id="${job_id%%;*}"
  echo "[submit] ${job_name}: ${job_id}" >&2
  printf '%s\n' "$job_id"
}

echo "[submit] runtime prep: ${PREP_ID}"
echo "[submit] passk_dir=${PASSK_DIR}"
echo "[submit] ablation_dir=${ABLATION_DIR}"
echo "[submit] hyper_dir=${HYPER_DIR}"
echo "[submit] order=runtime-prep -> passk-ablation -> passk-low/passk-high/passk-hyper"
echo "[submit] hyper_scaled max_prompts=${HYPER_MAX_PROMPTS} top_p=${HYPER_TOP_P_VALUES} temperature=${HYPER_TEMPERATURE_VALUES} time=24:00:00"

ABLATION_ID="$(
submit_job passk-ablation \
  "afterok:${PREP_ID}" \
  "96:00:00" \
  "EXPERIMENT_MODE=branch_ablation" \
  "RUN_TAG=raivn-ablation-${STAMP}" \
  "OUTPUT_DIR=${ABLATION_DIR}" \
  "MAX_K=16" \
  "METHODS=baseline,shared_trace" \
  "BRANCH_ABLATION_REASONING_PREFIX_TOKENS=1024" \
  "BRANCH_ABLATION_GROUP_SIZES=2,4,8,16"
)"

LOW_ID="$(
submit_job passk-low \
  "afterok:${ABLATION_ID}" \
  "96:00:00" \
  "EXPERIMENT_MODE=passk_sweep" \
  "RUN_TAG=raivn-passk-low-${STAMP}" \
  "OUTPUT_DIR=${PASSK_DIR}" \
  "MAX_K=64" \
  "METHODS=baseline,shared_trace" \
  "REASONING_PREFIX_TOKEN_VALUES=256,512,1024"
)"

HIGH_ID="$(
submit_job passk-high \
  "afterok:${ABLATION_ID}" \
  "96:00:00" \
  "EXPERIMENT_MODE=passk_sweep" \
  "RUN_TAG=raivn-passk-high-${STAMP}" \
  "OUTPUT_DIR=${PASSK_DIR}" \
  "MAX_K=64" \
  "METHODS=baseline,shared_trace" \
  "REASONING_PREFIX_TOKEN_VALUES=2048,4096,6144"
)"

submit_job passk-hyper \
  "afterok:${ABLATION_ID}" \
  "24:00:00" \
  "EXPERIMENT_MODE=hyperparam_sweep" \
  "RUN_TAG=raivn-hyper-${STAMP}" \
  "OUTPUT_DIR=${HYPER_DIR}" \
  "MAX_K=8" \
  "MAX_PROMPTS=${HYPER_MAX_PROMPTS}" \
  "METHODS=baseline,shared_trace" \
  "HYPERPARAM_REASONING_PREFIX_TOKENS=1024" \
  "HYPERPARAM_TOP_P_VALUES=${HYPER_TOP_P_VALUES}" \
  "HYPERPARAM_TEMPERATURE_VALUES=${HYPER_TEMPERATURE_VALUES}"
