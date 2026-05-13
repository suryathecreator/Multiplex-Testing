#!/bin/bash
set -euo pipefail

REPO_ROOT="/gscratch/scrubbed/suryadv/repos/Multiplex-Testing"
cd "$REPO_ROOT"

PLAIN_DIR="${PLAIN_DIR:-${REPO_ROOT}/final_eval_outputs/passk-ablation4096-raivn-20260428-003344}"
ADAPTIVE_DIR="${ADAPTIVE_DIR:-${REPO_ROOT}/final_eval_outputs/passk-adaptive-ablation4096-raivn-20260428-003344}"
MAX_PROMPTS="${MAX_PROMPTS:-30}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"

mkdir -p "$PLAIN_DIR" "$ADAPTIVE_DIR" slurm_logs

SBATCH_COMMON=(
  --account=raivn-ckpt
  --partition=ckpt-all
  --cpus-per-task=4
  --mem=128G
  --gres="$SLURM_GPU_GRES"
  --constraint="$SLURM_GPU_CONSTRAINT"
  --time=96:00:00
  --export=ALL
)

echo "[submit] resume plain_dir=${PLAIN_DIR}"
echo "[submit] resume adaptive_dir=${ADAPTIVE_DIR}"
echo "[submit] strict grid: max_k=16 max_prompts=${MAX_PROMPTS} reasoning_budget=4096"
echo "[submit] compact_jsonl=1"

plain_job_id="$(
  env \
    BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
    BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
    MPLCONFIGDIR=/tmp \
    COMPACT_JSONL=1 \
    EXPERIMENT_MODE=branch_ablation \
    RUN_TAG="raivn-ablation4096-strict-resume" \
    OUTPUT_DIR="$PLAIN_DIR" \
    PORT=30220 \
    MAX_K=16 \
    MAX_PROMPTS="${MAX_PROMPTS}" \
    METHODS=baseline,shared_trace \
    BRANCH_ABLATION_REASONING_PREFIX_TOKENS=4096 \
    BRANCH_ABLATION_GROUP_SIZES=2,4,8,16 \
    sbatch --parsable \
      --job-name=ab4096-plain-r \
      "${SBATCH_COMMON[@]}" \
      "${REPO_ROOT}/run.sh"
)"
plain_job_id="${plain_job_id%%;*}"

adaptive_job_id="$(
  env \
    BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
    BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
    MPLCONFIGDIR=/tmp \
    COMPACT_JSONL=1 \
    EXPERIMENT_MODE=adaptive_ablation \
    RUN_TAG="raivn-adaptive4096-strict-resume" \
    OUTPUT_DIR="$ADAPTIVE_DIR" \
    PORT=30221 \
    MAX_K=16 \
    MAX_PROMPTS="${MAX_PROMPTS}" \
    METHODS=baseline,shared_trace \
    BRANCH_ABLATION_REASONING_PREFIX_TOKENS=4096 \
    ADAPTIVE_ABLATION_SHARED_COUNTS=2,4,8,16 \
    ADAPTIVE_CONFIDENCE_THRESHOLD=0.75 \
    sbatch --parsable \
      --job-name=ab4096-adapt-r \
      "${SBATCH_COMMON[@]}" \
      "${REPO_ROOT}/run.sh"
)"
adaptive_job_id="${adaptive_job_id%%;*}"

echo "[submit] plain_ablation_job=${plain_job_id}"
echo "[submit] adaptive_ablation_job=${adaptive_job_id}"
echo "[submit] done"
