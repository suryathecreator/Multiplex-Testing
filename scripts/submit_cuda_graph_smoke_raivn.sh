#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

STAMP="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/cuda-graph-smoke-raivn-${STAMP}}"
JOB_NAME="${JOB_NAME:-cuda-graph-smoke}"
PORT="${PORT:-32031}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-raivn-ckpt}"
SLURM_GPU_PARTITION="${SLURM_GPU_PARTITION:-ckpt-all}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"

mkdir -p slurm_logs
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

SBATCH_ARGS=(
  --account="$SLURM_ACCOUNT"
  --partition="$SLURM_GPU_PARTITION"
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --mem=128G
  --gres="$SLURM_GPU_GRES"
  --constraint="$SLURM_GPU_CONSTRAINT"
  --time=03:00:00
  --export=ALL
)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] job_name=${JOB_NAME}"
  echo "[dry-run] output_dir=${OUTPUT_DIR}"
  echo "[dry-run] account=${SLURM_ACCOUNT} partition=${SLURM_GPU_PARTITION} gres=${SLURM_GPU_GRES} constraint=${SLURM_GPU_CONSTRAINT}"
  echo "[dry-run] MAX_PROMPTS=${MAX_PROMPTS:-2} MAX_K=${MAX_K:-2} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4608} REQUEST_BATCH first=64"
  exit 0
fi

job_id="$(
  env \
    BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
    BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
    MPLCONFIGDIR=/tmp \
    OUTPUT_DIR="$OUTPUT_DIR" \
    RUN_TAG="$JOB_NAME-${STAMP}" \
    PORT="$PORT" \
    SEED="${SEED:-409699}" \
    sbatch --parsable \
      --job-name="$JOB_NAME" \
      --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
      "${SBATCH_ARGS[@]}" \
      "${REPO_ROOT}/scripts/run_cuda_graph_smoke.sh"
)"
job_id="${job_id%%;*}"

echo "[submit] cuda graph smoke job=${job_id}"
echo "[submit] output_dir=${OUTPUT_DIR}"
echo "[submit] log=${REPO_ROOT}/slurm_logs/${JOB_NAME}-${job_id}.out"
