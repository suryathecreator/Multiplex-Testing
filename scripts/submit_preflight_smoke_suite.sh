#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-preflight-smoke-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/${RUN_TAG}}"
ACCOUNT="${ACCOUNT:-raivn-ckpt}"
PARTITION="${PARTITION:-ckpt-all}"
CONSTRAINT="${CONSTRAINT:-l40s}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GRES_SPEC="${GRES_SPEC:-gpu:${GPUS_PER_NODE}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-128G}"
WALLTIME="${WALLTIME:-03:00:00}"
ACCELERATOR_LABEL="${ACCELERATOR_LABEL:-L40S}"
DRY_RUN="${DRY_RUN:-False}"

mkdir -p "$OUTPUT_DIR" slurm_logs

export_args="ALL"
append_export() {
  local name="$1"
  local value="$2"
  export_args+=",${name}=${value}"
}

append_export RUN_TAG "$RUN_TAG"
append_export OUTPUT_DIR "$OUTPUT_DIR"
append_export GPUS_PER_NODE "$GPUS_PER_NODE"
append_export ACCELERATOR_LABEL "$ACCELERATOR_LABEL"

sbatch_args=(
  --parsable
  --job-name="preflight-smoke-${CONSTRAINT}"
  --account="$ACCOUNT"
  --partition="$PARTITION"
  --nodes=1
  --ntasks=1
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEM"
  --time="$WALLTIME"
  --gres="$GRES_SPEC"
  --constraint="$CONSTRAINT"
  --output="${REPO_ROOT}/slurm_logs/%x-%j.out"
  --chdir="$REPO_ROOT"
  --export="$export_args"
  scripts/run_preflight_smoke_suite.sh
)

{
  echo "run_tag=${RUN_TAG}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "account=${ACCOUNT}"
  echo "partition=${PARTITION}"
  echo "constraint=${CONSTRAINT}"
  echo "gres=${GRES_SPEC}"
  echo "mem=${MEM}"
  echo "walltime=${WALLTIME}"
  echo "accelerator_label=${ACCELERATOR_LABEL}"
} > "${OUTPUT_DIR}/preflight_smoke_submission.txt"

if [[ "$DRY_RUN" =~ ^(1|true|TRUE|True|yes|YES|on|ON)$ ]]; then
  {
    printf '[dry-run]'
    printf ' %q' sbatch "${sbatch_args[@]}"
    printf '\n'
  } | tee -a "${OUTPUT_DIR}/preflight_smoke_submission.txt"
  exit 0
fi

job_id="$(sbatch "${sbatch_args[@]}")"
echo "job_id=${job_id}" | tee -a "${OUTPUT_DIR}/preflight_smoke_submission.txt"
echo "[submit] preflight smoke job=${job_id} output_dir=${OUTPUT_DIR}"
