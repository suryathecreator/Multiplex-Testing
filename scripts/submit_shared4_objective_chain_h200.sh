#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

CHAIN_CHUNKS="${CHAIN_CHUNKS:-4}"
RUN_TAG="${RUN_TAG:-shared4-objective-chain-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/${RUN_TAG}}"
TOTAL_STEPS_PER_VARIANT="${TOTAL_STEPS_PER_VARIANT:-64}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

mkdir -p "$OUTPUT_DIR"

previous_job_id=""
submitted_job_ids=()
for chunk_index in $(seq 1 "$CHAIN_CHUNKS"); do
  target_steps="$(((TOTAL_STEPS_PER_VARIANT * chunk_index + CHAIN_CHUNKS - 1) / CHAIN_CHUNKS))"
  export_args="ALL,RUN_TAG=${RUN_TAG},OUTPUT_DIR=${OUTPUT_DIR},CHAIN_CHUNK_INDEX=${chunk_index},CHAIN_CHUNKS=${CHAIN_CHUNKS},TOTAL_STEPS_PER_VARIANT=${TOTAL_STEPS_PER_VARIANT},TRAIN_TARGET_STEPS=${target_steps},ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
  if [[ -n "$previous_job_id" ]]; then
    job_id="$(sbatch --parsable --dependency="afterany:${previous_job_id}" --export="$export_args" scripts/run_shared4_objective_suite_h200.sh)"
  else
    job_id="$(sbatch --parsable --export="$export_args" scripts/run_shared4_objective_suite_h200.sh)"
  fi
  submitted_job_ids+=("$job_id")
  previous_job_id="$job_id"
  echo "[submit] chunk=${chunk_index}/${CHAIN_CHUNKS} target_steps=${target_steps} job_id=${job_id}"
done

SUMMARY_PATH="${OUTPUT_DIR}/shared4_objective_chain_submission.txt"
{
  echo "run_tag=${RUN_TAG}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "total_steps_per_variant=${TOTAL_STEPS_PER_VARIANT}"
  echo "chain_chunks=${CHAIN_CHUNKS}"
  echo "target_steps=$(seq "$CHAIN_CHUNKS" | awk -v total="$TOTAL_STEPS_PER_VARIANT" -v chunks="$CHAIN_CHUNKS" '{target=int((total*$1+chunks-1)/chunks); print target}' | paste -sd, -)"
  echo "attn_implementation=${ATTN_IMPLEMENTATION}"
  echo "job_ids=${submitted_job_ids[*]}"
} > "$SUMMARY_PATH"
echo "[submit] summary=${SUMMARY_PATH}"
