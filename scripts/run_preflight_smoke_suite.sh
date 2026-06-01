#!/bin/bash
set -euo pipefail

START_TIME="$(date +%s)"
REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-preflight-smoke-${SLURM_JOB_ID:-local}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/${RUN_TAG}}"
SMOKE_SUMMARY="${OUTPUT_DIR}/preflight_smoke_summary.txt"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
ACCELERATOR_LABEL="${ACCELERATOR_LABEL:-L40S}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gscratch/scrubbed/suryadv}"
MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/.cache/multiplex-thinking/local-models/DeepSeek-R1-Distill-Qwen-1.5B}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
fi

mkdir -p "$OUTPUT_DIR"

append_summary() {
  printf '%s\n' "$*" | tee -a "$SMOKE_SUMMARY"
}

run_suite_step() {
  local label="$1"
  shift
  append_summary "[step] ${label} started $(date --iso-8601=seconds)"
  "$@"
  append_summary "[step] ${label} finished $(date --iso-8601=seconds)"
}

common_env=(
  REPO_ROOT="$REPO_ROOT"
  RUN_TAG="$RUN_TAG"
  OUTPUT_DIR="$OUTPUT_DIR"
  MODEL_PATH="$MODEL_PATH"
  GPUS_PER_NODE="$GPUS_PER_NODE"
  ACCELERATOR_LABEL="$ACCELERATOR_LABEL"
  ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
  ENFORCE_EAGER=True
  AUTO_OOM_RETRY=False
  SAVE_HF_MODEL=True
  CHECKPOINT_SAVE_CONTENTS=
  CHECKPOINT_LOAD_CONTENTS=
)

eval_env=(
  TASK_MODE=eval
  EVAL_MAX_K="${SMOKE_EVAL_MAX_K:-1}"
  EVAL_MAX_PROMPTS="${SMOKE_EVAL_MAX_PROMPTS:-1}"
  EVAL_MAX_NEW_TOKENS="${SMOKE_EVAL_MAX_NEW_TOKENS:-192}"
  REASONING_PREFIX_TOKENS="${SMOKE_REASONING_PREFIX_TOKENS:-64}"
  EVAL_REQUEST_BATCH_SIZE="${SMOKE_EVAL_REQUEST_BATCH_SIZE:-1}"
  EVAL_MAX_CONCURRENT_PROMPTS="${SMOKE_EVAL_MAX_CONCURRENT_PROMPTS:-1}"
  EVAL_SERVER_TIMEOUT_SECONDS="${SMOKE_EVAL_SERVER_TIMEOUT_SECONDS:-900}"
  EVAL_DISABLE_CUDA_GRAPH=True
  EVAL_MEM_FRACTION_STATIC="${SMOKE_EVAL_MEM_FRACTION_STATIC:-0.62}"
)

{
  echo "run_tag=${RUN_TAG}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "model_path=${MODEL_PATH}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "accelerator_label=${ACCELERATOR_LABEL}"
  echo "slurm_job_id=${SLURM_JOB_ID:-local}"
  echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
} > "$SMOKE_SUMMARY"

append_summary "[preflight] H200 production config smoke checks"
run_suite_step "h200-config-multiplex" env \
  CONFIG_SMOKE_ONLY=True \
  TASK_MODE=train \
  RUN_CONDITIONS=multiplex_thinking \
  TOTAL_STEPS_PER_VARIANT=40 \
  TRAIN_TARGET_STEPS=20 \
  TRAIN_EXAMPLES=40000 \
  SHARD_STEPS=20 \
  GPUS_PER_NODE=4 \
  ACCELERATOR_LABEL=H200 \
  ATTN_IMPLEMENTATION=flash_attention_2 \
  AUTO_OOM_RETRY=True \
  ENFORCE_EAGER=True \
  SAVE_HF_MODEL=True \
  TRAIN_BATCH_SIZE=32 \
  PPO_MINI_BATCH_SIZE=16 \
  ROLLOUT_N=4 \
  MAX_RESPONSE_LENGTH=1024 \
  MAX_TOKEN_LEN_PER_GPU=12288 \
  ROLLOUT_MAX_NUM_SEQS=128 \
  GPU_MEM_UTIL=0.68 \
  SAVE_FREQ=4 \
  bash scripts/run_shared4_objective_suite_h200.sh

run_suite_step "h200-config-shared4" env \
  CONFIG_SMOKE_ONLY=True \
  TASK_MODE=train \
  RUN_CONDITIONS=shared4_joint \
  TOTAL_STEPS_PER_VARIANT=40 \
  TRAIN_TARGET_STEPS=20 \
  TRAIN_EXAMPLES=40000 \
  SHARD_STEPS=20 \
  GPUS_PER_NODE=4 \
  ACCELERATOR_LABEL=H200 \
  ATTN_IMPLEMENTATION=flash_attention_2 \
  AUTO_OOM_RETRY=True \
  ENFORCE_EAGER=True \
  SAVE_HF_MODEL=True \
  TRAIN_BATCH_SIZE=32 \
  PPO_MINI_BATCH_SIZE=16 \
  ROLLOUT_N=16 \
  MAX_RESPONSE_LENGTH=4096 \
  MAX_TOKEN_LEN_PER_GPU=20480 \
  ROLLOUT_MAX_NUM_SEQS=96 \
  GPU_MEM_UTIL=0.68 \
  SAVE_FREQ=4 \
  bash scripts/run_shared4_objective_suite_h200.sh

append_summary "[preflight] fallback runtime smoke on allocated GPU"
run_suite_step "flash-attn-runtime" env \
  "${common_env[@]}" \
  FLASH_ATTN_RUNTIME_SMOKE_ONLY=True \
  TASK_MODE=train \
  RUN_CONDITIONS=multiplex_thinking \
  bash scripts/run_shared4_objective_suite_h200.sh

append_summary "[preflight] scaled training smoke with production batching invariants"
run_suite_step "train-multiplex-thinking" env \
  "${common_env[@]}" \
  TASK_MODE=train \
  RUN_CONDITIONS=multiplex_thinking \
  TOTAL_STEPS_PER_VARIANT=1 \
  TRAIN_TARGET_STEPS=1 \
  TRAIN_EXAMPLES="${SMOKE_TRAIN_EXAMPLES:-64}" \
  SHARD_STEPS=1 \
  TRAIN_BATCH_SIZE=32 \
  PPO_MINI_BATCH_SIZE=16 \
  ROLLOUT_N=4 \
  MAX_RESPONSE_LENGTH="${SMOKE_MULTIPLEX_MAX_RESPONSE_LENGTH:-512}" \
  MAX_TOKEN_LEN_PER_GPU="${SMOKE_MULTIPLEX_MAX_TOKEN_LEN_PER_GPU:-8192}" \
  ROLLOUT_MAX_NUM_SEQS="${SMOKE_MULTIPLEX_ROLLOUT_MAX_NUM_SEQS:-64}" \
  GPU_MEM_UTIL="${SMOKE_MULTIPLEX_GPU_MEM_UTIL:-0.62}" \
  SAVE_FREQ=1 \
  bash scripts/run_shared4_objective_suite_h200.sh

run_suite_step "train-shared4-joint" env \
  "${common_env[@]}" \
  TASK_MODE=train \
  RUN_CONDITIONS=shared4_joint \
  TOTAL_STEPS_PER_VARIANT=1 \
  TRAIN_TARGET_STEPS=1 \
  TRAIN_EXAMPLES="${SMOKE_TRAIN_EXAMPLES:-64}" \
  SHARD_STEPS=1 \
  TRAIN_BATCH_SIZE=32 \
  PPO_MINI_BATCH_SIZE=16 \
  ROLLOUT_N=16 \
  MAX_RESPONSE_LENGTH="${SMOKE_SHARED4_MAX_RESPONSE_LENGTH:-512}" \
  MAX_TOKEN_LEN_PER_GPU="${SMOKE_SHARED4_MAX_TOKEN_LEN_PER_GPU:-8192}" \
  ROLLOUT_MAX_NUM_SEQS="${SMOKE_SHARED4_ROLLOUT_MAX_NUM_SEQS:-32}" \
  GPU_MEM_UTIL="${SMOKE_SHARED4_GPU_MEM_UTIL:-0.60}" \
  SAVE_FREQ=1 \
  bash scripts/run_shared4_objective_suite_h200.sh

append_summary "[preflight] eval startup/checkpoint smoke"
run_suite_step "eval-multiplex-untrained" env \
  "${common_env[@]}" \
  "${eval_env[@]}" \
  EVAL_TASK=multiplex_untrained \
  EVAL_REPEAT_LABEL=repeat_0 \
  EVAL_PORT="${SMOKE_EVAL_PORT_MU:-36201}" \
  bash scripts/run_shared4_objective_suite_h200.sh

run_suite_step "eval-multiplex-trained" env \
  "${common_env[@]}" \
  "${eval_env[@]}" \
  EVAL_TASK=multiplex_trained \
  EVAL_REPEAT_LABEL=repeat_0 \
  EVAL_PORT="${SMOKE_EVAL_PORT_MT:-36202}" \
  bash scripts/run_shared4_objective_suite_h200.sh

run_suite_step "eval-shared4-untrained" env \
  "${common_env[@]}" \
  "${eval_env[@]}" \
  EVAL_TASK=shared4_untrained \
  EVAL_REPEAT_LABEL=repeat_0 \
  EVAL_PORT="${SMOKE_EVAL_PORT_S4U:-36203}" \
  bash scripts/run_shared4_objective_suite_h200.sh

run_suite_step "eval-shared4-joint-trained" env \
  "${common_env[@]}" \
  "${eval_env[@]}" \
  EVAL_TASK=shared4_joint_trained \
  EVAL_REPEAT_LABEL=repeat_0 \
  EVAL_PORT="${SMOKE_EVAL_PORT_S4J:-36204}" \
  bash scripts/run_shared4_objective_suite_h200.sh

append_summary "[preflight] aggregate plot smoke"
run_suite_step "aggregate" env \
  "${common_env[@]}" \
  TASK_MODE=aggregate \
  AGGREGATE_BOOTSTRAP_RUNS="${SMOKE_AGGREGATE_BOOTSTRAP_RUNS:-10}" \
  AGGREGATE_LABEL=preflight_smoke \
  bash scripts/run_shared4_objective_suite_h200.sh

END_TIME="$(date +%s)"
append_summary "wall_seconds=$((END_TIME - START_TIME))"
append_summary "status=success"
append_summary "[preflight] complete $(date --iso-8601=seconds)"
