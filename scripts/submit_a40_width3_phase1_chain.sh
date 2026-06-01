#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-a40-s4j-mt-w3-4x4-12step-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/${RUN_TAG}}"

ACCOUNT="${ACCOUNT:-raivn}"
PARTITION="${PARTITION:-gpu-a40}"
CONSTRAINT="${CONSTRAINT:-a40}"
JOB_SUFFIX="${JOB_SUFFIX:-a40w3}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TRAIN_CPUS_PER_TASK="${TRAIN_CPUS_PER_TASK:-32}"
TRAIN_MEM="${TRAIN_MEM:-256G}"
TRAIN_WALLTIME="${TRAIN_WALLTIME:-04:15:00}"
EVAL_CPUS_PER_TASK="${EVAL_CPUS_PER_TASK:-32}"
EVAL_MEM="${EVAL_MEM:-256G}"
EVAL_WALLTIME="${EVAL_WALLTIME:-03:00:00}"
POST_CPUS_PER_TASK="${POST_CPUS_PER_TASK:-16}"
POST_MEM="${POST_MEM:-128G}"
POSTPROCESS_WALLTIME="${POSTPROCESS_WALLTIME:-00:30:00}"

TOTAL_STEPS_PER_VARIANT="${TOTAL_STEPS_PER_VARIANT:-12}"
TRAIN_TARGET_STEPS="${TRAIN_TARGET_STEPS:-$TOTAL_STEPS_PER_VARIANT}"
SHARD_STEPS="${SHARD_STEPS:-$TRAIN_TARGET_STEPS}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-ALL}"
USE_FULL_TRAIN_SET="${USE_FULL_TRAIN_SET:-True}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-16}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-48}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.58}"
SAVE_FREQ="${SAVE_FREQ:-4}"
EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-1536}"
BRANCH_ROLLOUT_THINKING_TOKENS="${BRANCH_ROLLOUT_THINKING_TOKENS:-1536}"
BRANCH_ROLLOUT_CONTINUATION_TOKENS="${BRANCH_ROLLOUT_CONTINUATION_TOKENS:-511}"

MULTIPLEX_WIDTH_OVERRIDE="${MULTIPLEX_WIDTH_OVERRIDE:-3}"
SHARED4_MULTIPLEX_WIDTH_OVERRIDE="${SHARED4_MULTIPLEX_WIDTH_OVERRIDE:-3}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
ROLLOUT_NAME="${ROLLOUT_NAME:-soft_hf}"
PYTORCH_ROLLOUT_TIMEOUT_SECONDS="${PYTORCH_ROLLOUT_TIMEOUT_SECONDS:-900}"
PYTORCH_ROLLOUT_DEBUG="${PYTORCH_ROLLOUT_DEBUG:-True}"
RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
AUTO_OOM_RETRY="${AUTO_OOM_RETRY:-True}"
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-True}"
NORMALIZE_CUDA_VISIBLE_DEVICES="${NORMALIZE_CUDA_VISIBLE_DEVICES:-False}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-False}"
ENABLE_SLEEP_HACK="${ENABLE_SLEEP_HACK:-False}"
ENABLE_MEMORY_SAVER="${ENABLE_MEMORY_SAVER:-False}"
SGLANG_DISABLE_OVERLAP_SCHEDULE="${SGLANG_DISABLE_OVERLAP_SCHEDULE:-True}"
SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-pytorch}"

EVAL_SEED="${EVAL_SEED:-26010808}"
EVAL_REPEAT_LABEL="${EVAL_REPEAT_LABEL:-repeat_00}"
EVAL_MAX_K="${EVAL_MAX_K:-8}"
EVAL_MAX_PROMPTS="${EVAL_MAX_PROMPTS:-30}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-3072}"
REASONING_PREFIX_TOKENS="${REASONING_PREFIX_TOKENS:-2048}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-16}"
EVAL_MAX_CONCURRENT_PROMPTS="${EVAL_MAX_CONCURRENT_PROMPTS:-2}"
EVAL_DISABLE_CUDA_GRAPH="${EVAL_DISABLE_CUDA_GRAPH:-True}"
EVAL_MEM_FRACTION_STATIC="${EVAL_MEM_FRACTION_STATIC:-0.78}"
AGGREGATE_BOOTSTRAP_RUNS="${AGGREGATE_BOOTSTRAP_RUNS:-1000}"
AGGREGATE_SEED="${AGGREGATE_SEED:-26010808}"
AGGREGATE_LABEL="${AGGREGATE_LABEL:-phase1_a40_width3_12step}"

DRY_RUN="${DRY_RUN:-False}"

mkdir -p "$OUTPUT_DIR" slurm_logs

truthy() {
  [[ "${1:-}" =~ ^(1|true|TRUE|True|yes|YES|on|ON)$ ]]
}

append_export() {
  local name="$1"
  local value="$2"
  export_args+=",${name}=${value}"
}

submit_or_echo() {
  if truthy "$DRY_RUN"; then
    {
      printf '[dry-run]'
      printf ' %q' sbatch "$@"
      printf '\n'
    } >&2
    echo "DRYRUN"
  else
    sbatch "$@"
  fi
}

eval_slug() {
  case "$1" in
    multiplex_untrained) echo "mu" ;;
    multiplex_trained) echo "mt" ;;
    shared4_untrained) echo "s4u" ;;
    shared4_joint_trained) echo "s4jt" ;;
    *) echo "[config] unknown eval task: $1" >&2; exit 1 ;;
  esac
}

export_args="ALL"
append_export RUN_TAG "$RUN_TAG"
append_export OUTPUT_DIR "$OUTPUT_DIR"
append_export TASK_MODE train
append_export RUN_CONDITIONS "shared4_joint+multiplex_thinking"
append_export TOTAL_STEPS_PER_VARIANT "$TOTAL_STEPS_PER_VARIANT"
append_export TRAIN_TARGET_STEPS "$TRAIN_TARGET_STEPS"
append_export SHARD_STEPS "$SHARD_STEPS"
append_export TRAIN_EXAMPLES "$TRAIN_EXAMPLES"
append_export USE_FULL_TRAIN_SET "$USE_FULL_TRAIN_SET"
append_export GPUS_PER_NODE "$GPUS_PER_NODE"
append_export ACCELERATOR_LABEL A40
append_export TRAIN_BATCH_SIZE "$TRAIN_BATCH_SIZE"
append_export PPO_MINI_BATCH_SIZE "$PPO_MINI_BATCH_SIZE"
append_export ROLLOUT_N "$ROLLOUT_N"
append_export ROLLOUT_NAME "$ROLLOUT_NAME"
append_export PYTORCH_ROLLOUT_TIMEOUT_SECONDS "$PYTORCH_ROLLOUT_TIMEOUT_SECONDS"
append_export PYTORCH_ROLLOUT_DEBUG "$PYTORCH_ROLLOUT_DEBUG"
append_export RAY_DEDUP_LOGS "$RAY_DEDUP_LOGS"
append_export MAX_RESPONSE_LENGTH "$MAX_RESPONSE_LENGTH"
append_export MAX_TOKEN_LEN_PER_GPU "$MAX_TOKEN_LEN_PER_GPU"
append_export ROLLOUT_MAX_NUM_SEQS "$ROLLOUT_MAX_NUM_SEQS"
append_export GPU_MEM_UTIL "$GPU_MEM_UTIL"
append_export SAVE_FREQ "$SAVE_FREQ"
append_export SAVE_HF_MODEL "$SAVE_HF_MODEL"
append_export NORMALIZE_CUDA_VISIBLE_DEVICES "$NORMALIZE_CUDA_VISIBLE_DEVICES"
append_export FREE_CACHE_ENGINE "$FREE_CACHE_ENGINE"
append_export ENABLE_SLEEP_HACK "$ENABLE_SLEEP_HACK"
append_export ENABLE_MEMORY_SAVER "$ENABLE_MEMORY_SAVER"
append_export SGLANG_DISABLE_OVERLAP_SCHEDULE "$SGLANG_DISABLE_OVERLAP_SCHEDULE"
append_export SGLANG_SAMPLING_BACKEND "$SGLANG_SAMPLING_BACKEND"
append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
append_export AUTO_OOM_RETRY "$AUTO_OOM_RETRY"
append_export ENFORCE_EAGER "$ENFORCE_EAGER"
append_export EARLY_STOPPING_LENGTH_THRESHOLD "$EARLY_STOPPING_LENGTH_THRESHOLD"
append_export FORCE_THINK_END_AT_LENGTH True
append_export MULTIPLEX_WIDTH_OVERRIDE "$MULTIPLEX_WIDTH_OVERRIDE"
append_export SHARED4_MULTIPLEX_WIDTH_OVERRIDE "$SHARED4_MULTIPLEX_WIDTH_OVERRIDE"
append_export BRANCH_ROLLOUT True
append_export BRANCH_ROLLOUT_THINKING_TRACES 4
append_export BRANCH_ROLLOUT_ANSWERS_PER_TRACE 4
append_export BRANCH_ROLLOUT_THINKING_TOKENS "$BRANCH_ROLLOUT_THINKING_TOKENS"
append_export BRANCH_ROLLOUT_CONTINUATION_TOKENS "$BRANCH_ROLLOUT_CONTINUATION_TOKENS"

train_job_id="$(submit_or_echo \
  --parsable \
  --job-name="s4j-mt-w3-a40-12step" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$TRAIN_CPUS_PER_TASK" \
  --mem="$TRAIN_MEM" \
  --time="$TRAIN_WALLTIME" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --constraint="$CONSTRAINT" \
  --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
  --chdir="$REPO_ROOT" \
  --export="$export_args" \
  scripts/run_shared4_objective_suite_h200.sh)"
echo "[submit] train=${train_job_id}"

eval_job_ids=()
eval_index=0
for eval_task in multiplex_untrained multiplex_trained shared4_untrained shared4_joint_trained; do
  slug="$(eval_slug "$eval_task")"
  port="$((37101 + eval_index))"
  export_args="ALL"
  append_export RUN_TAG "$RUN_TAG"
  append_export OUTPUT_DIR "$OUTPUT_DIR"
  append_export TASK_MODE eval
  append_export GPUS_PER_NODE "$GPUS_PER_NODE"
  append_export ACCELERATOR_LABEL A40
  append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
  append_export SGLANG_SAMPLING_BACKEND "$SGLANG_SAMPLING_BACKEND"
  append_export EVAL_TASK "$eval_task"
  append_export EVAL_SEED "$EVAL_SEED"
  append_export EVAL_REPEAT_LABEL "$EVAL_REPEAT_LABEL"
  append_export EVAL_PORT "$port"
  append_export EVAL_MAX_K "$EVAL_MAX_K"
  append_export EVAL_MAX_PROMPTS "$EVAL_MAX_PROMPTS"
  append_export EVAL_MAX_NEW_TOKENS "$EVAL_MAX_NEW_TOKENS"
  append_export REASONING_PREFIX_TOKENS "$REASONING_PREFIX_TOKENS"
  append_export EVAL_REQUEST_BATCH_SIZE "$EVAL_REQUEST_BATCH_SIZE"
  append_export EVAL_MAX_CONCURRENT_PROMPTS "$EVAL_MAX_CONCURRENT_PROMPTS"
  append_export EVAL_DISABLE_CUDA_GRAPH "$EVAL_DISABLE_CUDA_GRAPH"
  append_export EVAL_MEM_FRACTION_STATIC "$EVAL_MEM_FRACTION_STATIC"
  append_export FREE_CACHE_ENGINE "$FREE_CACHE_ENGINE"
  append_export ENABLE_SLEEP_HACK "$ENABLE_SLEEP_HACK"
  append_export ENABLE_MEMORY_SAVER "$ENABLE_MEMORY_SAVER"
  append_export SGLANG_DISABLE_OVERLAP_SCHEDULE "$SGLANG_DISABLE_OVERLAP_SCHEDULE"

  eval_job_id="$(submit_or_echo \
    --parsable \
    --job-name="eval-${slug}-w3-a40" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$EVAL_CPUS_PER_TASK" \
    --mem="$EVAL_MEM" \
    --time="$EVAL_WALLTIME" \
    --gres="gpu:${GPUS_PER_NODE}" \
    --constraint="$CONSTRAINT" \
    --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
    --chdir="$REPO_ROOT" \
    --export="$export_args" \
    --dependency="afterok:${train_job_id}" \
    scripts/run_shared4_objective_suite_h200.sh)"
  eval_job_ids+=("$eval_job_id")
  echo "[submit] eval=${eval_task} job=${eval_job_id} dependency=${train_job_id}"
  eval_index="$((eval_index + 1))"
done

aggregate_dependency="$(IFS=:; echo "${eval_job_ids[*]}")"
export_args="ALL"
append_export RUN_TAG "$RUN_TAG"
append_export OUTPUT_DIR "$OUTPUT_DIR"
append_export TASK_MODE aggregate
append_export GPUS_PER_NODE 1
append_export ACCELERATOR_LABEL A40
append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
append_export AGGREGATE_BOOTSTRAP_RUNS "$AGGREGATE_BOOTSTRAP_RUNS"
append_export AGGREGATE_SEED "$AGGREGATE_SEED"
append_export AGGREGATE_LABEL "$AGGREGATE_LABEL"

aggregate_job_id="$(submit_or_echo \
  --parsable \
  --job-name="aggregate-w3-a40" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$POST_CPUS_PER_TASK" \
  --mem="$POST_MEM" \
  --time="$POSTPROCESS_WALLTIME" \
  --gres=gpu:1 \
  --constraint="$CONSTRAINT" \
  --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
  --chdir="$REPO_ROOT" \
  --export="$export_args" \
  --dependency="afterok:${aggregate_dependency}" \
  scripts/run_shared4_objective_suite_h200.sh)"
echo "[submit] aggregate=${aggregate_job_id} dependency=${aggregate_dependency}"

summary_path="${OUTPUT_DIR}/a40_width3_phase1_submission.txt"
{
  echo "run_tag=${RUN_TAG}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "train_job_id=${train_job_id}"
  echo "eval_job_ids=${eval_job_ids[*]}"
  echo "aggregate_job_id=${aggregate_job_id}"
  echo "account=${ACCOUNT}"
  echo "partition=${PARTITION}"
  echo "constraint=${CONSTRAINT}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "train_walltime=${TRAIN_WALLTIME}"
  echo "target_walltime_plan=04:00:00"
  echo "total_steps_per_variant=${TOTAL_STEPS_PER_VARIANT}"
  echo "train_batch_size=${TRAIN_BATCH_SIZE}"
  echo "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  echo "rollout_n=${ROLLOUT_N}"
  echo "rollout_name=${ROLLOUT_NAME}"
  echo "pytorch_rollout_timeout_seconds=${PYTORCH_ROLLOUT_TIMEOUT_SECONDS}"
  echo "pytorch_rollout_debug=${PYTORCH_ROLLOUT_DEBUG}"
  echo "ray_dedup_logs=${RAY_DEDUP_LOGS}"
  echo "effective_trajectories_per_step=$((TRAIN_BATCH_SIZE * ROLLOUT_N))"
  echo "effective_trajectories_per_condition=$((TRAIN_TARGET_STEPS * TRAIN_BATCH_SIZE * ROLLOUT_N))"
  echo "multiplex_width_override=${MULTIPLEX_WIDTH_OVERRIDE}"
  echo "shared4_multiplex_width_override=${SHARED4_MULTIPLEX_WIDTH_OVERRIDE}"
  echo "max_response_length=${MAX_RESPONSE_LENGTH}"
  echo "normalize_cuda_visible_devices=${NORMALIZE_CUDA_VISIBLE_DEVICES}"
  echo "free_cache_engine=${FREE_CACHE_ENGINE}"
  echo "enable_sleep_hack=${ENABLE_SLEEP_HACK}"
  echo "enable_memory_saver=${ENABLE_MEMORY_SAVER}"
  echo "sglang_disable_overlap_schedule=${SGLANG_DISABLE_OVERLAP_SCHEDULE}"
  echo "sglang_sampling_backend=${SGLANG_SAMPLING_BACKEND}"
  echo "attn_implementation=${ATTN_IMPLEMENTATION}"
  echo "early_stopping_length_threshold=${EARLY_STOPPING_LENGTH_THRESHOLD}"
  echo "branch_rollout=4x4"
  echo "run_conditions=shared4_joint+multiplex_thinking"
  echo "eval_tasks=multiplex_untrained multiplex_trained shared4_untrained shared4_joint_trained"
  echo "aggregate_label=${AGGREGATE_LABEL}"
} > "$summary_path"
echo "[submit] summary=${summary_path}"
