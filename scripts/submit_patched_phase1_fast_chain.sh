#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-patched-phase1-fast-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/${RUN_TAG}}"

ACCOUNT="${ACCOUNT:-raivn-ckpt}"
PARTITION="${PARTITION:-ckpt-all}"
CONSTRAINT="${CONSTRAINT:-a100}"
JOB_ACCELERATOR_LABEL="${JOB_ACCELERATOR_LABEL:-A100}"
JOB_SUFFIX="${JOB_SUFFIX:-a100}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"

PHASE1_TOTAL_STEPS="${PHASE1_TOTAL_STEPS:-30}"
PHASE1_SHARD_STEPS="${PHASE1_SHARD_STEPS:-30}"
NORMAL_TOTAL_STEPS="${NORMAL_TOTAL_STEPS:-40}"
NORMAL_SHARD_STEPS="${NORMAL_SHARD_STEPS:-20}"

PHASE1_TRAIN_WALLTIME="${PHASE1_TRAIN_WALLTIME:-10:00:00}"
NORMAL_TRAIN_WALLTIME="${NORMAL_TRAIN_WALLTIME:-12:00:00}"
EVAL_WALLTIME="${EVAL_WALLTIME:-04:00:00}"
POSTPROCESS_WALLTIME="${POSTPROCESS_WALLTIME:-00:20:00}"

TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-40000}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
AUTO_OOM_RETRY="${AUTO_OOM_RETRY:-True}"
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-True}"
NORMALIZE_CUDA_VISIBLE_DEVICES="${NORMALIZE_CUDA_VISIBLE_DEVICES:-True}"
SGLANG_GRAMMAR_BACKEND="${SGLANG_GRAMMAR_BACKEND:-none}"
SGLANG_WATCHDOG_TIMEOUT="${SGLANG_WATCHDOG_TIMEOUT:-1800}"
PRESTART_RAY="${PRESTART_RAY:-False}"
ENABLE_RAY_STOP="${ENABLE_RAY_STOP:-False}"
PHASE1_RETRY_COUNT="${PHASE1_RETRY_COUNT:-1}"
A100_PHASE1_CONSERVATIVE="${A100_PHASE1_CONSERVATIVE:-False}"

EVAL_SEED="${EVAL_SEED:-26010808}"
EVAL_REPEAT_LABEL="${EVAL_REPEAT_LABEL:-repeat_00}"
EVAL_MAX_K="${EVAL_MAX_K:-8}"
EVAL_MAX_PROMPTS="${EVAL_MAX_PROMPTS:-30}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-5120}"
REASONING_PREFIX_TOKENS="${REASONING_PREFIX_TOKENS:-4096}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-16}"
EVAL_MAX_CONCURRENT_PROMPTS="${EVAL_MAX_CONCURRENT_PROMPTS:-2}"
EVAL_DISABLE_CUDA_GRAPH="${EVAL_DISABLE_CUDA_GRAPH:-True}"
EVAL_MEM_FRACTION_STATIC="${EVAL_MEM_FRACTION_STATIC:-0.78}"
AGGREGATE_BOOTSTRAP_RUNS="${AGGREGATE_BOOTSTRAP_RUNS:-1000}"

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

print_command() {
  {
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  } >&2
}

condition_slug() {
  case "$1" in
    discrete_rl) echo "dr" ;;
    multiplex_thinking) echo "mt" ;;
    shared4_joint) echo "s4j" ;;
    shared4_thinking_only) echo "s4t" ;;
    shared4_answer_only) echo "s4a" ;;
    *) echo "[config] unknown condition: $1" >&2; exit 1 ;;
  esac
}

eval_slug() {
  case "$1" in
    discrete_untrained) echo "du" ;;
    discrete_trained) echo "dt" ;;
    multiplex_untrained) echo "mu" ;;
    multiplex_trained) echo "mt" ;;
    shared4_untrained) echo "s4u" ;;
    shared4_joint_trained) echo "s4jt" ;;
    shared4_thinking_trained) echo "s4tt" ;;
    shared4_answer_trained) echo "s4at" ;;
    *) echo "[config] unknown eval task: $1" >&2; exit 1 ;;
  esac
}

submit_or_echo() {
  local job_id
  if truthy "$DRY_RUN"; then
    print_command sbatch "$@"
    job_id="DRYRUN"
  else
    job_id="$(sbatch "$@")"
  fi
  echo "$job_id"
}

base_train_exports() {
  local phase_total="$1"
  local target_steps="$2"
  local shard_steps="$3"
  local condition="$4"

  export_args="ALL"
  append_export TASK_MODE train
  append_export RUN_TAG "$RUN_TAG"
  append_export OUTPUT_DIR "$OUTPUT_DIR"
  append_export RUN_CONDITIONS "$condition"
  append_export TOTAL_STEPS_PER_VARIANT "$phase_total"
  append_export TRAIN_TARGET_STEPS "$target_steps"
  append_export TRAIN_EXAMPLES "$TRAIN_EXAMPLES"
  append_export SHARD_STEPS "$shard_steps"
  append_export GPUS_PER_NODE "$GPUS_PER_NODE"
  append_export ACCELERATOR_LABEL "$JOB_ACCELERATOR_LABEL"
  append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
  append_export AUTO_OOM_RETRY "$AUTO_OOM_RETRY"
  append_export ENFORCE_EAGER "$ENFORCE_EAGER"
  append_export SAVE_HF_MODEL "$SAVE_HF_MODEL"
  append_export NORMALIZE_CUDA_VISIBLE_DEVICES "$NORMALIZE_CUDA_VISIBLE_DEVICES"
  append_export SGLANG_GRAMMAR_BACKEND "$SGLANG_GRAMMAR_BACKEND"
  append_export SGLANG_WATCHDOG_TIMEOUT "$SGLANG_WATCHDOG_TIMEOUT"
  append_export PRESTART_RAY "$PRESTART_RAY"
  append_export ENABLE_RAY_STOP "$ENABLE_RAY_STOP"
}

append_a100_phase1_conservative_overrides() {
  local phase="$1"
  local condition="$2"

  if ! truthy "$A100_PHASE1_CONSERVATIVE"; then
    return 0
  fi
  if [[ "${JOB_ACCELERATOR_LABEL,,}" != *a100* ]]; then
    return 0
  fi
  case "$phase" in
    phase1|phase1r*) ;;
    *) return 0 ;;
  esac

  case "$condition" in
    multiplex_thinking|discrete_rl)
      append_export TRAIN_BATCH_SIZE "${A100_PHASE1_MT_TRAIN_BATCH_SIZE:-16}"
      append_export PPO_MINI_BATCH_SIZE "${A100_PHASE1_MT_PPO_MINI_BATCH_SIZE:-8}"
      append_export ROLLOUT_N "${A100_PHASE1_MT_ROLLOUT_N:-4}"
      append_export MAX_RESPONSE_LENGTH "${A100_PHASE1_MT_MAX_RESPONSE_LENGTH:-1024}"
      append_export MAX_TOKEN_LEN_PER_GPU "${A100_PHASE1_MT_MAX_TOKEN_LEN_PER_GPU:-6144}"
      append_export ROLLOUT_MAX_NUM_SEQS "${A100_PHASE1_MT_ROLLOUT_MAX_NUM_SEQS:-64}"
      append_export GPU_MEM_UTIL "${A100_PHASE1_MT_GPU_MEM_UTIL:-0.60}"
      ;;
    shared4_joint|shared4_thinking_only|shared4_answer_only)
      append_export TRAIN_BATCH_SIZE "${A100_PHASE1_S4_TRAIN_BATCH_SIZE:-4}"
      append_export PPO_MINI_BATCH_SIZE "${A100_PHASE1_S4_PPO_MINI_BATCH_SIZE:-2}"
      append_export ROLLOUT_N "${A100_PHASE1_S4_ROLLOUT_N:-16}"
      append_export MAX_RESPONSE_LENGTH "${A100_PHASE1_S4_MAX_RESPONSE_LENGTH:-4096}"
      append_export MAX_TOKEN_LEN_PER_GPU "${A100_PHASE1_S4_MAX_TOKEN_LEN_PER_GPU:-8192}"
      append_export ROLLOUT_MAX_NUM_SEQS "${A100_PHASE1_S4_ROLLOUT_MAX_NUM_SEQS:-64}"
      append_export GPU_MEM_UTIL "${A100_PHASE1_S4_GPU_MEM_UTIL:-0.60}"
      ;;
  esac
}

submit_train_job() {
  local phase="$1"
  local condition="$2"
  local target_steps="$3"
  local total_steps="$4"
  local shard_steps="$5"
  local walltime="$6"
  local dependency="${7:-}"
  local slug padded_target job_name job_id

  slug="$(condition_slug "$condition")"
  printf -v padded_target "%03d" "$target_steps"
  job_name="${phase}-${slug}-${padded_target}-${JOB_SUFFIX}"
  base_train_exports "$total_steps" "$target_steps" "$shard_steps" "$condition"
  append_a100_phase1_conservative_overrides "$phase" "$condition"

  sbatch_args=(
    --parsable
    --job-name="$job_name"
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --time="$walltime"
    --gres="gpu:${GPUS_PER_NODE}"
    --constraint="$CONSTRAINT"
    --output="${REPO_ROOT}/slurm_logs/%x-%j.out"
    --chdir="$REPO_ROOT"
    --export="$export_args"
  )
  if [[ -n "$dependency" ]]; then
    sbatch_args+=(--dependency="$dependency")
  fi
  sbatch_args+=(scripts/run_shared4_objective_suite_h200.sh)

  job_id="$(submit_or_echo "${sbatch_args[@]}")"
  echo "[submit] train phase=${phase} condition=${condition} target=${target_steps} job_id=${job_id} dependency=${dependency:-none}" >&2
  echo "$job_id"
}

submit_eval_job() {
  local phase="$1"
  local eval_task="$2"
  local dependency="$3"
  local port="$4"
  local slug job_name job_id

  slug="$(eval_slug "$eval_task")"
  job_name="eval-${phase}-${slug}-r0-${JOB_SUFFIX}"
  export_args="ALL"
  append_export TASK_MODE eval
  append_export RUN_TAG "$RUN_TAG"
  append_export OUTPUT_DIR "$OUTPUT_DIR"
  append_export GPUS_PER_NODE "$GPUS_PER_NODE"
  append_export ACCELERATOR_LABEL "$JOB_ACCELERATOR_LABEL"
  append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
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

  job_id="$(submit_or_echo \
    --parsable \
    --job-name="$job_name" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$EVAL_WALLTIME" \
    --gres="gpu:${GPUS_PER_NODE}" \
    --constraint="$CONSTRAINT" \
    --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
    --chdir="$REPO_ROOT" \
    --export="$export_args" \
    --dependency="afterok:${dependency}" \
    scripts/run_shared4_objective_suite_h200.sh)"
  echo "[submit] eval phase=${phase} task=${eval_task} job_id=${job_id} dependency=${dependency}" >&2
  echo "$job_id"
}

submit_aggregate_job() {
  local phase="$1"
  local dependency="$2"
  local job_name job_id

  job_name="aggregate-${phase}-${JOB_SUFFIX}"
  export_args="ALL"
  append_export TASK_MODE aggregate
  append_export RUN_TAG "$RUN_TAG"
  append_export OUTPUT_DIR "$OUTPUT_DIR"
  append_export GPUS_PER_NODE "$GPUS_PER_NODE"
  append_export ACCELERATOR_LABEL "$JOB_ACCELERATOR_LABEL"
  append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
  append_export AGGREGATE_BOOTSTRAP_RUNS "$AGGREGATE_BOOTSTRAP_RUNS"
  append_export AGGREGATE_SEED "$EVAL_SEED"
  append_export AGGREGATE_LABEL "$phase"

  job_id="$(submit_or_echo \
    --parsable \
    --job-name="$job_name" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$POSTPROCESS_WALLTIME" \
    --gres="gpu:${GPUS_PER_NODE}" \
    --constraint="$CONSTRAINT" \
    --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
    --chdir="$REPO_ROOT" \
    --export="$export_args" \
    --dependency="afterok:${dependency}" \
    scripts/run_shared4_objective_suite_h200.sh)"
  echo "[submit] aggregate phase=${phase} job_id=${job_id} dependency=${dependency}" >&2
  echo "$job_id"
}

phase1_mt_root="$(submit_train_job phase1 multiplex_thinking "$PHASE1_TOTAL_STEPS" "$PHASE1_TOTAL_STEPS" "$PHASE1_SHARD_STEPS" "$PHASE1_TRAIN_WALLTIME")"
phase1_s4j_root="$(submit_train_job phase1 shared4_joint "$PHASE1_TOTAL_STEPS" "$PHASE1_TOTAL_STEPS" "$PHASE1_SHARD_STEPS" "$PHASE1_TRAIN_WALLTIME")"
phase1_mt="$phase1_mt_root"
phase1_s4j="$phase1_s4j_root"
phase1_mt_retries=()
phase1_s4j_retries=()
for ((retry_idx = 1; retry_idx <= PHASE1_RETRY_COUNT; retry_idx++)); do
  phase1_mt="$(submit_train_job "phase1r${retry_idx}" multiplex_thinking "$PHASE1_TOTAL_STEPS" "$PHASE1_TOTAL_STEPS" "$PHASE1_SHARD_STEPS" "$PHASE1_TRAIN_WALLTIME" "afterany:${phase1_mt}")"
  phase1_mt_retries+=("$phase1_mt")
  phase1_s4j="$(submit_train_job "phase1r${retry_idx}" shared4_joint "$PHASE1_TOTAL_STEPS" "$PHASE1_TOTAL_STEPS" "$PHASE1_SHARD_STEPS" "$PHASE1_TRAIN_WALLTIME" "afterany:${phase1_s4j}")"
  phase1_s4j_retries+=("$phase1_s4j")
done
phase1_both="${phase1_mt}:${phase1_s4j}"

phase1_eval_mu="$(submit_eval_job phase1 multiplex_untrained "$phase1_both" 37400)"
phase1_eval_mt="$(submit_eval_job phase1 multiplex_trained "$phase1_mt" 37401)"
phase1_eval_s4u="$(submit_eval_job phase1 shared4_untrained "$phase1_both" 37402)"
phase1_eval_s4jt="$(submit_eval_job phase1 shared4_joint_trained "$phase1_s4j" 37403)"
phase1_aggregate="$(submit_aggregate_job phase1 "${phase1_eval_mu}:${phase1_eval_mt}:${phase1_eval_s4u}:${phase1_eval_s4jt}")"

phase2_s4a_020="$(submit_train_job phase2 shared4_answer_only 20 "$NORMAL_TOTAL_STEPS" "$NORMAL_SHARD_STEPS" "$NORMAL_TRAIN_WALLTIME" "afterok:${phase1_aggregate}")"
phase2_s4a_040="$(submit_train_job phase2 shared4_answer_only "$NORMAL_TOTAL_STEPS" "$NORMAL_TOTAL_STEPS" "$NORMAL_SHARD_STEPS" "$NORMAL_TRAIN_WALLTIME" "afterany:${phase2_s4a_020}")"
phase2_s4t_020="$(submit_train_job phase2 shared4_thinking_only 20 "$NORMAL_TOTAL_STEPS" "$NORMAL_SHARD_STEPS" "$NORMAL_TRAIN_WALLTIME" "afterok:${phase1_aggregate}")"
phase2_s4t_040="$(submit_train_job phase2 shared4_thinking_only "$NORMAL_TOTAL_STEPS" "$NORMAL_TOTAL_STEPS" "$NORMAL_SHARD_STEPS" "$NORMAL_TRAIN_WALLTIME" "afterany:${phase2_s4t_020}")"

phase2_eval_s4at="$(submit_eval_job phase2 shared4_answer_trained "$phase2_s4a_040" 37500)"
phase2_eval_s4tt="$(submit_eval_job phase2 shared4_thinking_trained "$phase2_s4t_040" 37501)"
phase2_aggregate="$(submit_aggregate_job phase2 "${phase1_aggregate}:${phase2_eval_s4at}:${phase2_eval_s4tt}")"

phase3_dr_020="$(submit_train_job phase3 discrete_rl 20 "$NORMAL_TOTAL_STEPS" "$NORMAL_SHARD_STEPS" "$NORMAL_TRAIN_WALLTIME" "afterok:${phase2_aggregate}")"
phase3_dr_040="$(submit_train_job phase3 discrete_rl "$NORMAL_TOTAL_STEPS" "$NORMAL_TOTAL_STEPS" "$NORMAL_SHARD_STEPS" "$NORMAL_TRAIN_WALLTIME" "afterany:${phase3_dr_020}")"

phase3_eval_du="$(submit_eval_job phase3 discrete_untrained "$phase2_aggregate" 37600)"
phase3_eval_dt="$(submit_eval_job phase3 discrete_trained "$phase3_dr_040" 37601)"
phase3_aggregate="$(submit_aggregate_job phase3 "${phase2_aggregate}:${phase3_eval_du}:${phase3_eval_dt}")"

summary_path="${OUTPUT_DIR}/patched_phase1_fast_submission.txt"
{
  echo "run_tag=${RUN_TAG}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "constraint=${CONSTRAINT}"
  echo "job_accelerator_label=${JOB_ACCELERATOR_LABEL}"
  echo "gpus_per_job=${GPUS_PER_NODE}"
  echo "phase1_total_steps=${PHASE1_TOTAL_STEPS}"
  echo "phase1_train_walltime=${PHASE1_TRAIN_WALLTIME}"
  echo "normal_total_steps=${NORMAL_TOTAL_STEPS}"
  echo "normal_train_walltime=${NORMAL_TRAIN_WALLTIME}"
  echo "phase1_retry_count=${PHASE1_RETRY_COUNT}"
  echo "a100_phase1_conservative=${A100_PHASE1_CONSERVATIVE}"
  echo "runtime_fixes=normalize_cuda_visible_devices:${NORMALIZE_CUDA_VISIBLE_DEVICES},grammar_backend:${SGLANG_GRAMMAR_BACKEND},watchdog_timeout:${SGLANG_WATCHDOG_TIMEOUT},prestart_ray:${PRESTART_RAY},enable_ray_stop:${ENABLE_RAY_STOP}"
  echo
  echo "phase1_train_roots=${phase1_mt_root} ${phase1_s4j_root}"
  echo "phase1_train_retries=${phase1_mt_retries[*]:-none} ${phase1_s4j_retries[*]:-none}"
  echo "phase1_train_final=${phase1_mt} ${phase1_s4j}"
  echo "phase1_eval=${phase1_eval_mu} ${phase1_eval_mt} ${phase1_eval_s4u} ${phase1_eval_s4jt}"
  echo "phase1_aggregate=${phase1_aggregate}"
  echo "phase2_train=${phase2_s4a_020} ${phase2_s4a_040} ${phase2_s4t_020} ${phase2_s4t_040}"
  echo "phase2_eval=${phase2_eval_s4at} ${phase2_eval_s4tt}"
  echo "phase2_aggregate=${phase2_aggregate}"
  echo "phase3_train=${phase3_dr_020} ${phase3_dr_040}"
  echo "phase3_eval=${phase3_eval_du} ${phase3_eval_dt}"
  echo "phase3_aggregate=${phase3_aggregate}"
} > "$summary_path"

echo "[submit] summary=${summary_path}"
