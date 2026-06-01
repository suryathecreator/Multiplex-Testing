#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-h200-recovery-full-matrix-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/${RUN_TAG}}"
TOTAL_STEPS_PER_VARIANT="${TOTAL_STEPS_PER_VARIANT:-40}"
SHARD_STEPS="${SHARD_STEPS:-4}"

ACCOUNT="${ACCOUNT:-raivn-ckpt}"
PARTITION="${PARTITION:-ckpt-all}"
CONSTRAINT="${CONSTRAINT:-h200}"
JOB_ACCELERATOR_LABEL="${JOB_ACCELERATOR_LABEL:-H200}"
JOB_SUFFIX="${JOB_SUFFIX:-h200}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-256G}"
TRAIN_WALLTIME="${TRAIN_WALLTIME:-02:45:00}"
EVAL_WALLTIME="${EVAL_WALLTIME:-04:00:00}"
POSTPROCESS_WALLTIME="${POSTPROCESS_WALLTIME:-00:30:00}"
MAX_CRITICAL_PATH_HOURS="${MAX_CRITICAL_PATH_HOURS:-36}"

TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-40000}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
AUTO_OOM_RETRY="${AUTO_OOM_RETRY:-True}"
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-True}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-['model','hf_model','optimizer','extra']}"
CHECKPOINT_LOAD_CONTENTS="${CHECKPOINT_LOAD_CONTENTS:-['model','optimizer','extra']}"

EVAL_SEEDS_CSV="${EVAL_SEEDS_CSV:-26010808,26010809,26010810}"
EVAL_MAX_K="${EVAL_MAX_K:-8}"
EVAL_MAX_PROMPTS="${EVAL_MAX_PROMPTS:-30}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-5120}"
REASONING_PREFIX_TOKENS="${REASONING_PREFIX_TOKENS:-4096}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-16}"
EVAL_MAX_CONCURRENT_PROMPTS="${EVAL_MAX_CONCURRENT_PROMPTS:-2}"
AGGREGATE_BOOTSTRAP_RUNS="${AGGREGATE_BOOTSTRAP_RUNS:-1000}"

SUBMIT_SMOKE_JOB="${SUBMIT_SMOKE_JOB:-True}"
SMOKE_WALLTIME="${SMOKE_WALLTIME:-03:00:00}"
SMOKE_RUN_CONDITIONS="${SMOKE_RUN_CONDITIONS:-discrete_rl,multiplex_thinking,shared4_joint}"
SMOKE_TOTAL_STEPS="${SMOKE_TOTAL_STEPS:-1}"
SMOKE_TRAIN_TARGET_STEPS="${SMOKE_TRAIN_TARGET_STEPS:-1}"
SMOKE_TRAIN_EXAMPLES="${SMOKE_TRAIN_EXAMPLES:-128}"
SMOKE_SAVE_HF_MODEL="${SMOKE_SAVE_HF_MODEL:-False}"

DRY_RUN="${DRY_RUN:-False}"

TRAIN_CONDITIONS=(
  discrete_rl
  multiplex_thinking
  shared4_joint
  shared4_thinking_only
  shared4_answer_only
)

EVAL_TASKS=(
  discrete_untrained
  discrete_trained
  multiplex_untrained
  multiplex_trained
  discrete_trained_multiplex_eval
  multiplex_trained_discrete_eval
  shared4_untrained
  shared4_multiplex_trained
  shared4_joint_trained
  shared4_thinking_trained
  shared4_answer_trained
)

mkdir -p "$OUTPUT_DIR" slurm_logs

truthy() {
  [[ "${1:-}" =~ ^(1|true|TRUE|True|yes|YES|on|ON)$ ]]
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[config] ${name} must be a positive integer; got ${value}" >&2
    exit 1
  fi
}

walltime_to_minutes() {
  local value="$1"
  local hours minutes seconds
  IFS=: read -r hours minutes seconds <<< "$value"
  if [[ -z "${seconds:-}" ]]; then
    seconds="${minutes:-0}"
    minutes="${hours:-0}"
    hours=0
  fi
  echo "$((10#$hours * 60 + 10#$minutes + (10#$seconds + 59) / 60))"
}

condition_slug() {
  case "$1" in
    discrete_rl) echo "dr" ;;
    multiplex_thinking) echo "mt" ;;
    shared4_joint) echo "s4j" ;;
    shared4_thinking_only) echo "s4t" ;;
    shared4_answer_only) echo "s4a" ;;
    *)
      echo "[config] unknown condition: $1" >&2
      exit 1
      ;;
  esac
}

eval_slug() {
  case "$1" in
    discrete_untrained) echo "du" ;;
    discrete_trained) echo "dt" ;;
    multiplex_untrained) echo "mu" ;;
    multiplex_trained) echo "mt" ;;
    discrete_trained_multiplex_eval) echo "dtm" ;;
    multiplex_trained_discrete_eval) echo "mtd" ;;
    shared4_untrained) echo "s4u" ;;
    shared4_multiplex_trained) echo "s4mt" ;;
    shared4_joint_trained) echo "s4jt" ;;
    shared4_thinking_trained) echo "s4tt" ;;
    shared4_answer_trained) echo "s4at" ;;
    *)
      echo "[config] unknown eval task: $1" >&2
      exit 1
      ;;
  esac
}

train_knobs_for_condition() {
  case "$1" in
    discrete_rl|multiplex_thinking)
      TRAIN_BATCH_SIZE_FOR_JOB="${TRAIN_BATCH_SIZE_DISCRETE:-64}"
      PPO_MINI_BATCH_SIZE_FOR_JOB="${PPO_MINI_BATCH_SIZE_DISCRETE:-32}"
      ROLLOUT_N_FOR_JOB="${ROLLOUT_N_DISCRETE:-4}"
      MAX_RESPONSE_LENGTH_FOR_JOB="${MAX_RESPONSE_LENGTH_DISCRETE:-1024}"
      MAX_TOKEN_LEN_PER_GPU_FOR_JOB="${MAX_TOKEN_LEN_PER_GPU_DISCRETE:-16384}"
      ROLLOUT_MAX_NUM_SEQS_FOR_JOB="${ROLLOUT_MAX_NUM_SEQS_DISCRETE:-128}"
      GPU_MEM_UTIL_FOR_JOB="${GPU_MEM_UTIL_DISCRETE:-0.74}"
      SAVE_FREQ_FOR_JOB="${SAVE_FREQ_DISCRETE:-$SHARD_STEPS}"
      ;;
    shared4_joint|shared4_thinking_only|shared4_answer_only)
      TRAIN_BATCH_SIZE_FOR_JOB="${TRAIN_BATCH_SIZE_SHARED4:-32}"
      PPO_MINI_BATCH_SIZE_FOR_JOB="${PPO_MINI_BATCH_SIZE_SHARED4:-16}"
      ROLLOUT_N_FOR_JOB="${ROLLOUT_N_SHARED4:-16}"
      MAX_RESPONSE_LENGTH_FOR_JOB="${MAX_RESPONSE_LENGTH_SHARED4:-4096}"
      MAX_TOKEN_LEN_PER_GPU_FOR_JOB="${MAX_TOKEN_LEN_PER_GPU_SHARED4:-24576}"
      ROLLOUT_MAX_NUM_SEQS_FOR_JOB="${ROLLOUT_MAX_NUM_SEQS_SHARED4:-128}"
      GPU_MEM_UTIL_FOR_JOB="${GPU_MEM_UTIL_SHARED4:-0.72}"
      SAVE_FREQ_FOR_JOB="${SAVE_FREQ_SHARED4:-$SHARD_STEPS}"
      ;;
  esac
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

require_positive_int TOTAL_STEPS_PER_VARIANT "$TOTAL_STEPS_PER_VARIANT"
require_positive_int SHARD_STEPS "$SHARD_STEPS"
require_positive_int GPUS_PER_NODE "$GPUS_PER_NODE"
require_positive_int CPUS_PER_TASK "$CPUS_PER_TASK"
require_positive_int MAX_CRITICAL_PATH_HOURS "$MAX_CRITICAL_PATH_HOURS"

case "$CONSTRAINT" in
  h200|H200) ;;
  *)
    echo "[config] this recovery submitter is intentionally H200-only; got CONSTRAINT=${CONSTRAINT}" >&2
    exit 1
    ;;
esac

if ! truthy "$DRY_RUN" && ! command -v sbatch >/dev/null 2>&1; then
  echo "[submit] sbatch is not available; rerun with DRY_RUN=True to inspect planned jobs" >&2
  exit 1
fi

targets=()
for ((target = SHARD_STEPS; target < TOTAL_STEPS_PER_VARIANT; target += SHARD_STEPS)); do
  targets+=("$target")
done
if [[ "${#targets[@]}" -eq 0 ]]; then
  targets+=("$TOTAL_STEPS_PER_VARIANT")
else
  last_target_index="$((${#targets[@]} - 1))"
  if [[ "${targets[$last_target_index]}" -ne "$TOTAL_STEPS_PER_VARIANT" ]]; then
    targets+=("$TOTAL_STEPS_PER_VARIANT")
  fi
fi

train_minutes="$(walltime_to_minutes "$TRAIN_WALLTIME")"
eval_minutes="$(walltime_to_minutes "$EVAL_WALLTIME")"
postprocess_minutes="$(walltime_to_minutes "$POSTPROCESS_WALLTIME")"
smoke_minutes=0
if truthy "$SUBMIT_SMOKE_JOB"; then
  smoke_minutes="$(walltime_to_minutes "$SMOKE_WALLTIME")"
fi
critical_path_minutes="$((smoke_minutes + ${#targets[@]} * train_minutes + eval_minutes + postprocess_minutes))"
if [[ "$critical_path_minutes" -gt "$((MAX_CRITICAL_PATH_HOURS * 60))" ]]; then
  echo "[config] requested critical path is ${critical_path_minutes} minutes, over ${MAX_CRITICAL_PATH_HOURS} hours" >&2
  exit 1
fi

IFS=',' read -r -a eval_seeds <<< "$EVAL_SEEDS_CSV"

declare -A final_train_job_ids=()
submission_records=()
eval_job_ids=()
smoke_job_id=""

if truthy "$SUBMIT_SMOKE_JOB"; then
  export_args="ALL"
  append_export TASK_MODE train
  append_export RUN_TAG "${RUN_TAG}-smoke"
  append_export OUTPUT_DIR "${OUTPUT_DIR}/smoke"
  append_export RUN_CONDITIONS "$SMOKE_RUN_CONDITIONS"
  append_export TOTAL_STEPS_PER_VARIANT "$SMOKE_TOTAL_STEPS"
  append_export TRAIN_TARGET_STEPS "$SMOKE_TRAIN_TARGET_STEPS"
  append_export TRAIN_EXAMPLES "$SMOKE_TRAIN_EXAMPLES"
  append_export SHARD_STEPS "$SMOKE_TRAIN_TARGET_STEPS"
  append_export GPUS_PER_NODE "$GPUS_PER_NODE"
  append_export ACCELERATOR_LABEL "$JOB_ACCELERATOR_LABEL"
  append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
  append_export AUTO_OOM_RETRY "$AUTO_OOM_RETRY"
  append_export ENFORCE_EAGER "$ENFORCE_EAGER"
  append_export SAVE_HF_MODEL "$SMOKE_SAVE_HF_MODEL"
  append_export CHECKPOINT_SAVE_CONTENTS ""
  append_export CHECKPOINT_LOAD_CONTENTS ""

  smoke_sbatch_args=(
    --parsable
    --job-name="smoke-${JOB_SUFFIX}"
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --time="$SMOKE_WALLTIME"
    --gres="gpu:${GPUS_PER_NODE}"
    --constraint="$CONSTRAINT"
    --output="${REPO_ROOT}/slurm_logs/%x-%j.out"
    --chdir="$REPO_ROOT"
    --export="$export_args"
    scripts/run_shared4_objective_suite_h200.sh
  )
  if truthy "$DRY_RUN"; then
    print_command sbatch "${smoke_sbatch_args[@]}"
    smoke_job_id="DRYRUN-smoke-${JOB_SUFFIX}"
  else
    smoke_job_id="$(sbatch "${smoke_sbatch_args[@]}")"
  fi
  submission_records+=("smoke job_id=${smoke_job_id} conditions=${SMOKE_RUN_CONDITIONS}")
  echo "[submit] smoke job_id=${smoke_job_id} conditions=${SMOKE_RUN_CONDITIONS}"
fi

for condition in "${TRAIN_CONDITIONS[@]}"; do
  previous_job_id=""
  slug="$(condition_slug "$condition")"

  for target_steps in "${targets[@]}"; do
    printf -v padded_target "%03d" "$target_steps"
    job_name="${slug}-${padded_target}-${JOB_SUFFIX}"
    dependency="${previous_job_id:-${smoke_job_id:-none}}"

    export_args="ALL"
    append_export TASK_MODE train
    append_export RUN_TAG "$RUN_TAG"
    append_export OUTPUT_DIR "$OUTPUT_DIR"
    append_export RUN_CONDITIONS "$condition"
    append_export TOTAL_STEPS_PER_VARIANT "$TOTAL_STEPS_PER_VARIANT"
    append_export TRAIN_TARGET_STEPS "$target_steps"
    append_export TRAIN_EXAMPLES "$TRAIN_EXAMPLES"
    append_export SHARD_STEPS "$SHARD_STEPS"
    append_export GPUS_PER_NODE "$GPUS_PER_NODE"
    append_export ACCELERATOR_LABEL "$JOB_ACCELERATOR_LABEL"
    append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
    append_export AUTO_OOM_RETRY "$AUTO_OOM_RETRY"
    append_export ENFORCE_EAGER "$ENFORCE_EAGER"
    append_export SAVE_HF_MODEL "$SAVE_HF_MODEL"
    append_export CHECKPOINT_SAVE_CONTENTS ""
    append_export CHECKPOINT_LOAD_CONTENTS ""
    train_knobs_for_condition "$condition"
    append_export TRAIN_BATCH_SIZE "$TRAIN_BATCH_SIZE_FOR_JOB"
    append_export PPO_MINI_BATCH_SIZE "$PPO_MINI_BATCH_SIZE_FOR_JOB"
    append_export ROLLOUT_N "$ROLLOUT_N_FOR_JOB"
    append_export MAX_RESPONSE_LENGTH "$MAX_RESPONSE_LENGTH_FOR_JOB"
    append_export MAX_TOKEN_LEN_PER_GPU "$MAX_TOKEN_LEN_PER_GPU_FOR_JOB"
    append_export ROLLOUT_MAX_NUM_SEQS "$ROLLOUT_MAX_NUM_SEQS_FOR_JOB"
    append_export GPU_MEM_UTIL "$GPU_MEM_UTIL_FOR_JOB"
    append_export SAVE_FREQ "$SAVE_FREQ_FOR_JOB"

    sbatch_args=(
      --parsable
      --job-name="$job_name"
      --account="$ACCOUNT"
      --partition="$PARTITION"
      --nodes=1
      --ntasks=1
      --cpus-per-task="$CPUS_PER_TASK"
      --mem="$MEM"
      --time="$TRAIN_WALLTIME"
      --gres="gpu:${GPUS_PER_NODE}"
      --constraint="$CONSTRAINT"
      --output="${REPO_ROOT}/slurm_logs/%x-%j.out"
      --chdir="$REPO_ROOT"
      --export="$export_args"
    )
    if [[ -n "$previous_job_id" ]]; then
      sbatch_args+=(--dependency="afterany:${previous_job_id}")
    elif [[ -n "$smoke_job_id" ]]; then
      sbatch_args+=(--dependency="afterok:${smoke_job_id}")
    fi
    sbatch_args+=(scripts/run_shared4_objective_suite_h200.sh)

    if truthy "$DRY_RUN"; then
      print_command sbatch "${sbatch_args[@]}"
      job_id="DRYRUN-${job_name}"
    else
      job_id="$(sbatch "${sbatch_args[@]}")"
    fi
    submission_records+=("train condition=${condition} target_steps=${target_steps} job_id=${job_id} dependency=${dependency}")
    previous_job_id="$job_id"
    echo "[submit] train condition=${condition} target_steps=${target_steps} job_id=${job_id} dependency=${dependency}"
  done

  final_train_job_ids["$condition"]="$previous_job_id"
done

dependency_for_eval_task() {
  case "$1" in
    discrete_trained|discrete_trained_multiplex_eval) echo "${final_train_job_ids[discrete_rl]}" ;;
    multiplex_trained|multiplex_trained_discrete_eval|shared4_multiplex_trained) echo "${final_train_job_ids[multiplex_thinking]}" ;;
    shared4_joint_trained) echo "${final_train_job_ids[shared4_joint]}" ;;
    shared4_thinking_trained) echo "${final_train_job_ids[shared4_thinking_only]}" ;;
    shared4_answer_trained) echo "${final_train_job_ids[shared4_answer_only]}" ;;
    *) echo "$smoke_job_id" ;;
  esac
}

seed_index=0
for seed in "${eval_seeds[@]}"; do
  repeat_label="$(printf 'repeat_%02d' "$seed_index")"
  task_index=0
  for eval_task in "${EVAL_TASKS[@]}"; do
    slug="$(eval_slug "$eval_task")"
    printf -v padded_task "%02d" "$task_index"
    job_name="eval-${slug}-r${seed_index}-${JOB_SUFFIX}"
    port="$((37100 + seed_index * 100 + task_index))"
    dependency="$(dependency_for_eval_task "$eval_task")"

    export_args="ALL"
    append_export TASK_MODE eval
    append_export RUN_TAG "$RUN_TAG"
    append_export OUTPUT_DIR "$OUTPUT_DIR"
    append_export GPUS_PER_NODE "$GPUS_PER_NODE"
    append_export ACCELERATOR_LABEL "$JOB_ACCELERATOR_LABEL"
    append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
    append_export EVAL_TASK "$eval_task"
    append_export EVAL_SEED "$seed"
    append_export EVAL_REPEAT_LABEL "$repeat_label"
    append_export EVAL_PORT "$port"
    append_export EVAL_MAX_K "$EVAL_MAX_K"
    append_export EVAL_MAX_PROMPTS "$EVAL_MAX_PROMPTS"
    append_export EVAL_MAX_NEW_TOKENS "$EVAL_MAX_NEW_TOKENS"
    append_export REASONING_PREFIX_TOKENS "$REASONING_PREFIX_TOKENS"
    append_export EVAL_REQUEST_BATCH_SIZE "$EVAL_REQUEST_BATCH_SIZE"
    append_export EVAL_MAX_CONCURRENT_PROMPTS "$EVAL_MAX_CONCURRENT_PROMPTS"

    sbatch_args=(
      --parsable
      --job-name="$job_name"
      --account="$ACCOUNT"
      --partition="$PARTITION"
      --nodes=1
      --ntasks=1
      --cpus-per-task="$CPUS_PER_TASK"
      --mem="$MEM"
      --time="$EVAL_WALLTIME"
      --gres="gpu:${GPUS_PER_NODE}"
      --constraint="$CONSTRAINT"
      --output="${REPO_ROOT}/slurm_logs/%x-%j.out"
      --chdir="$REPO_ROOT"
      --export="$export_args"
    )
    if [[ -n "$dependency" ]]; then
      sbatch_args+=(--dependency="afterok:${dependency}")
    fi
    sbatch_args+=(scripts/run_shared4_objective_suite_h200.sh)

    if truthy "$DRY_RUN"; then
      print_command sbatch "${sbatch_args[@]}"
      job_id="DRYRUN-${job_name}"
    else
      job_id="$(sbatch "${sbatch_args[@]}")"
    fi
    submission_records+=("eval task=${eval_task} seed=${seed} job_id=${job_id} dependency=${dependency:-none}")
    eval_job_ids+=("$job_id")
    echo "[submit] eval task=${eval_task} seed=${seed} job_id=${job_id} dependency=${dependency:-none}"
    task_index="$((task_index + 1))"
  done
  seed_index="$((seed_index + 1))"
done

aggregate_job_id=""
if [[ "${#eval_job_ids[@]}" -gt 0 ]]; then
  aggregate_dependency="$(IFS=:; echo "${eval_job_ids[*]}")"
  export_args="ALL"
  append_export TASK_MODE aggregate
  append_export RUN_TAG "$RUN_TAG"
  append_export OUTPUT_DIR "$OUTPUT_DIR"
  append_export GPUS_PER_NODE "$GPUS_PER_NODE"
  append_export ACCELERATOR_LABEL "$JOB_ACCELERATOR_LABEL"
  append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
  append_export AGGREGATE_BOOTSTRAP_RUNS "$AGGREGATE_BOOTSTRAP_RUNS"
  append_export AGGREGATE_SEED "${AGGREGATE_SEED:-26010808}"

  aggregate_sbatch_args=(
    --parsable
    --job-name="aggregate-${JOB_SUFFIX}"
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --time="$POSTPROCESS_WALLTIME"
    --gres="gpu:${GPUS_PER_NODE}"
    --constraint="$CONSTRAINT"
    --output="${REPO_ROOT}/slurm_logs/%x-%j.out"
    --chdir="$REPO_ROOT"
    --export="$export_args"
    --dependency="afterok:${aggregate_dependency}"
    scripts/run_shared4_objective_suite_h200.sh
  )
  if truthy "$DRY_RUN"; then
    print_command sbatch "${aggregate_sbatch_args[@]}"
    aggregate_job_id="DRYRUN-aggregate-${JOB_SUFFIX}"
  else
    aggregate_job_id="$(sbatch "${aggregate_sbatch_args[@]}")"
  fi
  submission_records+=("aggregate job_id=${aggregate_job_id} dependency=${aggregate_dependency}")
  echo "[submit] aggregate job_id=${aggregate_job_id} dependency=${aggregate_dependency}"
fi

SUMMARY_PATH="${OUTPUT_DIR}/h200_full_matrix_submission.txt"
{
  echo "run_tag=${RUN_TAG}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "total_steps_per_variant=${TOTAL_STEPS_PER_VARIANT}"
  echo "shard_steps=${SHARD_STEPS}"
  echo "target_steps=${targets[*]}"
  echo "constraint=${CONSTRAINT}"
  echo "job_accelerator_label=${JOB_ACCELERATOR_LABEL}"
  echo "gpus_per_job=${GPUS_PER_NODE}"
  echo "train_walltime=${TRAIN_WALLTIME}"
  echo "eval_walltime=${EVAL_WALLTIME}"
  echo "postprocess_walltime=${POSTPROCESS_WALLTIME}"
  echo "smoke_walltime=${SMOKE_WALLTIME}"
  echo "mem=${MEM}"
  echo "critical_path_minutes=${critical_path_minutes}"
  echo "max_critical_path_hours=${MAX_CRITICAL_PATH_HOURS}"
  echo "train_conditions=${TRAIN_CONDITIONS[*]}"
  echo "eval_tasks=${EVAL_TASKS[*]}"
  echo "eval_seeds=${eval_seeds[*]}"
  echo "eval_max_k=${EVAL_MAX_K}"
  echo "eval_max_prompts=${EVAL_MAX_PROMPTS}"
  echo "eval_max_new_tokens=${EVAL_MAX_NEW_TOKENS}"
  echo "aggregate_bootstrap_runs=${AGGREGATE_BOOTSTRAP_RUNS}"
  echo "reasoning_prefix_tokens=${REASONING_PREFIX_TOKENS}"
  echo "enforce_eager=${ENFORCE_EAGER}"
  echo "checkpoint_save_contents=${CHECKPOINT_SAVE_CONTENTS}"
  echo "checkpoint_load_contents=${CHECKPOINT_LOAD_CONTENTS}"
  echo
  echo "runtime_train_profiles:"
  echo "- H200 discrete/multiplex: train_batch_size=64 ppo_mini_batch_size=32 rollout_n=4 max_response_length=1024 max_token_len_per_gpu=16384 rollout_max_num_seqs=128 gpu_mem_util=0.74 enforce_eager=True"
  echo "- H200 shared4: train_batch_size=32 ppo_mini_batch_size=16 rollout_n=16 max_response_length=4096 max_token_len_per_gpu=24576 rollout_max_num_seqs=128 gpu_mem_util=0.72 enforce_eager=True"
  echo "- Shared4 OOM retry: train_batch_size=16 ppo_mini_batch_size=8, lower token/sequence caps, gpu_mem_util=0.70"
  echo
  echo "smoke_job_id=${smoke_job_id:-none}"
  echo "aggregate_job_id=${aggregate_job_id:-none}"
  echo
  echo "final_train_job_ids:"
  for condition in "${TRAIN_CONDITIONS[@]}"; do
    echo "- ${condition}=${final_train_job_ids[$condition]}"
  done
  echo
  echo "submitted_jobs:"
  for record in "${submission_records[@]}"; do
    echo "- ${record}"
  done
} > "$SUMMARY_PATH"

echo "[submit] summary=${SUMMARY_PATH}"
