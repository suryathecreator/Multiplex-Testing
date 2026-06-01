#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-h200-priority-staged-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/${RUN_TAG}}"
TOTAL_STEPS_PER_VARIANT="${TOTAL_STEPS_PER_VARIANT:-40}"
SHARD_STEPS="${SHARD_STEPS:-20}"

ACCOUNT="${ACCOUNT:-raivn-ckpt}"
PARTITION="${PARTITION:-ckpt-all}"
CONSTRAINT="${CONSTRAINT:-h200}"
JOB_ACCELERATOR_LABEL="${JOB_ACCELERATOR_LABEL:-H200}"
JOB_SUFFIX="${JOB_SUFFIX:-h200}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GRES_SPEC="${GRES_SPEC:-gpu:${GPUS_PER_NODE}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-192G}"
TRAIN_WALLTIME="${TRAIN_WALLTIME:-08:00:00}"
EVAL_WALLTIME="${EVAL_WALLTIME:-03:30:00}"
POSTPROCESS_WALLTIME="${POSTPROCESS_WALLTIME:-00:20:00}"
MAX_FIRST_RESULTS_HOURS="${MAX_FIRST_RESULTS_HOURS:-24}"

TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-40000}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
AUTO_OOM_RETRY="${AUTO_OOM_RETRY:-True}"
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-True}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-}"
CHECKPOINT_LOAD_CONTENTS="${CHECKPOINT_LOAD_CONTENTS:-}"

EVAL_SEEDS_CSV="${EVAL_SEEDS_CSV:-26010808}"
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

if [[ "$CHECKPOINT_SAVE_CONTENTS" == *,* || "$CHECKPOINT_LOAD_CONTENTS" == *,* ]]; then
  echo "[config] checkpoint content lists contain commas; leaving them to train.sh defaults to avoid sbatch --export truncation" >&2
  CHECKPOINT_SAVE_CONTENTS=""
  CHECKPOINT_LOAD_CONTENTS=""
fi

PHASE1_TRAIN_CONDITIONS=(multiplex_thinking shared4_joint)
PHASE1_EVAL_TASKS=(multiplex_untrained multiplex_trained shared4_untrained shared4_joint_trained)
PHASE2_TRAIN_CONDITIONS=(shared4_answer_only shared4_thinking_only)
PHASE2_EVAL_TASKS=(shared4_answer_trained shared4_thinking_trained)
PHASE3_TRAIN_CONDITIONS=(discrete_rl)
PHASE3_EVAL_TASKS=(discrete_untrained discrete_trained)

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
    shared4_untrained) echo "s4u" ;;
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
      TRAIN_BATCH_SIZE_FOR_JOB="${TRAIN_BATCH_SIZE_DISCRETE:-32}"
      PPO_MINI_BATCH_SIZE_FOR_JOB="${PPO_MINI_BATCH_SIZE_DISCRETE:-16}"
      ROLLOUT_N_FOR_JOB="${ROLLOUT_N_DISCRETE:-4}"
      MAX_RESPONSE_LENGTH_FOR_JOB="${MAX_RESPONSE_LENGTH_DISCRETE:-1024}"
      MAX_TOKEN_LEN_PER_GPU_FOR_JOB="${MAX_TOKEN_LEN_PER_GPU_DISCRETE:-12288}"
      ROLLOUT_MAX_NUM_SEQS_FOR_JOB="${ROLLOUT_MAX_NUM_SEQS_DISCRETE:-128}"
      GPU_MEM_UTIL_FOR_JOB="${GPU_MEM_UTIL_DISCRETE:-0.68}"
      SAVE_FREQ_FOR_JOB="${SAVE_FREQ_DISCRETE:-4}"
      ;;
    shared4_joint|shared4_thinking_only|shared4_answer_only)
      TRAIN_BATCH_SIZE_FOR_JOB="${TRAIN_BATCH_SIZE_SHARED4:-32}"
      PPO_MINI_BATCH_SIZE_FOR_JOB="${PPO_MINI_BATCH_SIZE_SHARED4:-16}"
      ROLLOUT_N_FOR_JOB="${ROLLOUT_N_SHARED4:-16}"
      MAX_RESPONSE_LENGTH_FOR_JOB="${MAX_RESPONSE_LENGTH_SHARED4:-4096}"
      MAX_TOKEN_LEN_PER_GPU_FOR_JOB="${MAX_TOKEN_LEN_PER_GPU_SHARED4:-20480}"
      ROLLOUT_MAX_NUM_SEQS_FOR_JOB="${ROLLOUT_MAX_NUM_SEQS_SHARED4:-96}"
      GPU_MEM_UTIL_FOR_JOB="${GPU_MEM_UTIL_SHARED4:-0.68}"
      SAVE_FREQ_FOR_JOB="${SAVE_FREQ_SHARED4:-4}"
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
require_positive_int MAX_FIRST_RESULTS_HOURS "$MAX_FIRST_RESULTS_HOURS"

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
first_results_minutes="$((${#targets[@]} * train_minutes + eval_minutes + postprocess_minutes))"
if [[ "$first_results_minutes" -gt "$((MAX_FIRST_RESULTS_HOURS * 60))" ]]; then
  echo "[config] first-result critical path is ${first_results_minutes} minutes, over ${MAX_FIRST_RESULTS_HOURS} hours" >&2
  exit 1
fi

IFS=',' read -r -a eval_seeds <<< "$EVAL_SEEDS_CSV"

declare -A final_train_job_ids=()
submission_records=()
phase1_eval_job_ids=()
phase2_eval_job_ids=()
phase3_eval_job_ids=()
phase1_aggregate_job_id=""
phase2_aggregate_job_id=""
phase3_aggregate_job_id=""
phase1_train_dependency=""
LAST_AGGREGATE_JOB_ID=""

submit_train_chain() {
  local phase="$1"
  local condition="$2"
  local start_dependency="${3:-}"
  local previous_job_id=""
  local slug
  slug="$(condition_slug "$condition")"

  for target_steps in "${targets[@]}"; do
    local padded_target job_name dependency job_id
    printf -v padded_target "%03d" "$target_steps"
    job_name="${phase}-${slug}-${padded_target}-${JOB_SUFFIX}"
    dependency="${previous_job_id:-${start_dependency:-none}}"

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
    if [[ -n "$CHECKPOINT_SAVE_CONTENTS" ]]; then
      append_export CHECKPOINT_SAVE_CONTENTS "$CHECKPOINT_SAVE_CONTENTS"
    fi
    if [[ -n "$CHECKPOINT_LOAD_CONTENTS" ]]; then
      append_export CHECKPOINT_LOAD_CONTENTS "$CHECKPOINT_LOAD_CONTENTS"
    fi
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
      --gres="$GRES_SPEC"
      --constraint="$CONSTRAINT"
      --output="${REPO_ROOT}/slurm_logs/%x-%j.out"
      --chdir="$REPO_ROOT"
      --export="$export_args"
    )
    if [[ -n "$previous_job_id" ]]; then
      sbatch_args+=(--dependency="afterany:${previous_job_id}")
    elif [[ -n "$start_dependency" ]]; then
      sbatch_args+=(--dependency="afterok:${start_dependency}")
    fi
    sbatch_args+=(scripts/run_shared4_objective_suite_h200.sh)

    if truthy "$DRY_RUN"; then
      print_command sbatch "${sbatch_args[@]}"
      job_id="DRYRUN-${job_name}"
    else
      job_id="$(sbatch "${sbatch_args[@]}")"
    fi
    submission_records+=("train phase=${phase} condition=${condition} target_steps=${target_steps} job_id=${job_id} dependency=${dependency}")
    previous_job_id="$job_id"
    echo "[submit] train phase=${phase} condition=${condition} target_steps=${target_steps} job_id=${job_id} dependency=${dependency}"
  done

  final_train_job_ids["$condition"]="$previous_job_id"
}

dependency_for_eval_task() {
  local task="$1"
  local default_dependency="${2:-}"
  case "$task" in
    discrete_trained) echo "${final_train_job_ids[discrete_rl]}" ;;
    multiplex_trained) echo "${final_train_job_ids[multiplex_thinking]}" ;;
    shared4_joint_trained) echo "${final_train_job_ids[shared4_joint]}" ;;
    shared4_thinking_trained) echo "${final_train_job_ids[shared4_thinking_only]}" ;;
    shared4_answer_trained) echo "${final_train_job_ids[shared4_answer_only]}" ;;
    *) echo "$default_dependency" ;;
  esac
}

submit_eval_phase() {
  local phase="$1"
  local start_dependency="${2:-}"
  shift 2
  local tasks=("$@")
  local -n phase_eval_ids_ref="${phase}_eval_job_ids"
  local seed_index=0

  for seed in "${eval_seeds[@]}"; do
    local repeat_label
    repeat_label="$(printf 'repeat_%02d' "$seed_index")"
    local task_index=0
    for eval_task in "${tasks[@]}"; do
      local slug padded_task job_name port dependency job_id
      slug="$(eval_slug "$eval_task")"
      printf -v padded_task "%02d" "$task_index"
      job_name="eval-${phase}-${slug}-r${seed_index}-${JOB_SUFFIX}"
      port="$((37100 + seed_index * 100 + task_index))"
      dependency="$(dependency_for_eval_task "$eval_task" "$start_dependency")"

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
      append_export EVAL_DISABLE_CUDA_GRAPH "$EVAL_DISABLE_CUDA_GRAPH"
      append_export EVAL_MEM_FRACTION_STATIC "$EVAL_MEM_FRACTION_STATIC"

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
        --gres="$GRES_SPEC"
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
      phase_eval_ids_ref+=("$job_id")
      submission_records+=("eval phase=${phase} task=${eval_task} seed=${seed} job_id=${job_id} dependency=${dependency:-none}")
      echo "[submit] eval phase=${phase} task=${eval_task} seed=${seed} job_id=${job_id} dependency=${dependency:-none}"
      task_index="$((task_index + 1))"
    done
    seed_index="$((seed_index + 1))"
  done
}

submit_aggregate_phase() {
  local phase="$1"
  shift
  local deps=("$@")
  local aggregate_dependency job_id
  aggregate_dependency="$(IFS=:; echo "${deps[*]}")"

  export_args="ALL"
  append_export TASK_MODE aggregate
  append_export RUN_TAG "$RUN_TAG"
  append_export OUTPUT_DIR "$OUTPUT_DIR"
  append_export GPUS_PER_NODE "$GPUS_PER_NODE"
  append_export ACCELERATOR_LABEL "$JOB_ACCELERATOR_LABEL"
  append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
  append_export AGGREGATE_BOOTSTRAP_RUNS "$AGGREGATE_BOOTSTRAP_RUNS"
  append_export AGGREGATE_SEED "${AGGREGATE_SEED:-26010808}"
  append_export AGGREGATE_LABEL "$phase"

  aggregate_sbatch_args=(
    --parsable
    --job-name="aggregate-${phase}-${JOB_SUFFIX}"
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --time="$POSTPROCESS_WALLTIME"
    --gres="$GRES_SPEC"
    --constraint="$CONSTRAINT"
    --output="${REPO_ROOT}/slurm_logs/%x-%j.out"
    --chdir="$REPO_ROOT"
    --export="$export_args"
    --dependency="afterok:${aggregate_dependency}"
    scripts/run_shared4_objective_suite_h200.sh
  )
  if truthy "$DRY_RUN"; then
    print_command sbatch "${aggregate_sbatch_args[@]}"
    job_id="DRYRUN-aggregate-${phase}-${JOB_SUFFIX}"
  else
    job_id="$(sbatch "${aggregate_sbatch_args[@]}")"
  fi
  submission_records+=("aggregate phase=${phase} job_id=${job_id} dependency=${aggregate_dependency}")
  echo "[submit] aggregate phase=${phase} job_id=${job_id} dependency=${aggregate_dependency}"
  LAST_AGGREGATE_JOB_ID="$job_id"
}

for condition in "${PHASE1_TRAIN_CONDITIONS[@]}"; do
  submit_train_chain phase1 "$condition"
done
phase1_train_dependency="$(IFS=:; echo "${final_train_job_ids[multiplex_thinking]}:${final_train_job_ids[shared4_joint]}")"
submit_eval_phase phase1 "$phase1_train_dependency" "${PHASE1_EVAL_TASKS[@]}"
submit_aggregate_phase phase1 "${phase1_eval_job_ids[@]}"
phase1_aggregate_job_id="$LAST_AGGREGATE_JOB_ID"

for condition in "${PHASE2_TRAIN_CONDITIONS[@]}"; do
  submit_train_chain phase2 "$condition" "$phase1_aggregate_job_id"
done
submit_eval_phase phase2 "$phase1_aggregate_job_id" "${PHASE2_EVAL_TASKS[@]}"
submit_aggregate_phase phase2 "$phase1_aggregate_job_id" "${phase2_eval_job_ids[@]}"
phase2_aggregate_job_id="$LAST_AGGREGATE_JOB_ID"

for condition in "${PHASE3_TRAIN_CONDITIONS[@]}"; do
  submit_train_chain phase3 "$condition" "$phase2_aggregate_job_id"
done
submit_eval_phase phase3 "$phase2_aggregate_job_id" "${PHASE3_EVAL_TASKS[@]}"
submit_aggregate_phase phase3 "$phase2_aggregate_job_id" "${phase3_eval_job_ids[@]}"
phase3_aggregate_job_id="$LAST_AGGREGATE_JOB_ID"

SUMMARY_PATH="${OUTPUT_DIR}/h200_staged_priority_submission.txt"
{
  echo "run_tag=${RUN_TAG}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "total_steps_per_variant=${TOTAL_STEPS_PER_VARIANT}"
  echo "shard_steps=${SHARD_STEPS}"
  echo "target_steps=${targets[*]}"
  echo "constraint=${CONSTRAINT}"
  echo "job_accelerator_label=${JOB_ACCELERATOR_LABEL}"
  echo "gpus_per_job=${GPUS_PER_NODE}"
  echo "gres=${GRES_SPEC}"
  echo "mem=${MEM}"
  echo "train_walltime=${TRAIN_WALLTIME}"
  echo "eval_walltime=${EVAL_WALLTIME}"
  echo "postprocess_walltime=${POSTPROCESS_WALLTIME}"
  echo "first_results_minutes=${first_results_minutes}"
  echo "max_first_results_hours=${MAX_FIRST_RESULTS_HOURS}"
  echo "train_examples=${TRAIN_EXAMPLES}"
  echo "eval_seeds=${eval_seeds[*]}"
  echo "eval_max_k=${EVAL_MAX_K}"
  echo "eval_max_prompts=${EVAL_MAX_PROMPTS}"
	  echo "eval_max_new_tokens=${EVAL_MAX_NEW_TOKENS}"
	  echo "eval_disable_cuda_graph=${EVAL_DISABLE_CUDA_GRAPH}"
	  echo "eval_mem_fraction_static=${EVAL_MEM_FRACTION_STATIC}"
	  echo "aggregate_bootstrap_runs=${AGGREGATE_BOOTSTRAP_RUNS}"
	  echo "reasoning_prefix_tokens=${REASONING_PREFIX_TOKENS}"
  echo "enforce_eager=${ENFORCE_EAGER}"
  echo "checkpoint_save_contents=${CHECKPOINT_SAVE_CONTENTS}"
  echo "checkpoint_load_contents=${CHECKPOINT_LOAD_CONTENTS}"
  echo
	  echo "phases:"
	  echo "- phase1: train=${PHASE1_TRAIN_CONDITIONS[*]} eval=${PHASE1_EVAL_TASKS[*]} aggregate=${phase1_aggregate_job_id}"
	  echo "- phase1_baseline_eval_dependency=${phase1_train_dependency:-none}"
  echo "- phase2: train=${PHASE2_TRAIN_CONDITIONS[*]} eval=${PHASE2_EVAL_TASKS[*]} aggregate=${phase2_aggregate_job_id}"
  echo "- phase3: train=${PHASE3_TRAIN_CONDITIONS[*]} eval=${PHASE3_EVAL_TASKS[*]} aggregate=${phase3_aggregate_job_id}"
  echo
  echo "runtime_train_profiles:"
  train_knobs_for_condition multiplex_thinking
  echo "- ${JOB_ACCELERATOR_LABEL} discrete/multiplex: train_batch_size=${TRAIN_BATCH_SIZE_FOR_JOB} ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE_FOR_JOB} rollout_n=${ROLLOUT_N_FOR_JOB} max_response_length=${MAX_RESPONSE_LENGTH_FOR_JOB} max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU_FOR_JOB} rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS_FOR_JOB} gpu_mem_util=${GPU_MEM_UTIL_FOR_JOB} enforce_eager=${ENFORCE_EAGER} save_freq=${SAVE_FREQ_FOR_JOB}"
  train_knobs_for_condition shared4_joint
  echo "- ${JOB_ACCELERATOR_LABEL} shared4: train_batch_size=${TRAIN_BATCH_SIZE_FOR_JOB} ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE_FOR_JOB} rollout_n=${ROLLOUT_N_FOR_JOB} max_response_length=${MAX_RESPONSE_LENGTH_FOR_JOB} max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU_FOR_JOB} rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS_FOR_JOB} gpu_mem_util=${GPU_MEM_UTIL_FOR_JOB} enforce_eager=${ENFORCE_EAGER} save_freq=${SAVE_FREQ_FOR_JOB}"
	  echo "- OOM retry keeps prompt batch >=32 and PPO minibatch >=16, lowers token/sequence caps, and lowers gpu_mem_util to 0.66"
	  echo "- Eval disables CUDA graph capture by default and uses mem_fraction_static=${EVAL_MEM_FRACTION_STATIC} to avoid startup health-check stalls"
  echo
  echo "final_train_job_ids:"
  for condition in multiplex_thinking shared4_joint shared4_answer_only shared4_thinking_only discrete_rl; do
    echo "- ${condition}=${final_train_job_ids[$condition]:-none}"
  done
  echo
  echo "submitted_jobs:"
  for record in "${submission_records[@]}"; do
    echo "- ${record}"
  done
} > "$SUMMARY_PATH"

echo "[submit] summary=${SUMMARY_PATH}"
