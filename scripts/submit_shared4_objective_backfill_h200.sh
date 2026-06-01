#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-shared4-objective-backfill-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/${RUN_TAG}}"
TOTAL_STEPS_PER_VARIANT="${TOTAL_STEPS_PER_VARIANT:-64}"
SHARD_STEPS="${SHARD_STEPS:-8}"
RUN_CONDITIONS="${RUN_CONDITIONS:-shared4_joint,shared4_thinking_only,shared4_answer_only}"

ACCOUNT="${ACCOUNT:-raivn-ckpt}"
PARTITION="${PARTITION:-ckpt-all}"
CONSTRAINT="${CONSTRAINT:-h200}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-192G}"
WALLTIME="${WALLTIME:-03:00:00}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
AUTO_OOM_RETRY="${AUTO_OOM_RETRY:-True}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-16384}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.76}"
SAVE_FREQ="${SAVE_FREQ:-2}"

DRY_RUN="${DRY_RUN:-False}"

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

condition_slug() {
  case "$1" in
    shared4_joint) echo "s4j" ;;
    shared4_thinking_only) echo "s4t" ;;
    shared4_answer_only) echo "s4a" ;;
    *)
      echo "[config] unknown condition: $1" >&2
      exit 1
      ;;
  esac
}

latest_checkpoint_step() {
  local checkpoint_dir="$1"
  if [[ ! -d "$checkpoint_dir" ]]; then
    echo 0
    return 0
  fi
  local tracker="${checkpoint_dir}/latest_checkpointed_iteration.txt"
  if [[ -f "$tracker" ]]; then
    local tracked_step
    tracked_step="$(tr -dc '0-9' < "$tracker" || true)"
    if [[ -n "$tracked_step" ]]; then
      echo "$tracked_step"
      return 0
    fi
  fi
  find "$checkpoint_dir" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null \
    | sed -E 's/^global_step_//' \
    | sort -n \
    | tail -n 1
}

append_export() {
  local name="$1"
  local value="$2"
  export_args+=",${name}=${value}"
}

print_command() {
  printf '[dry-run]'
  printf ' %q' "$@"
  printf '\n'
}

require_positive_int TOTAL_STEPS_PER_VARIANT "$TOTAL_STEPS_PER_VARIANT"
require_positive_int SHARD_STEPS "$SHARD_STEPS"
require_positive_int GPUS_PER_NODE "$GPUS_PER_NODE"
require_positive_int CPUS_PER_TASK "$CPUS_PER_TASK"

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

IFS=',' read -r -a raw_conditions <<< "$RUN_CONDITIONS"
conditions=()
for raw_condition in "${raw_conditions[@]}"; do
  condition="${raw_condition//[[:space:]]/}"
  [[ -z "$condition" ]] && continue
  condition_slug "$condition" >/dev/null
  conditions+=("$condition")
done

if [[ "${#conditions[@]}" -eq 0 ]]; then
  echo "[config] no valid RUN_CONDITIONS were provided" >&2
  exit 1
fi

submission_records=()
skipped_records=()

for condition in "${conditions[@]}"; do
  previous_job_id=""
  checkpoint_dir="${OUTPUT_DIR}/checkpoints/${condition}"
  latest_step="$(latest_checkpoint_step "$checkpoint_dir")"
  latest_step="${latest_step:-0}"

  for target_steps in "${targets[@]}"; do
    if [[ "$latest_step" -ge "$target_steps" ]]; then
      skipped_records+=("condition=${condition} target_steps=${target_steps} latest_step=${latest_step}")
      echo "[submit] skip condition=${condition} target_steps=${target_steps}; latest_step=${latest_step}"
      continue
    fi

    slug="$(condition_slug "$condition")"
    printf -v padded_target "%03d" "$target_steps"
    job_name="${slug}-${padded_target}"
    dependency="${previous_job_id:-none}"

    export_args="ALL"
    append_export RUN_TAG "$RUN_TAG"
    append_export OUTPUT_DIR "$OUTPUT_DIR"
    append_export RUN_CONDITIONS "$condition"
    append_export TOTAL_STEPS_PER_VARIANT "$TOTAL_STEPS_PER_VARIANT"
    append_export TRAIN_TARGET_STEPS "$target_steps"
    append_export GPUS_PER_NODE "$GPUS_PER_NODE"
    append_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
    append_export AUTO_OOM_RETRY "$AUTO_OOM_RETRY"
    append_export TRAIN_BATCH_SIZE "$TRAIN_BATCH_SIZE"
    append_export PPO_MINI_BATCH_SIZE "$PPO_MINI_BATCH_SIZE"
    append_export MAX_TOKEN_LEN_PER_GPU "$MAX_TOKEN_LEN_PER_GPU"
    append_export ROLLOUT_MAX_NUM_SEQS "$ROLLOUT_MAX_NUM_SEQS"
    append_export GPU_MEM_UTIL "$GPU_MEM_UTIL"
    append_export SAVE_FREQ "$SAVE_FREQ"

    sbatch_args=(
      --parsable
      --job-name="$job_name"
      --account="$ACCOUNT"
      --partition="$PARTITION"
      --nodes=1
      --ntasks=1
      --cpus-per-task="$CPUS_PER_TASK"
      --mem="$MEM"
      --time="$WALLTIME"
      --gres="gpu:${GPUS_PER_NODE}"
      --constraint="$CONSTRAINT"
      --output="${REPO_ROOT}/slurm_logs/%x-%j.out"
      --chdir="$REPO_ROOT"
      --export="$export_args"
    )
    if [[ -n "$previous_job_id" ]]; then
      sbatch_args+=(--dependency="afterany:${previous_job_id}")
    fi
    sbatch_args+=(scripts/run_shared4_objective_suite_h200.sh)

    if truthy "$DRY_RUN"; then
      print_command sbatch "${sbatch_args[@]}"
      job_id="DRYRUN-${slug}-${padded_target}"
    else
      job_id="$(sbatch "${sbatch_args[@]}")"
    fi

    submission_records+=("condition=${condition} target_steps=${target_steps} job_id=${job_id} dependency=${dependency}")
    previous_job_id="$job_id"
    echo "[submit] condition=${condition} target_steps=${target_steps} job_id=${job_id} dependency=${dependency}"
  done
done

SUMMARY_PATH="${OUTPUT_DIR}/shared4_objective_backfill_submission.txt"
{
  echo "run_tag=${RUN_TAG}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "total_steps_per_variant=${TOTAL_STEPS_PER_VARIANT}"
  echo "shard_steps=${SHARD_STEPS}"
  echo "target_steps=${targets[*]}"
  echo "conditions=${conditions[*]}"
  echo "account=${ACCOUNT}"
  echo "partition=${PARTITION}"
  echo "constraint=${CONSTRAINT}"
  echo "gpus_per_job=${GPUS_PER_NODE}"
  echo "cpus_per_task=${CPUS_PER_TASK}"
  echo "mem=${MEM}"
  echo "walltime=${WALLTIME}"
  echo "attn_implementation=${ATTN_IMPLEMENTATION}"
  echo "train_batch_size=${TRAIN_BATCH_SIZE}"
  echo "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  echo "max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
  echo "rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
  echo "gpu_mem_util=${GPU_MEM_UTIL}"
  echo "save_freq=${SAVE_FREQ}"
  echo "auto_oom_retry=${AUTO_OOM_RETRY}"
  echo
  echo "submitted_jobs:"
  for record in "${submission_records[@]}"; do
    echo "- ${record}"
  done
  echo
  echo "skipped_targets:"
  for record in "${skipped_records[@]}"; do
    echo "- ${record}"
  done
} > "$SUMMARY_PATH"

echo "[submit] summary=${SUMMARY_PATH}"
