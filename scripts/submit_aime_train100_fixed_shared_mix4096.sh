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
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/final_eval_outputs/aime-train100-fixed-shared-mix4096-r3-${STAMP}}"
MAX_PROMPTS="${MAX_PROMPTS:-100}"
MAX_K="${MAX_K:-32}"
PROMPT_SHARDS="${PROMPT_SHARDS:-4}"
SEEDS=(409600 409601 409602)
SELECTED_AIME_TRAIN_INDICES=(
  8 18 19 34 40 42 46 54 58 75 78 80 87 89 120 123 130
  140 150 154 173 180 187 239 249 256 271 278 283 288 291
  292 311 343 350 353 363 365 367 374 379 383 390 393 399
  403 426 428 432 453 459 462 466 467 506 509 523 527 550
  551 557 581 582 590 600 609 617 622 623 627 632 634 636
  641 646 665 671 711 739 749 751 755 756 760 793 804 810
  811 812 814 831 847 865 872 910 921 922 944 954 967
)

if ! [[ "$PROMPT_SHARDS" =~ ^[0-9]+$ ]] || [[ "$PROMPT_SHARDS" -lt 1 ]]; then
  echo "PROMPT_SHARDS must be a positive integer, got: ${PROMPT_SHARDS}" >&2
  exit 2
fi

mkdir -p slurm_logs
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$ROOT_DIR"
fi

SLURM_ACCOUNT="${SLURM_ACCOUNT:-raivn-ckpt}"
SLURM_GPU_PARTITION="${SLURM_GPU_PARTITION:-ckpt-all}"
SLURM_CPU_PARTITION="${SLURM_CPU_PARTITION:-$SLURM_GPU_PARTITION}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"
SLURM_CPU_GRES="${SLURM_CPU_GRES:-}"
SLURM_GPU_TIME="${SLURM_GPU_TIME:-09:05:00}"
SLURM_CPU_TIME="${SLURM_CPU_TIME:-02:00:00}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run_cuda_graph_smoke.sh}"

SBATCH_GPU_COMMON=(
  --account="$SLURM_ACCOUNT"
  --partition="$SLURM_GPU_PARTITION"
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --mem=128G
  --gres="$SLURM_GPU_GRES"
  --constraint="$SLURM_GPU_CONSTRAINT"
  --time="$SLURM_GPU_TIME"
  --export=ALL
)

SBATCH_CPU_COMMON=(
  --account="$SLURM_ACCOUNT"
  --partition="$SLURM_CPU_PARTITION"
  --nodes=1
  --ntasks=1
  --cpus-per-task=4
  --mem=32G
  --time="$SLURM_CPU_TIME"
  --export=ALL
)
if [[ -n "$SLURM_CPU_GRES" ]]; then
  SBATCH_CPU_COMMON+=(--gres="$SLURM_CPU_GRES")
fi

print_env_line() {
  printf '    %s=%q\n' "$1" "$2"
}

prompt_indices_for_shard() {
  local shard_index="$1"
  local start=$((shard_index * MAX_PROMPTS / PROMPT_SHARDS))
  local end=$(((shard_index + 1) * MAX_PROMPTS / PROMPT_SHARDS))
  local indices=""
  local idx
  for ((idx = start; idx < end; idx++)); do
    if [[ "$idx" -ge "${#SELECTED_AIME_TRAIN_INDICES[@]}" ]]; then
      break
    fi
    if [[ -n "$indices" ]]; then
      indices+=","
    fi
    indices+="${SELECTED_AIME_TRAIN_INDICES[$idx]}"
  done
  echo "$indices"
}

submit_generation_job() {
  local job_name="$1"
  local output_dir="$2"
  local seed="$3"
  local port="$4"
  local methods="$5"
  local group_sizes="$6"
  local no_baseline="$7"
  local fallback_configs="$8"
  local prompt_indices="$9"

  if [[ "$DRY_RUN" == "1" ]]; then
    {
      echo "[dry-run] sbatch ${job_name}"
      print_env_line OUTPUT_DIR "$output_dir"
      print_env_line SEED "$seed"
      print_env_line PORT "$port"
      print_env_line MAX_K "$MAX_K"
      print_env_line MAX_PROMPTS "$MAX_PROMPTS"
      print_env_line PROMPT_INDICES "$prompt_indices"
      print_env_line METHODS "$methods"
      print_env_line BRANCH_ABLATION_GROUP_SIZES "$group_sizes"
      print_env_line BRANCH_ABLATION_NO_BASELINE "$no_baseline"
      print_env_line FALLBACK_CONFIGS "$fallback_configs"
      print_env_line RUNNER_SCRIPT "$RUNNER_SCRIPT"
    } >&2
    echo "DRYRUN-${job_name}"
    return
  fi

  local job_id
  if ! job_id="$(
    env \
      BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
      BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
      MPLCONFIGDIR=/tmp \
      EXPERIMENT_MODE=branch_ablation \
      BENCHMARK=deepscaler_aime_train \
      RUN_TAG="$job_name" \
      OUTPUT_DIR="$output_dir" \
      PROMPT_INDICES="$prompt_indices" \
      PORT="$port" \
      MAX_K="$MAX_K" \
      MAX_PROMPTS="$MAX_PROMPTS" \
      METHODS="$methods" \
      SEED="$seed" \
      MAX_NEW_TOKENS=8192 \
      BRANCH_ABLATION_REASONING_PREFIX_TOKENS=4096 \
      BRANCH_ABLATION_GROUP_SIZES="$group_sizes" \
      BRANCH_ABLATION_NO_BASELINE="$no_baseline" \
      CHECKPOINT_MATCHED_PROMPTS_STEP=5 \
      COMPACT_JSONL=1 \
      DP_SIZE=2 \
      TP_SIZE=1 \
      FALLBACK_CONFIGS="$fallback_configs" \
      SERVER_TIMEOUT_SECONDS=900 \
      sbatch --parsable \
        --job-name="$job_name" \
        --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
        "${SBATCH_GPU_COMMON[@]}" \
        "$RUNNER_SCRIPT"
  )"; then
    echo "[submit] failed to submit ${job_name}" >&2
    exit 1
  fi
  echo "${job_id%%;*}"
}

submit_aggregate_job() {
  local dependency="$1"
  local job_name="aime4096-mix-aggregate-${STAMP}"
  local command="cd '$REPO_ROOT' && MPLCONFIGDIR=/tmp /mmfs1/home/suryadv/.conda/envs/multiplex-thinking/bin/python scripts/aggregate_aime_train100_fixed_shared_mix4096.py --root '$ROOT_DIR' --repeats 3 --seeds 409600,409601,409602 --max-k '$MAX_K'"

  if [[ "$DRY_RUN" == "1" ]]; then
    {
      echo "[dry-run] sbatch ${job_name}"
      print_env_line DEPENDENCY "$dependency"
      print_env_line COMMAND "$command"
    } >&2
    echo "DRYRUN-${job_name}"
    return
  fi

  local job_id
  if ! job_id="$(
    sbatch --parsable \
      --job-name="$job_name" \
      --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
      --dependency="$dependency" \
      "${SBATCH_CPU_COMMON[@]}" \
      --wrap "$command"
  )"; then
    echo "[submit] failed to submit ${job_name}" >&2
    exit 1
  fi
  echo "${job_id%%;*}"
}

echo "[submit] root_dir=${ROOT_DIR}"
echo "[submit] max_prompts=${MAX_PROMPTS} max_k=${MAX_K} reasoning_budget=4096"
echo "[submit] prompt_shards=${PROMPT_SHARDS}"
echo "[submit] slurm_account=${SLURM_ACCOUNT} gpu_partition=${SLURM_GPU_PARTITION} gpu_gres=${SLURM_GPU_GRES} gpu_time=${SLURM_GPU_TIME}"
echo "[submit] cpu_partition=${SLURM_CPU_PARTITION} cpu_gres=${SLURM_CPU_GRES:-none} cpu_time=${SLURM_CPU_TIME}"
echo "[submit] runner_script=${RUNNER_SCRIPT}"
echo "[submit] layout: 3 repeats x 2 conditions x ${PROMPT_SHARDS} prompt shards + 1 aggregate job"

generation_job_ids=()
for repeat_index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$repeat_index]}"
  repeat_dir="${ROOT_DIR}/repeat_$(printf '%02d' "$repeat_index")"
  port_base=$((33000 + repeat_index * 1000))
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$repeat_dir"
  fi

  for ((shard_index = 0; shard_index < PROMPT_SHARDS; shard_index++)); do
    shard_label="$(printf 'shard%02d' "$shard_index")"
    prompt_indices="$(prompt_indices_for_shard "$shard_index")"
    fixed_job="$(
      submit_generation_job \
        "aime4096-mix-r${repeat_index}-fixed32-${shard_label}" \
        "${repeat_dir}/fixed32_${shard_label}" \
        "$seed" \
        "$((port_base + 10 + shard_index))" \
        baseline \
        "" \
        0 \
        "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive" \
        "$prompt_indices"
    )"
    shared_job="$(
      submit_generation_job \
        "aime4096-mix-r${repeat_index}-shared32-${shard_label}" \
        "${repeat_dir}/shared32_${shard_label}" \
        "$seed" \
        "$((port_base + 110 + shard_index))" \
        shared_trace \
        32 \
        1 \
        "48:3:6:aggressive;32:2:4:aggressive;24:2:4:aggressive;16:1:2:aggressive" \
        "$prompt_indices"
    )"
    generation_job_ids+=("$fixed_job" "$shared_job")
    echo "[submit] repeat=${repeat_index} shard=${shard_label} fixed=${fixed_job} shared32=${shared_job}"
  done
done

aggregate_dependency="afterok:$(IFS=:; echo "${generation_job_ids[*]}")"
aggregate_job="$(submit_aggregate_job "$aggregate_dependency")"

echo "[submit] aggregate_job=${aggregate_job}"
echo "[submit] done"
