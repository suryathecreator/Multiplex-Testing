#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/final_eval_outputs/aime-train100-ablation4096-r3-stable3b-20260501-103533}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run_cuda_graph_smoke.sh}"

MAX_PROMPTS="${MAX_PROMPTS:-100}"
MAX_K="${MAX_K:-32}"
PROMPT_SHARDS="${PROMPT_SHARDS:-50}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-10}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-raivn-ckpt}"
SLURM_GPU_PARTITION="${SLURM_GPU_PARTITION:-ckpt-all}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"
SLURM_GPU_TIME="${SLURM_GPU_TIME:-09:05:00}"
SQUEUE_USER="${SQUEUE_USER:-${USER:-suryadv}}"

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

SBATCH_GPU_COMMON=(
  --account="$SLURM_ACCOUNT"
  --partition="$SLURM_GPU_PARTITION"
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --mem=128G
  --gres="$SLURM_GPU_GRES"
  --time="$SLURM_GPU_TIME"
  --export=ALL
)
if [[ -n "$SLURM_GPU_CONSTRAINT" ]]; then
  SBATCH_GPU_COMMON+=(--constraint="$SLURM_GPU_CONSTRAINT")
fi

prompt_indices_for_shard() {
  local shard_index="$1"
  local shard_count="$2"
  local total="${#SELECTED_AIME_TRAIN_INDICES[@]}"
  local start=$((shard_index * total / shard_count))
  local end=$(((shard_index + 1) * total / shard_count))
  local indices=""
  local idx
  for ((idx = start; idx < end; idx++)); do
    if [[ -n "$indices" ]]; then
      indices+=","
    fi
    indices+="${SELECTED_AIME_TRAIN_INDICES[$idx]}"
  done
  echo "$indices"
}

submit_topup_job() {
  local job_name="$1"
  local output_dir="$2"
  local seed="$3"
  local port="$4"
  local base_group="$5"
  local topup_group="$6"
  local prompt_indices="$7"
  local campaign="$8"
  local before_label="$9"
  local after_label="${10}"
  local plot_title="${11}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] ${job_name} output=${output_dir} base=${base_group} topup_group=${topup_group} prompts=${prompt_indices} campaign=${campaign}" >&2
    echo "DRYRUN-${job_name}"
    return
  fi

  local existing_job
  existing_job="$(squeue -h -u "$SQUEUE_USER" -n "$job_name" -o "%i" | head -1 || true)"
  if [[ -n "$existing_job" ]]; then
    echo "[skip-active] ${job_name} existing_job=${existing_job}" >&2
    echo "$existing_job"
    return
  fi

  mkdir -p "$output_dir"
  local job_id
  job_id="$(
    env \
      BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
      BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
      MPLCONFIGDIR=/tmp \
      EXPERIMENT_MODE=memory_match_topup \
      BENCHMARK=deepscaler_aime_train \
      RUN_TAG="$job_name" \
      OUTPUT_DIR="$output_dir" \
      MEMORY_MATCH_SOURCE_ROOT="$ABLATION_ROOT" \
      MEMORY_MATCH_SHARED_GROUPS="$base_group" \
      MEMORY_MATCH_TOPUP_GENERATOR=shared_group \
      MEMORY_MATCH_TOPUP_SHARED_GROUP_SIZE="$topup_group" \
      MEMORY_MATCH_CHECKPOINT_FAMILY=shared_topup \
      MEMORY_MATCH_CHECKPOINT_ROOT="$ABLATION_ROOT" \
      MEMORY_MATCH_CHECKPOINT_CAMPAIGN="$campaign" \
      MEMORY_MATCH_CHECKPOINT_STEP="$CHECKPOINT_STEP" \
      MEMORY_MATCH_CHECKPOINT_BEFORE_LABEL="$before_label" \
      MEMORY_MATCH_CHECKPOINT_AFTER_LABEL="$after_label" \
      MEMORY_MATCH_CHECKPOINT_TITLE="$plot_title" \
      PROMPT_INDICES="$prompt_indices" \
      PORT="$port" \
      MAX_K="$MAX_K" \
      MAX_PROMPTS="$MAX_PROMPTS" \
      METHODS=baseline,shared_trace \
      SEED="$seed" \
      MAX_NEW_TOKENS=8192 \
      BRANCH_ABLATION_REASONING_PREFIX_TOKENS=4096 \
      BRANCH_ABLATION_GROUP_SIZES="$base_group" \
      BRANCH_ABLATION_NO_BASELINE=1 \
      CHECKPOINT_MATCHED_PROMPTS_STEP=5 \
      COMPACT_JSONL=1 \
      DP_SIZE=2 \
      TP_SIZE=1 \
      FALLBACK_CONFIGS="${FALLBACK_CONFIGS:-32:2:4:aggressive;16:1:2:aggressive}" \
      SERVER_TIMEOUT_SECONDS=900 \
      sbatch --parsable \
        --job-name="$job_name" \
        --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
        "${SBATCH_GPU_COMMON[@]}" \
        "$RUNNER_SCRIPT"
  )"
  echo "${job_id%%;*}"
}

declare -a CAMPAIGN_SPECS=(
  "memory_match_topup_shared2_shared8_fast|s2s8top|2|8|Shared every 2|Shared every 2 + shared8 top-up|AIME Train: Shared Every 2 Shared8 Top-Up"
  "memory_match_topup_shared8_shared8_fast|s8s8top|8|8|Shared every 8|Shared every 8 + shared8 top-up|AIME Train: Shared Every 8 Shared8 Top-Up"
  "memory_match_topup_shared16_shared16_fast|s16s16top|16|16|Shared every 16|Shared every 16 + shared16 top-up|AIME Train: Shared Every 16 Shared16 Top-Up"
)

echo "[shared8/16-topup] ablation_root=${ABLATION_ROOT}"
echo "[shared8/16-topup] prompt_shards=${PROMPT_SHARDS} checkpoint_step=${CHECKPOINT_STEP}"
echo "[shared8/16-topup] gpu_partition=${SLURM_GPU_PARTITION} gpu_gres=${SLURM_GPU_GRES} gpu_constraint=${SLURM_GPU_CONSTRAINT:-none}"

all_jobs=()
campaign_index=0
for spec in "${CAMPAIGN_SPECS[@]}"; do
  IFS='|' read -r campaign job_prefix base_group topup_group before_label after_label plot_title <<<"$spec"
  campaign_jobs=()
  for repeat_index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$repeat_index]}"
    repeat_dir="${ABLATION_ROOT}/repeat_$(printf '%02d' "$repeat_index")"
    for ((shard_index = 0; shard_index < PROMPT_SHARDS; shard_index++)); do
      shard_label="$(printf 'shard%02d' "$shard_index")"
      prompt_indices="$(prompt_indices_for_shard "$shard_index" "$PROMPT_SHARDS")"
      port=$((43000 + campaign_index * 1000 + repeat_index * 100 + shard_index))
      job_id="$(
        submit_topup_job \
          "aime4096-${job_prefix}-r${repeat_index}-${shard_label}" \
          "${repeat_dir}/${campaign}/${shard_label}" \
          "$seed" \
          "$port" \
          "$base_group" \
          "$topup_group" \
          "$prompt_indices" \
          "$campaign" \
          "$before_label" \
          "$after_label" \
          "$plot_title"
      )"
      campaign_jobs+=("$job_id")
      all_jobs+=("$job_id")
      echo "[shared8/16-topup] campaign=${campaign} repeat=${repeat_index} shard=${shard_label} job=${job_id}"
    done
  done
  echo "[shared8/16-topup] campaign=${campaign} jobs=$(IFS=:; echo "${campaign_jobs[*]}")"
  campaign_index=$((campaign_index + 1))
done

echo "[shared8/16-topup] all_jobs=$(IFS=:; echo "${all_jobs[*]}")"
echo "[shared8/16-topup] done"
