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
SLURM_GPU_EXCLUDE="${SLURM_GPU_EXCLUDE:-g3042,g3043,g3050,g3051,g3061,g3063,g3071,g3075,g3099}"
SLURM_GPU_TIME="${SLURM_GPU_TIME:-09:05:00}"

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
if [[ -n "$SLURM_GPU_EXCLUDE" ]]; then
  SBATCH_GPU_COMMON+=(--exclude="$SLURM_GPU_EXCLUDE")
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

submit_rescue_job() {
  local repeat_index="$1"
  local shard_index="$2"
  local campaign="$3"
  local job_prefix="$4"
  local base_group="$5"
  local topup_generator="$6"
  local topup_group="$7"
  local before_label="$8"
  local after_label="$9"
  local plot_title="${10}"

  local seed="${SEEDS[$repeat_index]}"
  local shard_label
  shard_label="$(printf 'shard%02d' "$shard_index")"
  local job_name="aime4096-${job_prefix}-r${repeat_index}-${shard_label}-rescue2"
  local output_dir="${ABLATION_ROOT}/repeat_$(printf '%02d' "$repeat_index")/${campaign}/${shard_label}"
  local prompt_indices
  prompt_indices="$(prompt_indices_for_shard "$shard_index" "$PROMPT_SHARDS")"
  local port=$((50000 + repeat_index * 100 + shard_index))

  mkdir -p "$output_dir"
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
    MEMORY_MATCH_TOPUP_GENERATOR="$topup_generator" \
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
}

jobs=()
submit_and_log() {
  local job_id
  job_id="$(submit_rescue_job "$@")"
  jobs+=("$job_id")
  echo "[rescue2] repeat=$1 shard=$(printf 'shard%02d' "$2") campaign=$3 job=${job_id}"
}

submit_and_log 0 10 memory_match_topup_shared8_shared2_fast s8s2top 8 shared_group 2 "Shared every 8" "Shared every 8 + shared2 top-up" "AIME Train: Shared Every 8 Shared2 Top-Up"
submit_and_log 0 49 memory_match_topup_shared4_fixedgen_fast s4fixtop 4 fixed 0 "Shared every 4" "Shared every 4 + fixed top-up" "AIME Train: Shared Every 4 Fixed Top-Up"
submit_and_log 2 41 memory_match_topup_shared4_fixedgen_fast s4fixtop 4 fixed 0 "Shared every 4" "Shared every 4 + fixed top-up" "AIME Train: Shared Every 4 Fixed Top-Up"
submit_and_log 2 6 memory_match_topup_shared4_shared8_fast s4s8top 4 shared_group 8 "Shared every 4" "Shared every 4 + shared8 top-up" "AIME Train: Shared Every 4 Shared8 Top-Up"
submit_and_log 0 10 memory_match_topup_shared4_shared8_fast s4s8top 4 shared_group 8 "Shared every 4" "Shared every 4 + shared8 top-up" "AIME Train: Shared Every 4 Shared8 Top-Up"
submit_and_log 0 49 memory_match_topup_shared4_shared8_fast s4s8top 4 shared_group 8 "Shared every 4" "Shared every 4 + shared8 top-up" "AIME Train: Shared Every 4 Shared8 Top-Up"
submit_and_log 0 8 memory_match_topup_shared4_shared8_fast s4s8top 4 shared_group 8 "Shared every 4" "Shared every 4 + shared8 top-up" "AIME Train: Shared Every 4 Shared8 Top-Up"
submit_and_log 1 37 memory_match_topup_shared4_shared8_fast s4s8top 4 shared_group 8 "Shared every 4" "Shared every 4 + shared8 top-up" "AIME Train: Shared Every 4 Shared8 Top-Up"

echo "[rescue2] jobs=$(IFS=:; echo "${jobs[*]}")"
