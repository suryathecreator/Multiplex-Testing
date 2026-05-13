#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/final_eval_outputs/aime-train100-ablation4096-r3-stable3b-20260501-103533}"
MIX_ROOT="${MIX_ROOT:-${REPO_ROOT}/final_eval_outputs/aime-train100-fixed-shared-mix4096-r3-mix4096-20260501-221133}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run_cuda_graph_smoke.sh}"

MAX_PROMPTS="${MAX_PROMPTS:-100}"
MAX_K="${MAX_K:-32}"
SHARED2_PROMPT_SHARDS="${SHARED2_PROMPT_SHARDS:-20}"
MIX_PROMPT_SHARDS="${MIX_PROMPT_SHARDS:-20}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-raivn-ckpt}"
SLURM_GPU_PARTITION="${SLURM_GPU_PARTITION:-ckpt-all}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"
SLURM_GPU_TIME="${SLURM_GPU_TIME:-09:05:00}"
MIX_NICE="${MIX_NICE:-1000}"

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
  local source_root="$3"
  local seed="$4"
  local port="$5"
  local group_size="$6"
  local prompt_indices="$7"
  local experiment_mode="$8"
  local topup_generator="$9"
  local checkpoint_family="${10}"
  local checkpoint_root="${11}"
  local checkpoint_campaign="${12}"
  local nice_value="${13:-}"

  local nice_args=()
  if [[ -n "$nice_value" ]]; then
    nice_args=(--nice="$nice_value")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] ${job_name} output=${output_dir} mode=${experiment_mode} source=${source_root} group=${group_size} generator=${topup_generator} prompts=${prompt_indices} checkpoint=${checkpoint_family}/${checkpoint_campaign} nice=${nice_value:-default}" >&2
    echo "DRYRUN-${job_name}"
    return
  fi

  mkdir -p "$output_dir"
  local job_id
  job_id="$(
    env \
      BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
      BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
      MPLCONFIGDIR=/tmp \
      EXPERIMENT_MODE="$experiment_mode" \
      BENCHMARK=deepscaler_aime_train \
      RUN_TAG="$job_name" \
      OUTPUT_DIR="$output_dir" \
      MEMORY_MATCH_SOURCE_ROOT="$source_root" \
      MEMORY_MATCH_SHARED_GROUPS="$group_size" \
      MEMORY_MATCH_TOPUP_GENERATOR="$topup_generator" \
      MEMORY_MATCH_CHECKPOINT_FAMILY="$checkpoint_family" \
      MEMORY_MATCH_CHECKPOINT_ROOT="$checkpoint_root" \
      MEMORY_MATCH_CHECKPOINT_CAMPAIGN="$checkpoint_campaign" \
      MEMORY_MATCH_CHECKPOINT_STEP=20 \
      PROMPT_INDICES="$prompt_indices" \
      PORT="$port" \
      MAX_K="$MAX_K" \
      MAX_PROMPTS="$MAX_PROMPTS" \
      METHODS=baseline,shared_trace \
      SEED="$seed" \
      MAX_NEW_TOKENS=8192 \
      BRANCH_ABLATION_REASONING_PREFIX_TOKENS=4096 \
      BRANCH_ABLATION_GROUP_SIZES="$group_size" \
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
        "${nice_args[@]}" \
        "${SBATCH_GPU_COMMON[@]}" \
        "$RUNNER_SCRIPT"
  )"
  echo "${job_id%%;*}"
}

echo "[targeted-topup] ablation_root=${ABLATION_ROOT}"
echo "[targeted-topup] mix_root=${MIX_ROOT}"
echo "[targeted-topup] shared2_prompt_shards=${SHARED2_PROMPT_SHARDS}"
echo "[targeted-topup] mix_prompt_shards=${MIX_PROMPT_SHARDS} mix_nice=${MIX_NICE}"
echo "[targeted-topup] checkpoint_step=20"
echo "[targeted-topup] gpu_partition=${SLURM_GPU_PARTITION} gpu_gres=${SLURM_GPU_GRES} gpu_constraint=${SLURM_GPU_CONSTRAINT:-none}"

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -s "${MIX_ROOT}/aggregate/per_prompt_pure.csv" || ! -s "${MIX_ROOT}/aggregate/per_prompt_mixed.csv" ]]; then
    echo "[targeted-topup] missing ${MIX_ROOT}/aggregate/per_prompt_{pure,mixed}.csv; refresh the mix aggregate before submitting" >&2
    exit 1
  fi
  echo "[targeted-topup] using existing ${MIX_ROOT}/aggregate/per_prompt_{pure,mixed}.csv"
else
  echo "[dry-run] would require existing ${MIX_ROOT}/aggregate/per_prompt_{pure,mixed}.csv" >&2
fi

shared2_jobs=()
SHARED2_CAMPAIGN="memory_match_topup_shared2_sharedgen_fast"
for repeat_index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$repeat_index]}"
  repeat_dir="${ABLATION_ROOT}/repeat_$(printf '%02d' "$repeat_index")"
  for ((shard_index = 0; shard_index < SHARED2_PROMPT_SHARDS; shard_index++)); do
    shard_label="$(printf 'shard%02d' "$shard_index")"
    prompt_indices="$(prompt_indices_for_shard "$shard_index" "$SHARED2_PROMPT_SHARDS")"
    port=$((37000 + repeat_index * 100 + shard_index))
    job_id="$(
      submit_topup_job \
        "aime4096-s2sharedtop-r${repeat_index}-${shard_label}" \
        "${repeat_dir}/${SHARED2_CAMPAIGN}/${shard_label}" \
        "$repeat_dir" \
        "$seed" \
        "$port" \
        2 \
        "$prompt_indices" \
        memory_match_topup \
        shared_group \
        shared2 \
        "$ABLATION_ROOT" \
        "$SHARED2_CAMPAIGN"
    )"
    shared2_jobs+=("$job_id")
    echo "[targeted-topup] shared2 repeat=${repeat_index} shard=${shard_label} job=${job_id}"
  done
done

mix_jobs=()
MIX_CAMPAIGN="memory_match_topup_fixed_shared_fixedgen_fast"
for repeat_index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$repeat_index]}"
  repeat_dir="${MIX_ROOT}/repeat_$(printf '%02d' "$repeat_index")"
  for ((shard_index = 0; shard_index < MIX_PROMPT_SHARDS; shard_index++)); do
    shard_label="$(printf 'shard%02d' "$shard_index")"
    prompt_indices="$(prompt_indices_for_shard "$shard_index" "$MIX_PROMPT_SHARDS")"
    port=$((38000 + repeat_index * 100 + shard_index))
    job_id="$(
      submit_topup_job \
        "aime4096-mixfixtop-r${repeat_index}-${shard_label}" \
        "${repeat_dir}/${MIX_CAMPAIGN}/${shard_label}" \
        "$MIX_ROOT" \
        "$seed" \
        "$port" \
        32 \
        "$prompt_indices" \
        mixed_memory_match_topup \
        fixed \
        mixed_fixed_shared \
        "$MIX_ROOT" \
        "$MIX_CAMPAIGN" \
        "$MIX_NICE"
    )"
    mix_jobs+=("$job_id")
    echo "[targeted-topup] mix repeat=${repeat_index} shard=${shard_label} job=${job_id}"
  done
done

echo "[targeted-topup] shared2_jobs=$(IFS=:; echo "${shared2_jobs[*]}")"
echo "[targeted-topup] mix_jobs=$(IFS=:; echo "${mix_jobs[*]}")"
echo "[targeted-topup] done"
