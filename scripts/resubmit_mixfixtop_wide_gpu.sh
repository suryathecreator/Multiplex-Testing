#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

MIX_ROOT="${MIX_ROOT:-${REPO_ROOT}/final_eval_outputs/aime-train100-fixed-shared-mix4096-r3-mix4096-20260501-221133}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run_cuda_graph_smoke.sh}"
JOB_NAMES_FILE="${JOB_NAMES_FILE:-}"

MAX_PROMPTS="${MAX_PROMPTS:-100}"
MAX_K="${MAX_K:-32}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-20}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-raivn-ckpt}"
SLURM_GPU_PARTITION="${SLURM_GPU_PARTITION:-ckpt-all}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"
SLURM_GPU_TIME="${SLURM_GPU_TIME:-09:05:00}"
SQUEUE_USER="${SQUEUE_USER:-${USER:-suryadv}}"

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
  local shard_count=20
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

if [[ -z "$JOB_NAMES_FILE" || ! -s "$JOB_NAMES_FILE" ]]; then
  echo "JOB_NAMES_FILE must point to a non-empty file of aime4096-mixfixtop job names" >&2
  exit 2
fi

echo "[mix-wide] mix_root=${MIX_ROOT}"
echo "[mix-wide] gpu_partition=${SLURM_GPU_PARTITION} gpu_gres=${SLURM_GPU_GRES} gpu_constraint=${SLURM_GPU_CONSTRAINT:-none}"
echo "[mix-wide] job_names_file=${JOB_NAMES_FILE}"

jobs=()
while IFS= read -r job_name; do
  [[ -z "$job_name" ]] && continue
  if [[ ! "$job_name" =~ ^aime4096-mixfixtop-r([0-2])-shard([0-9][0-9])$ ]]; then
    echo "[mix-wide] skip unparsable job name: $job_name" >&2
    continue
  fi
  repeat_index="${BASH_REMATCH[1]}"
  shard_index=$((10#${BASH_REMATCH[2]}))
  shard_label="$(printf 'shard%02d' "$shard_index")"
  seed="${SEEDS[$repeat_index]}"
  repeat_dir="${MIX_ROOT}/repeat_$(printf '%02d' "$repeat_index")"
  output_dir="${repeat_dir}/memory_match_topup_fixed_shared_fixedgen_fast/${shard_label}"
  prompt_indices="$(prompt_indices_for_shard "$shard_index")"
  port=$((43000 + repeat_index * 100 + shard_index))

  existing_job="$(squeue -h -u "$SQUEUE_USER" -n "$job_name" -o "%i" | head -1 || true)"
  if [[ -n "$existing_job" ]]; then
    echo "[mix-wide] skip-active ${job_name} existing_job=${existing_job}"
    jobs+=("$existing_job")
    continue
  fi

  mkdir -p "$output_dir"
  job_id="$(
    env \
      BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
      BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
      MPLCONFIGDIR=/tmp \
      EXPERIMENT_MODE=mixed_memory_match_topup \
      BENCHMARK=deepscaler_aime_train \
      RUN_TAG="$job_name" \
      OUTPUT_DIR="$output_dir" \
      MEMORY_MATCH_SOURCE_ROOT="$MIX_ROOT" \
      MEMORY_MATCH_SHARED_GROUPS=32 \
      MEMORY_MATCH_TOPUP_GENERATOR=fixed \
      MEMORY_MATCH_CHECKPOINT_FAMILY=mixed_fixed_shared \
      MEMORY_MATCH_CHECKPOINT_ROOT="$MIX_ROOT" \
      MEMORY_MATCH_CHECKPOINT_CAMPAIGN=memory_match_topup_fixed_shared_fixedgen_fast \
      MEMORY_MATCH_CHECKPOINT_STEP="$CHECKPOINT_STEP" \
      PROMPT_INDICES="$prompt_indices" \
      PORT="$port" \
      MAX_K="$MAX_K" \
      MAX_PROMPTS="$MAX_PROMPTS" \
      METHODS=baseline,shared_trace \
      SEED="$seed" \
      MAX_NEW_TOKENS=8192 \
      BRANCH_ABLATION_REASONING_PREFIX_TOKENS=4096 \
      BRANCH_ABLATION_GROUP_SIZES=32 \
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
  job_id="${job_id%%;*}"
  jobs+=("$job_id")
  echo "[mix-wide] repeat=${repeat_index} shard=${shard_label} job=${job_id}"
done < "$JOB_NAMES_FILE"

echo "[mix-wide] jobs=$(IFS=:; echo "${jobs[*]}")"
echo "[mix-wide] done"
