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
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/final_eval_outputs/aime-train100-ablation4096-r3-${STAMP}}"
MAX_PROMPTS="${MAX_PROMPTS:-100}"
MAX_K="${MAX_K:-32}"
PROMPT_SHARDS="${PROMPT_SHARDS:-1}"
SEEDS=(409600 409601 409602)

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
SLURM_GPU_TIME="${SLURM_GPU_TIME:-24:00:00}"
SLURM_CPU_TIME="${SLURM_CPU_TIME:-02:00:00}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run_aime_job_with_fallback.sh}"

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

submit_gpu_job() {
  local job_name="$1"
  local output_dir="$2"
  local repeat_dir="$3"
  local seed="$4"
  local port="$5"
  local experiment_mode="$6"
  local group_sizes="$7"
  local no_baseline="$8"
  local fallback_configs="$9"
  local memory_match_groups="${10:-2,4,8,16,32}"
  local dependency="${11:-}"
  local prompt_indices="${12:-}"

  if [[ "$DRY_RUN" == "1" ]]; then
    {
      echo "[dry-run] sbatch ${job_name}"
      print_env_line EXPERIMENT_MODE "$experiment_mode"
      print_env_line BENCHMARK "deepscaler_aime_train"
      print_env_line OUTPUT_DIR "$output_dir"
      print_env_line MEMORY_MATCH_SOURCE_ROOT "$repeat_dir"
      print_env_line SEED "$seed"
      print_env_line PORT "$port"
      print_env_line MAX_K "$MAX_K"
      print_env_line MAX_PROMPTS "$MAX_PROMPTS"
      print_env_line BRANCH_ABLATION_GROUP_SIZES "$group_sizes"
      print_env_line BRANCH_ABLATION_NO_BASELINE "$no_baseline"
      print_env_line MEMORY_MATCH_SHARED_GROUPS "$memory_match_groups"
      print_env_line PROMPT_INDICES "${prompt_indices:-all}"
      print_env_line FALLBACK_CONFIGS "$fallback_configs"
      print_env_line SERVER_TIMEOUT_SECONDS "900"
      print_env_line DEPENDENCY "${dependency:-none}"
    } >&2
    echo "DRYRUN-${job_name}"
    return
  fi

  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="$dependency")
  fi

  local job_id
  if ! job_id="$(
    env \
      BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
      BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
      MPLCONFIGDIR=/tmp \
      EXPERIMENT_MODE="$experiment_mode" \
      BENCHMARK=deepscaler_aime_train \
      RUN_TAG="$job_name" \
      OUTPUT_DIR="$output_dir" \
      MEMORY_MATCH_SOURCE_ROOT="$repeat_dir" \
      MEMORY_MATCH_SHARED_GROUPS="$memory_match_groups" \
      PROMPT_INDICES="$prompt_indices" \
      PORT="$port" \
      MAX_K="$MAX_K" \
      MAX_PROMPTS="$MAX_PROMPTS" \
      METHODS=baseline,shared_trace \
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
        "${dependency_args[@]}" \
        "${SBATCH_GPU_COMMON[@]}" \
        "$RUNNER_SCRIPT"
  )"; then
    echo "[submit] failed to submit ${job_name}" >&2
    exit 1
  fi
  echo "${job_id%%;*}"
}

prompt_indices_for_shard() {
  local shard_index="$1"
  local start=$((shard_index * MAX_PROMPTS / PROMPT_SHARDS))
  local end=$(((shard_index + 1) * MAX_PROMPTS / PROMPT_SHARDS))
  local indices=""
  local idx
  for ((idx = start; idx < end; idx++)); do
    if [[ -n "$indices" ]]; then
      indices+=","
    fi
    indices+="$idx"
  done
  echo "$indices"
}

submit_generation_bundle() {
  local job_name="$1"
  local output_dir="$2"
  local repeat_dir="$3"
  local seed="$4"
  local port="$5"
  local group_sizes="$6"
  local no_baseline="$7"
  local fallback_configs="$8"
  local job_ids=()

  if [[ "$PROMPT_SHARDS" -le 1 ]]; then
    job_ids+=("$(
      submit_gpu_job \
        "$job_name" \
        "$output_dir" \
        "$repeat_dir" \
        "$seed" \
        "$port" \
        branch_ablation \
        "$group_sizes" \
        "$no_baseline" \
        "$fallback_configs"
    )")
  else
    local shard_index shard_label shard_output shard_job shard_prompt_indices shard_port
    for ((shard_index = 0; shard_index < PROMPT_SHARDS; shard_index++)); do
      shard_label="$(printf 'shard%02d' "$shard_index")"
      shard_output="${output_dir}_${shard_label}"
      shard_job="${job_name}-${shard_label}"
      shard_prompt_indices="$(prompt_indices_for_shard "$shard_index")"
      shard_port=$((port + shard_index * 100))
      job_ids+=("$(
        submit_gpu_job \
          "$shard_job" \
          "$shard_output" \
          "$repeat_dir" \
          "$seed" \
          "$shard_port" \
          branch_ablation \
          "$group_sizes" \
          "$no_baseline" \
          "$fallback_configs" \
          "2,4,8,16,32" \
          "" \
          "$shard_prompt_indices"
      )")
    done
  fi

  local joined
  joined="$(IFS=:; echo "${job_ids[*]}")"
  echo "$joined"
}

submit_aggregate_job() {
  local dependency="$1"
  local job_name="aime4096-aggregate-${STAMP}"
  local command="cd '$REPO_ROOT' && MPLCONFIGDIR=/tmp /mmfs1/home/suryadv/.conda/envs/multiplex-thinking/bin/python scripts/aggregate_aime_train100_ablation4096.py --root '$ROOT_DIR' --repeats 3 --seeds 409600,409601,409602"

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
echo "[submit] layout: 3 repeats x 4 generation bundles x ${PROMPT_SHARDS} prompt shard(s) + 6 split memory-match top-up jobs + 1 aggregate job"

topup_job_ids=()
for repeat_index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$repeat_index]}"
  repeat_dir="${ROOT_DIR}/repeat_$(printf '%02d' "$repeat_index")"
  bundle_fixed="${repeat_dir}/bundle_fixed"
  bundle_shared_low="${repeat_dir}/bundle_shared_low"
  bundle_shared_16="${repeat_dir}/bundle_shared_16"
  bundle_shared_32="${repeat_dir}/bundle_shared_32"
  topup_low_dir="${repeat_dir}/memory_match_topup_low"
  topup_high_dir="${repeat_dir}/memory_match_topup_high"
  port_base=$((31000 + repeat_index * 10))

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$repeat_dir"
  fi

  job_fixed="$(
    submit_generation_bundle \
      "aime4096-r${repeat_index}-fixed" \
      "$bundle_fixed" \
      "$repeat_dir" \
      "$seed" \
      "$((port_base + 1))" \
      "" \
      0 \
      "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive"
  )"
  job_shared_low="$(
    submit_generation_bundle \
      "aime4096-r${repeat_index}-shared-low" \
      "$bundle_shared_low" \
      "$repeat_dir" \
      "$seed" \
      "$((port_base + 2))" \
      2,4,8 \
      1 \
      "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive"
  )"
  job_shared_16="$(
    submit_generation_bundle \
      "aime4096-r${repeat_index}-shared16" \
      "$bundle_shared_16" \
      "$repeat_dir" \
      "$seed" \
      "$((port_base + 3))" \
      16 \
      1 \
      "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive"
  )"
  job_shared_32="$(
    submit_generation_bundle \
      "aime4096-r${repeat_index}-shared32" \
      "$bundle_shared_32" \
      "$repeat_dir" \
      "$seed" \
      "$((port_base + 4))" \
      32 \
      1 \
      "48:3:6:aggressive;32:2:4:aggressive;24:2:4:aggressive;16:1:2:aggressive"
  )"
  dependency="afterok:${job_fixed}:${job_shared_low}:${job_shared_16}:${job_shared_32}"
  topup_low_job="$(
    submit_gpu_job \
      "aime4096-r${repeat_index}-topup-low" \
      "$topup_low_dir" \
      "$repeat_dir" \
      "$seed" \
      "$((port_base + 5))" \
      memory_match_topup \
      2,4,8 \
      1 \
      "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive" \
      2,4,8 \
      "$dependency"
  )"
  topup_high_job="$(
    submit_gpu_job \
      "aime4096-r${repeat_index}-topup-high" \
      "$topup_high_dir" \
      "$repeat_dir" \
      "$seed" \
      "$((port_base + 6))" \
      memory_match_topup \
      16,32 \
      1 \
      "48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive" \
      16,32 \
      "$dependency"
  )"
  topup_job_ids+=("$topup_low_job" "$topup_high_job")
  echo "[submit] repeat=${repeat_index} seed=${seed} fixed=${job_fixed} shared_low=${job_shared_low} shared16=${job_shared_16} shared32=${job_shared_32} topup_low=${topup_low_job} topup_high=${topup_high_job}"
done

aggregate_dependency="afterok:$(IFS=:; echo "${topup_job_ids[*]}")"
aggregate_job="$(submit_aggregate_job "$aggregate_dependency")"

echo "[submit] aggregate_job=${aggregate_job}"
echo "[submit] done"
