#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/final_eval_outputs/aime-train100-ablation4096-r3-stable3b-20260501-103533}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run_cuda_graph_smoke.sh}"
MAX_PROMPTS="${MAX_PROMPTS:-100}"
MAX_K="${MAX_K:-32}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-raivn-ckpt}"
SLURM_GPU_PARTITION="${SLURM_GPU_PARTITION:-ckpt-all}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"
SLURM_GPU_TIME="${SLURM_GPU_TIME:-24:00:00}"
SLURM_CPU_PARTITION="${SLURM_CPU_PARTITION:-ckpt-all}"
SLURM_CPU_TIME="${SLURM_CPU_TIME:-02:00:00}"

# These two jobs were already running on standard RAIVN when the migration began.
R0_FIXED_JOB="${R0_FIXED_JOB:-34986925}"
R0_SHARED_LOW_JOB="${R0_SHARED_LOW_JOB:-34986926}"

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

mkdir -p slurm_logs "$ROOT_DIR"

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

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRYRUN ${job_name} dependency=${dependency:-none} output=${output_dir}" >&2
    echo "DRYRUN-${job_name}"
    return
  fi

  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="$dependency")
  fi

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
      MEMORY_MATCH_SOURCE_ROOT="$repeat_dir" \
      MEMORY_MATCH_SHARED_GROUPS="$memory_match_groups" \
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
  )"
  echo "${job_id%%;*}"
}

submit_aggregate_job() {
  local dependency="$1"
  local job_name="aime4096-aggregate-ckpt-migrated"
  local command="cd '$REPO_ROOT' && MPLCONFIGDIR=/tmp /mmfs1/home/suryadv/.conda/envs/multiplex-thinking/bin/python scripts/aggregate_aime_train100_ablation4096.py --root '$ROOT_DIR' --repeats 3 --seeds 409600,409601,409602"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRYRUN ${job_name} dependency=${dependency}" >&2
    echo "DRYRUN-${job_name}"
    return
  fi

  local job_id
  job_id="$(
    sbatch --parsable \
      --job-name="$job_name" \
      --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
      --dependency="$dependency" \
      "${SBATCH_CPU_COMMON[@]}" \
      --wrap "$command"
  )"
  echo "${job_id%%;*}"
}

echo "[migrate] root_dir=${ROOT_DIR}"
echo "[migrate] runner_script=${RUNNER_SCRIPT}"
echo "[migrate] keeping standard RAIVN jobs: r0_fixed=${R0_FIXED_JOB} r0_shared_low=${R0_SHARED_LOW_JOB}"

topup_job_ids=()

repeat_dir="${ROOT_DIR}/repeat_00"
r0_shared16="$(
  submit_gpu_job aime4096-r0-shared16-ckpt "${repeat_dir}/bundle_shared_16" "$repeat_dir" 409600 31003 branch_ablation 16 1 \
    "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive"
)"
r0_shared32="$(
  submit_gpu_job aime4096-r0-shared32-ckpt "${repeat_dir}/bundle_shared_32" "$repeat_dir" 409600 31004 branch_ablation 32 1 \
    "48:3:6:aggressive;32:2:4:aggressive;24:2:4:aggressive;16:1:2:aggressive"
)"
r0_dep="afterok:${R0_FIXED_JOB}:${R0_SHARED_LOW_JOB}:${r0_shared16}:${r0_shared32}"
r0_topup_low="$(submit_gpu_job aime4096-r0-topup-low-ckpt "${repeat_dir}/memory_match_topup_low" "$repeat_dir" 409600 31005 memory_match_topup 2,4,8 1 "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive" 2,4,8 "$r0_dep")"
r0_topup_high="$(submit_gpu_job aime4096-r0-topup-high-ckpt "${repeat_dir}/memory_match_topup_high" "$repeat_dir" 409600 31006 memory_match_topup 16,32 1 "48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive" 16,32 "$r0_dep")"
topup_job_ids+=("$r0_topup_low" "$r0_topup_high")
echo "[migrate] repeat=0 shared16=${r0_shared16} shared32=${r0_shared32} topup_low=${r0_topup_low} topup_high=${r0_topup_high}"

for repeat_index in 1 2; do
  seed=$((409600 + repeat_index))
  repeat_dir="${ROOT_DIR}/repeat_$(printf '%02d' "$repeat_index")"
  port_base=$((31000 + repeat_index * 10))

  fixed="$(submit_gpu_job "aime4096-r${repeat_index}-fixed-ckpt" "${repeat_dir}/bundle_fixed" "$repeat_dir" "$seed" "$((port_base + 1))" branch_ablation "" 0 "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive")"
  shared_low="$(submit_gpu_job "aime4096-r${repeat_index}-shared-low-ckpt" "${repeat_dir}/bundle_shared_low" "$repeat_dir" "$seed" "$((port_base + 2))" branch_ablation 2,4,8 1 "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive")"
  shared16="$(submit_gpu_job "aime4096-r${repeat_index}-shared16-ckpt" "${repeat_dir}/bundle_shared_16" "$repeat_dir" "$seed" "$((port_base + 3))" branch_ablation 16 1 "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive")"
  shared32="$(submit_gpu_job "aime4096-r${repeat_index}-shared32-ckpt" "${repeat_dir}/bundle_shared_32" "$repeat_dir" "$seed" "$((port_base + 4))" branch_ablation 32 1 "48:3:6:aggressive;32:2:4:aggressive;24:2:4:aggressive;16:1:2:aggressive")"
  dep="afterok:${fixed}:${shared_low}:${shared16}:${shared32}"
  topup_low="$(submit_gpu_job "aime4096-r${repeat_index}-topup-low-ckpt" "${repeat_dir}/memory_match_topup_low" "$repeat_dir" "$seed" "$((port_base + 5))" memory_match_topup 2,4,8 1 "64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive" 2,4,8 "$dep")"
  topup_high="$(submit_gpu_job "aime4096-r${repeat_index}-topup-high-ckpt" "${repeat_dir}/memory_match_topup_high" "$repeat_dir" "$seed" "$((port_base + 6))" memory_match_topup 16,32 1 "48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive" 16,32 "$dep")"
  topup_job_ids+=("$topup_low" "$topup_high")
  echo "[migrate] repeat=${repeat_index} fixed=${fixed} shared_low=${shared_low} shared16=${shared16} shared32=${shared32} topup_low=${topup_low} topup_high=${topup_high}"
done

aggregate_dependency="afterok:$(IFS=:; echo "${topup_job_ids[*]}")"
aggregate_job="$(submit_aggregate_job "$aggregate_dependency")"
echo "[migrate] aggregate_job=${aggregate_job}"
echo "[migrate] done"
