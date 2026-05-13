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
    echo "DRYRUN ${job_name} dependency=${dependency:-none} output=${output_dir} groups=${group_sizes}" >&2
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
  local job_name="aime4096-old-repair-aggregate"
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

echo "[repair] root_dir=${ROOT_DIR}"
echo "[repair] runner_script=${RUNNER_SCRIPT}"
echo "[repair] gpu_partition=${SLURM_GPU_PARTITION} gpu_time=${SLURM_GPU_TIME}"

LOW_RETRY_CONFIGS="16:1:2:aggressive;16:1:2:aggressive;16:1:2:aggressive;16:1:2:aggressive"
FIXED_RETRY_CONFIGS="32:2:4:aggressive;16:1:2:aggressive;16:1:2:aggressive"
TOPUP_RETRY_CONFIGS="32:2:4:aggressive;16:1:2:aggressive;16:1:2:aggressive"

r0_dir="${ROOT_DIR}/repeat_00"
r1_dir="${ROOT_DIR}/repeat_01"
r2_dir="${ROOT_DIR}/repeat_02"

r0_low="$(submit_gpu_job aime4096-old-r0-shared-low-repair "${r0_dir}/bundle_shared_low" "$r0_dir" 409600 36002 branch_ablation 2,4,8 1 "$LOW_RETRY_CONFIGS")"
r1_low="$(submit_gpu_job aime4096-old-r1-shared-low-repair "${r1_dir}/bundle_shared_low" "$r1_dir" 409601 36012 branch_ablation 2,4,8 1 "$LOW_RETRY_CONFIGS")"
r2_fixed="$(submit_gpu_job aime4096-old-r2-fixed-repair "${r2_dir}/bundle_fixed" "$r2_dir" 409602 36021 branch_ablation "" 0 "$FIXED_RETRY_CONFIGS")"
r2_low="$(submit_gpu_job aime4096-old-r2-shared-low-repair "${r2_dir}/bundle_shared_low" "$r2_dir" 409602 36022 branch_ablation 2,4,8 1 "$LOW_RETRY_CONFIGS")"

r0_topup_low="$(submit_gpu_job aime4096-old-r0-topup-low-repair "${r0_dir}/memory_match_topup_low" "$r0_dir" 409600 36005 memory_match_topup 2,4,8 1 "$TOPUP_RETRY_CONFIGS" 2,4,8 "afterok:${r0_low}")"
r0_topup_high="$(submit_gpu_job aime4096-old-r0-topup-high-repair "${r0_dir}/memory_match_topup_high" "$r0_dir" 409600 36006 memory_match_topup 16,32 1 "$TOPUP_RETRY_CONFIGS" 16,32 "afterok:${r0_low}")"

r1_topup_low="$(submit_gpu_job aime4096-old-r1-topup-low-repair "${r1_dir}/memory_match_topup_low" "$r1_dir" 409601 36015 memory_match_topup 2,4,8 1 "$TOPUP_RETRY_CONFIGS" 2,4,8 "afterok:${r1_low}")"
r1_topup_high="$(submit_gpu_job aime4096-old-r1-topup-high-repair "${r1_dir}/memory_match_topup_high" "$r1_dir" 409601 36016 memory_match_topup 16,32 1 "$TOPUP_RETRY_CONFIGS" 16,32 "afterok:${r1_low}")"

r2_dep="afterok:${r2_fixed}:${r2_low}"
r2_topup_low="$(submit_gpu_job aime4096-old-r2-topup-low-repair "${r2_dir}/memory_match_topup_low" "$r2_dir" 409602 36025 memory_match_topup 2,4,8 1 "$TOPUP_RETRY_CONFIGS" 2,4,8 "$r2_dep")"
r2_topup_high="$(submit_gpu_job aime4096-old-r2-topup-high-repair "${r2_dir}/memory_match_topup_high" "$r2_dir" 409602 36026 memory_match_topup 16,32 1 "$TOPUP_RETRY_CONFIGS" 16,32 "$r2_dep")"

aggregate_dep="afterok:${r0_topup_low}:${r0_topup_high}:${r1_topup_low}:${r1_topup_high}:${r2_topup_low}:${r2_topup_high}"
aggregate_job="$(submit_aggregate_job "$aggregate_dep")"

echo "[repair] r0_low=${r0_low} r0_topup_low=${r0_topup_low} r0_topup_high=${r0_topup_high}"
echo "[repair] r1_low=${r1_low} r1_topup_low=${r1_topup_low} r1_topup_high=${r1_topup_high}"
echo "[repair] r2_fixed=${r2_fixed} r2_low=${r2_low} r2_topup_low=${r2_topup_low} r2_topup_high=${r2_topup_high}"
echo "[repair] aggregate_job=${aggregate_job}"
echo "[repair] done"
