#!/bin/bash
set -euo pipefail

REPO_ROOT="/gscratch/scrubbed/suryadv/repos/Multiplex-Testing"
cd "$REPO_ROOT"

STAMP="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/final_eval_outputs/passk-ablation32-raivn-${STAMP}}"
REASONING_BUDGETS="${REASONING_BUDGETS:-1024 2048 4096}"
GROUP_SIZES="${GROUP_SIZES:-2,4,8,16,32}"
MAX_PROMPTS="${MAX_PROMPTS:-30}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_PLOT_GPU_GRES="${SLURM_PLOT_GPU_GRES:-gpu:1}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"

mkdir -p "$ROOT_DIR"

SBATCH_COMMON=(
  --account=raivn-ckpt
  --partition=ckpt-all
  --cpus-per-task=4
  --export=ALL
)

echo "[submit] root_dir=${ROOT_DIR}"
echo "[submit] reasoning_budgets=${REASONING_BUDGETS}"
echo "[submit] group_sizes=${GROUP_SIZES}"
echo "[submit] max_k=32 max_prompts=${MAX_PROMPTS}"

job_ids=()
for budget in ${REASONING_BUDGETS}; do
  out_dir="${ROOT_DIR}/budget_${budget}"
  mkdir -p "$out_dir"
  job_id="$(
    env \
      BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
      BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
      MPLCONFIGDIR=/tmp \
      EXPERIMENT_MODE=branch_ablation \
      RUN_TAG="raivn-ablation32-t${budget}-${STAMP}" \
      OUTPUT_DIR="$out_dir" \
      MAX_K=32 \
      MAX_PROMPTS="${MAX_PROMPTS}" \
      METHODS=baseline,shared_trace \
      BRANCH_ABLATION_REASONING_PREFIX_TOKENS="${budget}" \
      BRANCH_ABLATION_GROUP_SIZES="${GROUP_SIZES}" \
      sbatch --parsable \
        --job-name="ab32-t${budget}" \
        --gres="$SLURM_GPU_GRES" \
        --constraint="$SLURM_GPU_CONSTRAINT" \
        --time=96:00:00 \
        "${SBATCH_COMMON[@]}" \
        "${REPO_ROOT}/run.sh"
  )"
  job_id="${job_id%%;*}"
  job_ids+=("$job_id")
  echo "[submit] budget_${budget}: ${job_id}"
done

dependency="afterok:$(IFS=:; echo "${job_ids[*]}")"
plot_job_id="$(
  sbatch --parsable \
    --job-name=ab32-plots \
    --gres="$SLURM_PLOT_GPU_GRES" \
    --constraint="$SLURM_GPU_CONSTRAINT" \
    --time=01:00:00 \
    --dependency="$dependency" \
    "${SBATCH_COMMON[@]}" \
    --wrap="cd '${REPO_ROOT}' && MPLCONFIGDIR=/tmp /mmfs1/home/suryadv/.conda/envs/multiplex-thinking/bin/python scripts/plot_branch_ablation_budget_sweep.py --root '${ROOT_DIR}'"
)"
plot_job_id="${plot_job_id%%;*}"
echo "[submit] aggregate_plots: ${plot_job_id} (${dependency})"
echo "[submit] done"
