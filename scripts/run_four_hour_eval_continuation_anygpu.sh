#!/bin/bash
#SBATCH --job-name=mux4h-finish
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=/gscratch/scrubbed/suryadv/repos/Multiplex-Testing/slurm_logs/%x-%j.out
#SBATCH --chdir=/gscratch/scrubbed/suryadv/repos/Multiplex-Testing

set -euo pipefail

START_TIME="$(date +%s)"
REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

SCRATCH_ROOT="${SCRATCH_ROOT:-/gscratch/scrubbed/suryadv}"
SCRATCH_CACHE_ROOT="${SCRATCH_ROOT}/.cache/multiplex-thinking"
SCRATCH_TMP_ROOT="${SCRATCH_ROOT}/tmp/multiplex-thinking-four-hour-finish-${SLURM_JOB_ID:-local}"
SCRATCH_RUNTIME_ROOT="${SCRATCH_CACHE_ROOT}/runtime-envs"
JOB_OVERLAY_DIR="${SCRATCH_TMP_ROOT}/job-overlay"
JOB_BIN_DIR="${SCRATCH_TMP_ROOT}/bin"
SELECTED_RUNTIME_MANIFEST="${SCRATCH_TMP_ROOT}/runtime-manifest.json"
LOCAL_PYTHONPATH="${PWD}/verl-latest:${PWD}/sglang-0.4.9.post6:${PWD}/transformers-4.54.0/src"
ENV_PREFIX="/mmfs1/home/suryadv/.conda/envs/multiplex-thinking"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

mkdir -p slurm_logs "$SCRATCH_CACHE_ROOT" "$SCRATCH_TMP_ROOT" "$SCRATCH_RUNTIME_ROOT" "$JOB_OVERLAY_DIR" "$JOB_BIN_DIR"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate multiplex-thinking
elif [[ -x "$PYTHON_BIN" ]]; then
  export PATH="${ENV_PREFIX}/bin:${PATH}"
  export CONDA_DEFAULT_ENV="multiplex-thinking"
else
  echo "[setup] unable to find conda or ${PYTHON_BIN}" >&2
  exit 1
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

if command -v module >/dev/null 2>&1; then
  module load gcc/11.2.0 >/dev/null 2>&1 || true
  module load cuda/12.8.1 >/dev/null 2>&1 || true
fi

if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
  export CUDAHOSTCXX="$CXX"
  echo "[setup] gcc=$("$CC" --version | head -n 1)"
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  for candidate in /mmfs1/sw/cuda/12.8.1 /mmfs1/sw/cuda/12.4.1 /sw/cuda/12.8.1 /sw/cuda/12.4.1; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="$candidate"
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  export CUDA_PATH="$CUDA_HOME"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  echo "[setup] cuda_home=${CUDA_HOME}"
fi

export TMPDIR="$SCRATCH_TMP_ROOT"
export TMP="$SCRATCH_TMP_ROOT"
export TEMP="$SCRATCH_TMP_ROOT"
export RAY_TMPDIR="${RAY_TMPDIR:-${SCRATCH_ROOT}/ray/${SLURM_JOB_ID:-local}}"
export XDG_CACHE_HOME="${SCRATCH_CACHE_ROOT}/xdg"
export PIP_CACHE_DIR="${SCRATCH_CACHE_ROOT}/pip"
export HF_HOME="${SCRATCH_CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${SCRATCH_CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${SCRATCH_CACHE_ROOT}/triton"
export SGLANG_CACHE_ROOT="${SCRATCH_CACHE_ROOT}/sglang"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${SCRATCH_CACHE_ROOT}/flashinfer}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SCRATCH_CACHE_ROOT}/torch_extensions}"
export MPLCONFIGDIR="${SCRATCH_CACHE_ROOT}/matplotlib"
export CUDA_CACHE_PATH="${SCRATCH_CACHE_ROOT}/cuda"
export PYTHONPYCACHEPREFIX="${SCRATCH_CACHE_ROOT}/pycache"
export PYTHONPATH="${LOCAL_PYTHONPATH}"
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p \
  "$RAY_TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$SGLANG_CACHE_ROOT" \
  "$FLASHINFER_WORKSPACE_BASE" \
  "$TORCH_EXTENSIONS_DIR" \
  "$MPLCONFIGDIR" \
  "$CUDA_CACHE_PATH" \
  "$PYTHONPYCACHEPREFIX"

BOOTSTRAP_LOG="${SCRATCH_CACHE_ROOT}/bootstrap-runtime-four-hour-finish-${SLURM_JOB_ID:-local}.log"
BOOTSTRAP_LOCK="${SCRATCH_CACHE_ROOT}/bootstrap-runtime.lock"
echo "[setup] bootstrapping scratch-managed runtime"
(
  flock -x 200
  "$PYTHON_BIN" scripts/bootstrap_runtime_overlay.py \
    --python "$PYTHON_BIN" \
    --runtime-root "$SCRATCH_RUNTIME_ROOT" \
    --job-overlay-dir "$JOB_OVERLAY_DIR" \
    --job-bin-dir "$JOB_BIN_DIR" \
    --manifest-path "$SELECTED_RUNTIME_MANIFEST" \
    --log-file "$BOOTSTRAP_LOG" \
    --repair
) 200>"$BOOTSTRAP_LOCK"

export SELECTED_RUNTIME_MANIFEST
MANAGED_RUNTIME_SITE="$("$PYTHON_BIN" -c 'import json, os; print(json.load(open(os.environ["SELECTED_RUNTIME_MANIFEST"]))["runtime_site_packages"])')"
LIGHTWEIGHT_OVERLAY_CACHE_DIR="$("$PYTHON_BIN" -c 'import json, os; print(json.load(open(os.environ["SELECTED_RUNTIME_MANIFEST"])).get("lightweight_overlay_cache_dir", ""))')"
if [[ -n "$LIGHTWEIGHT_OVERLAY_CACHE_DIR" ]]; then
  export PYTHONPATH="${MANAGED_RUNTIME_SITE}:${JOB_OVERLAY_DIR}:${LIGHTWEIGHT_OVERLAY_CACHE_DIR}:${LOCAL_PYTHONPATH}"
  export PATH="${JOB_OVERLAY_DIR}/bin:${LIGHTWEIGHT_OVERLAY_CACHE_DIR}/bin:${JOB_BIN_DIR}:${PATH}"
else
  export PYTHONPATH="${MANAGED_RUNTIME_SITE}:${JOB_OVERLAY_DIR}:${LOCAL_PYTHONPATH}"
  export PATH="${JOB_OVERLAY_DIR}/bin:${JOB_BIN_DIR}:${PATH}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/four-hour-multiplex-scaling-35231035-20260514-090658}"
EVAL_ROOT="${OUTPUT_DIR}/eval"
REPEAT_LABEL="${REPEAT_LABEL:-repeat_02}"
REPEAT_DIR="${EVAL_ROOT}/${REPEAT_LABEL}"
MULTIPLEX_HF_CHECKPOINT="${MULTIPLEX_HF_CHECKPOINT:-${OUTPUT_DIR}/checkpoints/multiplex_thinking/global_step_48/actor/huggingface}"
SEED="${SEED:-26010810}"
EVAL_MAX_K="${EVAL_MAX_K:-8}"
EVAL_MAX_PROMPTS="${EVAL_MAX_PROMPTS:-30}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-5120}"
REASONING_PREFIX_TOKENS="${REASONING_PREFIX_TOKENS:-4096}"

if [[ ! -f "${MULTIPLEX_HF_CHECKPOINT}/config.json" ]]; then
  echo "[checkpoint] missing multiplex HF checkpoint: ${MULTIPLEX_HF_CHECKPOINT}" >&2
  exit 1
fi

common_eval_args() {
  local port="$1"
  echo \
    --model "$MULTIPLEX_HF_CHECKPOINT" \
    --benchmark aime2024 \
    --port "$port" \
    --dp-size 2 \
    --tp-size 1 \
    --resource-profile auto \
    --throughput-profile safe_auto \
    --request-batch-size 16 \
    --max-concurrent-prompts 2 \
    --prompts-per-rank 1 \
    --rank-scheduler dynamic \
    --server-timeout-seconds 900 \
    --max-k "$EVAL_MAX_K" \
    --max-prompts "$EVAL_MAX_PROMPTS" \
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
    --seed "$SEED" \
    --compact-jsonl \
    --resume
}

mkdir -p "$REPEAT_DIR"

echo "[eval] completing shared4 branch eval for ${REPEAT_LABEL}"
"$PYTHON_BIN" scripts/compare_passk_aime.py \
  --experiment-mode branch_ablation \
  --methods baseline,shared_trace \
  $(common_eval_args 37104) \
  --branch-ablation-reasoning-prefix-tokens "$REASONING_PREFIX_TOKENS" \
  --branch-ablation-group-sizes 4 \
  --output-dir "${REPEAT_DIR}/multiplex_branch_shared4"

if ! grep -q '^shared_trace_group_4,8,' "${REPEAT_DIR}/multiplex_branch_shared4/summary_ablation.csv"; then
  echo "[eval] shared4 branch eval did not produce pass@8 row" >&2
  exit 1
fi

echo "[eval] completing shared4 + shared2 top-up for ${REPEAT_LABEL}"
"$PYTHON_BIN" scripts/compare_passk_aime.py \
  --experiment-mode memory_match_topup \
  --methods baseline,shared_trace \
  $(common_eval_args 37105) \
  --branch-ablation-reasoning-prefix-tokens "$REASONING_PREFIX_TOKENS" \
  --memory-match-source-root "${REPEAT_DIR}/multiplex_branch_shared4" \
  --memory-match-shared-groups 4 \
  --memory-match-topup-generator shared_group \
  --memory-match-topup-shared-group-size 2 \
  --output-dir "${REPEAT_DIR}/multiplex_shared4_shared2_topup"

if [[ ! -f "${REPEAT_DIR}/multiplex_shared4_shared2_topup/summary_memory_match.md" ]]; then
  echo "[eval] missing top-up summary for ${REPEAT_LABEL}" >&2
  exit 1
fi

echo "[plots] writing final scaling artifacts"
"$PYTHON_BIN" scripts/plot_training_logs.py \
  --run "discrete_rl=${OUTPUT_DIR}/logs/discrete_rl_train.log" \
  --run "multiplex_thinking=${OUTPUT_DIR}/logs/multiplex_thinking_train.log" \
  --output-dir "${OUTPUT_DIR}/training_plots" \
  --summary-json "${OUTPUT_DIR}/training_plots/training_plot_summary.json" || true

"$PYTHON_BIN" scripts/aggregate_four_hour_scaling.py \
  --eval-root "$EVAL_ROOT" \
  --output-dir "${OUTPUT_DIR}/final_scaling_plots" \
  --max-k "$EVAL_MAX_K"

END_TIME="$(date +%s)"
SUMMARY_PATH="${OUTPUT_DIR}/finish_eval_summary.md"
{
  echo "# Four-Hour Eval Continuation"
  echo
  echo "- Slurm job: ${SLURM_JOB_ID:-local}"
  echo "- Wall time seconds: $((END_TIME - START_TIME))"
  echo "- Finished repeat: ${REPEAT_LABEL}"
  echo "- Shared4 summary: ${REPEAT_DIR}/multiplex_branch_shared4/summary_ablation.md"
  echo "- Shared4 + shared2 top-up summary: ${REPEAT_DIR}/multiplex_shared4_shared2_topup/summary_memory_match.md"
  echo "- Final scaling plots: ${OUTPUT_DIR}/final_scaling_plots"
} > "$SUMMARY_PATH"

echo "[done] summary=${SUMMARY_PATH}"
