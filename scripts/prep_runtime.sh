#!/bin/bash
#SBATCH --job-name=passk-runtime-prep
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40|l40s|h200
#SBATCH --output=/gscratch/scrubbed/suryadv/repos/Multiplex-Testing/slurm_logs/%x-%j.out
#SBATCH --chdir=/gscratch/scrubbed/suryadv/repos/Multiplex-Testing

set -euo pipefail

SCRATCH_ROOT="/gscratch/scrubbed/suryadv"
SCRATCH_CACHE_ROOT="${SCRATCH_ROOT}/.cache/multiplex-thinking"
SCRATCH_TMP_ROOT="${SCRATCH_ROOT}/tmp/multiplex-thinking-runtime-prep-${SLURM_JOB_ID:-local}"
SCRATCH_RUNTIME_ROOT="${SCRATCH_CACHE_ROOT}/runtime-envs"
JOB_OVERLAY_DIR="${SCRATCH_TMP_ROOT}/job-overlay"
JOB_BIN_DIR="${SCRATCH_TMP_ROOT}/bin"
SELECTED_RUNTIME_MANIFEST="${SCRATCH_TMP_ROOT}/runtime-manifest.json"
LOCAL_PYTHONPATH="${PWD}/sglang-0.4.9.post6:${PWD}/transformers-4.54.0/src"
ENV_PREFIX="/mmfs1/home/suryadv/.conda/envs/multiplex-thinking"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

mkdir -p slurm_logs "$SCRATCH_CACHE_ROOT" "$SCRATCH_TMP_ROOT" "$SCRATCH_RUNTIME_ROOT" "$JOB_OVERLAY_DIR" "$JOB_BIN_DIR"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate multiplex-thinking
elif [ -x "$PYTHON_BIN" ]; then
  export PATH="${ENV_PREFIX}/bin:${PATH}"
  export CONDA_DEFAULT_ENV="multiplex-thinking"
else
  echo "[prep] unable to find conda or ${PYTHON_BIN}" >&2
  exit 1
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

export TMPDIR="$SCRATCH_TMP_ROOT"
export TMP="$SCRATCH_TMP_ROOT"
export TEMP="$SCRATCH_TMP_ROOT"
export XDG_CACHE_HOME="${SCRATCH_CACHE_ROOT}/xdg"
export PIP_CACHE_DIR="${SCRATCH_CACHE_ROOT}/pip"
export HF_HOME="${SCRATCH_CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${SCRATCH_CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${SCRATCH_CACHE_ROOT}/triton"
export SGLANG_CACHE_ROOT="${SCRATCH_CACHE_ROOT}/sglang"
export FLASHINFER_WORKSPACE_BASE="${SCRATCH_CACHE_ROOT}/flashinfer"
export MPLCONFIGDIR="${SCRATCH_CACHE_ROOT}/matplotlib"
export CUDA_CACHE_PATH="${SCRATCH_CACHE_ROOT}/cuda"
export PYTHONPYCACHEPREFIX="${SCRATCH_CACHE_ROOT}/pycache"
export PYTHONPATH="${LOCAL_PYTHONPATH}"
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
export PYTHONNOUSERSITE=1

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$SGLANG_CACHE_ROOT" \
  "$FLASHINFER_WORKSPACE_BASE" \
  "$MPLCONFIGDIR" \
  "$CUDA_CACHE_PATH" \
  "$PYTHONPYCACHEPREFIX"

BOOTSTRAP_LOG="${SCRATCH_CACHE_ROOT}/bootstrap-runtime-prep-${SLURM_JOB_ID:-local}.log"
BOOTSTRAP_LOCK="${SCRATCH_CACHE_ROOT}/bootstrap-runtime.lock"
echo "[prep] bootstrapping scratch-managed runtime"
echo "[prep] bootstrap_log=${BOOTSTRAP_LOG}"
echo "[prep] bootstrap_lock=${BOOTSTRAP_LOCK}"
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

echo "[prep] runtime ready"
