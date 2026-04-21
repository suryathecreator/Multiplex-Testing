#!/bin/bash
#SBATCH --job-name=run-gpu2
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=/gscratch/scrubbed/suryadv/repos/Multiplex-Testing/slurm_logs/%x-%j.out
#SBATCH --chdir=/gscratch/scrubbed/suryadv/repos/Multiplex-Testing

set -euo pipefail

SCRATCH_ROOT="/gscratch/scrubbed/suryadv"
SCRATCH_CACHE_ROOT="${SCRATCH_ROOT}/.cache/multiplex-thinking"
SCRATCH_TMP_ROOT="${SCRATCH_ROOT}/tmp/multiplex-thinking-${SLURM_JOB_ID:-local}"
SCRATCH_RUNTIME_ROOT="${SCRATCH_CACHE_ROOT}/runtime-envs"
JOB_OVERLAY_DIR="${SCRATCH_TMP_ROOT}/job-overlay"
JOB_BIN_DIR="${SCRATCH_TMP_ROOT}/bin"
SELECTED_RUNTIME_MANIFEST="${SCRATCH_TMP_ROOT}/runtime-manifest.json"
LOCAL_PYTHONPATH="${PWD}/sglang-0.4.9.post6:${PWD}/transformers-4.54.0/src"

mkdir -p slurm_logs
mkdir -p final_eval_outputs
mkdir -p "$SCRATCH_CACHE_ROOT" "$SCRATCH_TMP_ROOT" "$SCRATCH_RUNTIME_ROOT" "$JOB_OVERLAY_DIR" "$JOB_BIN_DIR"

ENV_PREFIX="/mmfs1/home/suryadv/.conda/envs/multiplex-thinking"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate multiplex-thinking
elif [ -x "$PYTHON_BIN" ]; then
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

unset NCCL_SOCKET_IFNAME || true
unset NCCL_IB_HCA || true

BOOTSTRAP_LOG="${SCRATCH_CACHE_ROOT}/bootstrap-runtime-${SLURM_JOB_ID:-local}.log"
echo "[setup] bootstrapping scratch-managed runtime"
echo "[setup] bootstrap_log=${BOOTSTRAP_LOG}"
"$PYTHON_BIN" scripts/bootstrap_runtime_overlay.py \
  --python "$PYTHON_BIN" \
  --runtime-root "$SCRATCH_RUNTIME_ROOT" \
  --job-overlay-dir "$JOB_OVERLAY_DIR" \
  --job-bin-dir "$JOB_BIN_DIR" \
  --manifest-path "$SELECTED_RUNTIME_MANIFEST" \
  --log-file "$BOOTSTRAP_LOG" \
  --repair

export SELECTED_RUNTIME_MANIFEST
MANAGED_RUNTIME_SITE="$("$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["SELECTED_RUNTIME_MANIFEST"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(manifest["runtime_site_packages"])
PY
)"
MANAGED_RUNTIME_ROOT="$("$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["SELECTED_RUNTIME_MANIFEST"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(manifest["runtime_root"])
PY
)"
LIGHTWEIGHT_OVERLAY_CACHE_DIR="$("$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["SELECTED_RUNTIME_MANIFEST"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(manifest.get("lightweight_overlay_cache_dir", ""))
PY
)"
if [[ -n "$LIGHTWEIGHT_OVERLAY_CACHE_DIR" ]]; then
  export PYTHONPATH="${MANAGED_RUNTIME_SITE}:${JOB_OVERLAY_DIR}:${LIGHTWEIGHT_OVERLAY_CACHE_DIR}:${LOCAL_PYTHONPATH}"
  export PATH="${JOB_OVERLAY_DIR}/bin:${LIGHTWEIGHT_OVERLAY_CACHE_DIR}/bin:${JOB_BIN_DIR}:${PATH}"
else
  export PYTHONPATH="${MANAGED_RUNTIME_SITE}:${JOB_OVERLAY_DIR}:${LOCAL_PYTHONPATH}"
  export PATH="${JOB_OVERLAY_DIR}/bin:${JOB_BIN_DIR}:${PATH}"
fi

echo "[setup] runtime_root=${MANAGED_RUNTIME_ROOT}"
echo "[setup] runtime_manifest=${SELECTED_RUNTIME_MANIFEST}"
echo "[setup] runtime_site=${MANAGED_RUNTIME_SITE}"
echo "[setup] job_overlay_dir=${JOB_OVERLAY_DIR}"
echo "[setup] job_bin_dir=${JOB_BIN_DIR}"
echo "[setup] lightweight_overlay_cache_dir=${LIGHTWEIGHT_OVERLAY_CACHE_DIR:-unset}"
echo "[setup] flashinfer_workspace_base=${FLASHINFER_WORKSPACE_BASE}"
echo "[setup] job_overlay_bin=${JOB_OVERLAY_DIR}/bin"
echo "[setup] lightweight_overlay_bin=${LIGHTWEIGHT_OVERLAY_CACHE_DIR:-unset}/bin"
echo "[setup] ninja_path=$(command -v ninja || echo missing)"

"$PYTHON_BIN" - <<'PY'
import importlib
import importlib.util

required_targets = ["torch", "tensordict", "sglang", "transformers"]
optional_targets = ["verl"]

for target in required_targets:
    module = importlib.import_module(target)
    version = getattr(module, "__version__", "n/a")
    print(f"[setup] import={target} version={version} file={getattr(module, '__file__', None)}")

for target in optional_targets:
    spec = importlib.util.find_spec(target)
    if spec is None:
        print(f"[setup] optional_import={target} status=missing")
        continue
    origin = getattr(spec, "origin", None)
    print(f"[setup] optional_import={target} status=available origin={origin}")
PY

RUN_TAG="${RUN_TAG:-}"
if [[ -n "$RUN_TAG" ]]; then
  RUN_STAMP="$RUN_TAG"
else
  RUN_STAMP="${SLURM_JOB_ID:-local}-$(date +%Y%m%d-%H%M%S)"
fi
OUTPUT_DIR="/gscratch/scrubbed/suryadv/repos/Multiplex-Testing/final_eval_outputs/aime-shared-trace-passk-${RUN_STAMP}"
mkdir -p "$OUTPUT_DIR"

DP_SIZE="${DP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
MAX_K="${MAX_K:-16}"
MAX_PROMPTS="${MAX_PROMPTS:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
REASONING_PREFIX_TOKEN_VALUES="${REASONING_PREFIX_TOKEN_VALUES:-512,1024,2048}"
CHECKPOINT_MATCHED_PROMPTS_STEP="${CHECKPOINT_MATCHED_PROMPTS_STEP:-5}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-}"
RESOURCE_PROFILE="${RESOURCE_PROFILE:-auto}"
REQUEST_BATCH_SIZE_DISPLAY="${REQUEST_BATCH_SIZE:-auto}"

echo "[setup] scratch_cache_root=${SCRATCH_CACHE_ROOT}"
echo "[setup] output_dir=${OUTPUT_DIR}"
echo "[setup] run_tag=${RUN_STAMP}"
echo "[setup] slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
echo "[setup] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[setup] requested_launch_config dp_size=${DP_SIZE} tp_size=${TP_SIZE} max_k=${MAX_K} max_prompts=${MAX_PROMPTS} max_new_tokens=${MAX_NEW_TOKENS} reasoning_prefix_token_values=${REASONING_PREFIX_TOKEN_VALUES} checkpoint_matched_prompts_step=${CHECKPOINT_MATCHED_PROMPTS_STEP} request_batch_size=${REQUEST_BATCH_SIZE_DISPLAY} resource_profile=${RESOURCE_PROFILE}"
echo "[setup] starting compare_passk_aime.py"

COMPARE_ARGS=(
  --model Qwen/Qwen3-4B
  --benchmark aime2024
  --max-k "$MAX_K"
  --max-prompts "$MAX_PROMPTS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --reasoning-prefix-token-values "$REASONING_PREFIX_TOKEN_VALUES"
  --checkpoint-matched-prompts-step "$CHECKPOINT_MATCHED_PROMPTS_STEP"
  --methods baseline,shared_trace,standard_generation
  --seed 1234
  --output-dir "$OUTPUT_DIR"
  --dp-size "$DP_SIZE"
  --tp-size "$TP_SIZE"
  --resource-profile "$RESOURCE_PROFILE"
  --resume
)

if [[ -n "$REQUEST_BATCH_SIZE" ]]; then
  COMPARE_ARGS+=(--request-batch-size "$REQUEST_BATCH_SIZE")
fi

"$PYTHON_BIN" scripts/compare_passk_aime.py \
  "${COMPARE_ARGS[@]}"

echo "[setup] final summary: ${OUTPUT_DIR}/summary_overall.md"
