#!/bin/bash
#SBATCH --job-name=run-gpu2
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=a40|a100|l40|l40s|h200
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
  "$TORCH_EXTENSIONS_DIR" \
  "$MPLCONFIGDIR" \
  "$CUDA_CACHE_PATH" \
  "$PYTHONPYCACHEPREFIX"

unset NCCL_SOCKET_IFNAME || true
unset NCCL_IB_HCA || true

BOOTSTRAP_LOG="${SCRATCH_CACHE_ROOT}/bootstrap-runtime-${SLURM_JOB_ID:-local}.log"
BOOTSTRAP_LOCK="${SCRATCH_CACHE_ROOT}/bootstrap-runtime.lock"
echo "[setup] bootstrapping scratch-managed runtime"
echo "[setup] bootstrap_log=${BOOTSTRAP_LOG}"
echo "[setup] bootstrap_lock=${BOOTSTRAP_LOCK}"
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
echo "[setup] torch_extensions_dir=${TORCH_EXTENSIONS_DIR}"
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
OUTPUT_DIR="${OUTPUT_DIR:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing/final_eval_outputs/aime-shared-trace-passk-${RUN_STAMP}}"
mkdir -p "$OUTPUT_DIR"

EXPERIMENT_MODE="${EXPERIMENT_MODE:-passk_sweep}"
BENCHMARK="${BENCHMARK:-aime2024}"
PORT="${PORT:-30000}"
DP_SIZE="${DP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
MAX_K="${MAX_K:-64}"
MAX_PROMPTS="${MAX_PROMPTS:-50}"
SEED="${SEED:-1234}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
REASONING_PREFIX_TOKEN_VALUES="${REASONING_PREFIX_TOKEN_VALUES:-256,512,1024,2048,4096,6144}"
BRANCH_ABLATION_REASONING_PREFIX_TOKENS="${BRANCH_ABLATION_REASONING_PREFIX_TOKENS:-1024}"
BRANCH_ABLATION_GROUP_SIZES="${BRANCH_ABLATION_GROUP_SIZES-2,4,8,16}"
BRANCH_ABLATION_NO_BASELINE="${BRANCH_ABLATION_NO_BASELINE:-0}"
ADAPTIVE_ABLATION_SHARED_COUNTS="${ADAPTIVE_ABLATION_SHARED_COUNTS:-2,4,6,8,10,12,14,16}"
ADAPTIVE_CONFIDENCE_THRESHOLD="${ADAPTIVE_CONFIDENCE_THRESHOLD:-0.75}"
MEMORY_MATCH_SOURCE_ROOT="${MEMORY_MATCH_SOURCE_ROOT:-}"
MEMORY_MATCH_SHARED_GROUPS="${MEMORY_MATCH_SHARED_GROUPS:-2,4,8,16,32}"
MEMORY_MATCH_TOPUP_SHARED_GROUP_SIZE="${MEMORY_MATCH_TOPUP_SHARED_GROUP_SIZE:-0}"
MEMORY_MATCH_CHECKPOINT_BEFORE_LABEL="${MEMORY_MATCH_CHECKPOINT_BEFORE_LABEL:-}"
MEMORY_MATCH_CHECKPOINT_AFTER_LABEL="${MEMORY_MATCH_CHECKPOINT_AFTER_LABEL:-}"
MEMORY_MATCH_CHECKPOINT_TITLE="${MEMORY_MATCH_CHECKPOINT_TITLE:-}"
HYPERPARAM_REASONING_PREFIX_TOKENS="${HYPERPARAM_REASONING_PREFIX_TOKENS:-1024}"
HYPERPARAM_TOP_P_VALUES="${HYPERPARAM_TOP_P_VALUES:-0.75,0.85,0.90,0.95,1.00}"
HYPERPARAM_TEMPERATURE_VALUES="${HYPERPARAM_TEMPERATURE_VALUES:-0.4,0.6,0.8,1.0,1.2}"
CHECKPOINT_MATCHED_PROMPTS_STEP="${CHECKPOINT_MATCHED_PROMPTS_STEP:-5}"
COMPACT_JSONL="${COMPACT_JSONL:-0}"
METHODS="${METHODS:-baseline,shared_trace}"
PROMPT_INDICES="${PROMPT_INDICES:-}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-}"
RESOURCE_PROFILE="${RESOURCE_PROFILE:-auto}"
REQUEST_BATCH_SIZE_DISPLAY="${REQUEST_BATCH_SIZE:-auto}"
THROUGHPUT_PROFILE="${THROUGHPUT_PROFILE:-safe_auto}"
MAX_CONCURRENT_PROMPTS="${MAX_CONCURRENT_PROMPTS:-}"
PROMPTS_PER_RANK="${PROMPTS_PER_RANK:-1}"
RANK_SCHEDULER="${RANK_SCHEDULER:-dynamic}"
MAX_CONCURRENT_PROMPTS_DISPLAY="${MAX_CONCURRENT_PROMPTS:-auto}"
SERVER_TIMEOUT_SECONDS="${SERVER_TIMEOUT_SECONDS:-900}"

resolve_local_qwen3_model() {
  local local_model_dir="${SCRATCH_CACHE_ROOT}/local-models/Qwen3-4B"
  local metadata_snapshot="${TRANSFORMERS_CACHE}/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
  local weights_snapshot="${HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

  if [[ ! -f "${metadata_snapshot}/config.json" || ! -f "${weights_snapshot}/model.safetensors.index.json" ]]; then
    return 1
  fi

  mkdir -p "$local_model_dir"
  local snapshot entry resolved
  for snapshot in "$metadata_snapshot" "$weights_snapshot"; do
    for entry in "${snapshot}"/*; do
      if [[ ! -e "$entry" && ! -L "$entry" ]]; then
        continue
      fi
      resolved="$(readlink -f "$entry")"
      ln -sfn "$resolved" "${local_model_dir}/$(basename "$entry")"
    done
  done
  echo "$local_model_dir"
}

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  if MODEL_PATH="$(resolve_local_qwen3_model)"; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
  else
    MODEL_PATH="Qwen/Qwen3-4B"
  fi
fi

echo "[setup] scratch_cache_root=${SCRATCH_CACHE_ROOT}"
echo "[setup] output_dir=${OUTPUT_DIR}"
echo "[setup] run_tag=${RUN_STAMP}"
echo "[setup] slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
echo "[setup] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[setup] model_path=${MODEL_PATH}"
echo "[setup] requested_launch_config experiment_mode=${EXPERIMENT_MODE} benchmark=${BENCHMARK} methods=${METHODS} seed=${SEED} port=${PORT} dp_size=${DP_SIZE} tp_size=${TP_SIZE} max_k=${MAX_K} max_prompts=${MAX_PROMPTS} prompt_indices=${PROMPT_INDICES:-all} max_new_tokens=${MAX_NEW_TOKENS} reasoning_prefix_token_values=${REASONING_PREFIX_TOKEN_VALUES} branch_ablation_reasoning_prefix_tokens=${BRANCH_ABLATION_REASONING_PREFIX_TOKENS} branch_ablation_group_sizes=${BRANCH_ABLATION_GROUP_SIZES} branch_ablation_no_baseline=${BRANCH_ABLATION_NO_BASELINE} adaptive_ablation_shared_counts=${ADAPTIVE_ABLATION_SHARED_COUNTS} adaptive_confidence_threshold=${ADAPTIVE_CONFIDENCE_THRESHOLD} memory_match_source_root=${MEMORY_MATCH_SOURCE_ROOT:-unset} memory_match_shared_groups=${MEMORY_MATCH_SHARED_GROUPS} hyperparam_reasoning_prefix_tokens=${HYPERPARAM_REASONING_PREFIX_TOKENS} hyperparam_top_p_values=${HYPERPARAM_TOP_P_VALUES} hyperparam_temperature_values=${HYPERPARAM_TEMPERATURE_VALUES} checkpoint_matched_prompts_step=${CHECKPOINT_MATCHED_PROMPTS_STEP} compact_jsonl=${COMPACT_JSONL} request_batch_size=${REQUEST_BATCH_SIZE_DISPLAY} resource_profile=${RESOURCE_PROFILE} throughput_profile=${THROUGHPUT_PROFILE} max_concurrent_prompts=${MAX_CONCURRENT_PROMPTS_DISPLAY} prompts_per_rank=${PROMPTS_PER_RANK} rank_scheduler=${RANK_SCHEDULER} server_timeout_seconds=${SERVER_TIMEOUT_SECONDS}"
echo "[setup] starting compare_passk_aime.py"

COMPARE_ARGS=(
  --model "$MODEL_PATH"
  --experiment-mode "$EXPERIMENT_MODE"
  --benchmark "$BENCHMARK"
  --port "$PORT"
  --max-k "$MAX_K"
  --max-prompts "$MAX_PROMPTS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --reasoning-prefix-token-values "$REASONING_PREFIX_TOKEN_VALUES"
  --branch-ablation-reasoning-prefix-tokens "$BRANCH_ABLATION_REASONING_PREFIX_TOKENS"
  --branch-ablation-group-sizes "$BRANCH_ABLATION_GROUP_SIZES"
  --adaptive-ablation-shared-counts "$ADAPTIVE_ABLATION_SHARED_COUNTS"
  --adaptive-confidence-threshold "$ADAPTIVE_CONFIDENCE_THRESHOLD"
  --memory-match-shared-groups "$MEMORY_MATCH_SHARED_GROUPS"
  --memory-match-topup-generator "${MEMORY_MATCH_TOPUP_GENERATOR:-fixed}"
  --memory-match-topup-shared-group-size "$MEMORY_MATCH_TOPUP_SHARED_GROUP_SIZE"
  --memory-match-checkpoint-family "${MEMORY_MATCH_CHECKPOINT_FAMILY:-}"
  --memory-match-checkpoint-root "${MEMORY_MATCH_CHECKPOINT_ROOT:-}"
  --memory-match-checkpoint-campaign "${MEMORY_MATCH_CHECKPOINT_CAMPAIGN:-}"
  --memory-match-checkpoint-step "${MEMORY_MATCH_CHECKPOINT_STEP:-20}"
  --memory-match-checkpoint-before-label "$MEMORY_MATCH_CHECKPOINT_BEFORE_LABEL"
  --memory-match-checkpoint-after-label "$MEMORY_MATCH_CHECKPOINT_AFTER_LABEL"
  --memory-match-checkpoint-title "$MEMORY_MATCH_CHECKPOINT_TITLE"
  --hyperparam-reasoning-prefix-tokens "$HYPERPARAM_REASONING_PREFIX_TOKENS"
  --hyperparam-top-p-values "$HYPERPARAM_TOP_P_VALUES"
  --hyperparam-temperature-values "$HYPERPARAM_TEMPERATURE_VALUES"
  --checkpoint-matched-prompts-step "$CHECKPOINT_MATCHED_PROMPTS_STEP"
  --methods "$METHODS"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
  --dp-size "$DP_SIZE"
  --tp-size "$TP_SIZE"
  --resource-profile "$RESOURCE_PROFILE"
  --throughput-profile "$THROUGHPUT_PROFILE"
  --prompts-per-rank "$PROMPTS_PER_RANK"
  --rank-scheduler "$RANK_SCHEDULER"
  --server-timeout-seconds "$SERVER_TIMEOUT_SECONDS"
  --resume
)

if [[ "$BRANCH_ABLATION_NO_BASELINE" == "1" || "$BRANCH_ABLATION_NO_BASELINE" == "true" || "$BRANCH_ABLATION_NO_BASELINE" == "TRUE" || "$BRANCH_ABLATION_NO_BASELINE" == "yes" || "$BRANCH_ABLATION_NO_BASELINE" == "on" ]]; then
  COMPARE_ARGS+=(--branch-ablation-no-baseline)
fi

if [[ -n "$MEMORY_MATCH_SOURCE_ROOT" ]]; then
  COMPARE_ARGS+=(--memory-match-source-root "$MEMORY_MATCH_SOURCE_ROOT")
fi

if [[ "$COMPACT_JSONL" == "1" || "$COMPACT_JSONL" == "true" || "$COMPACT_JSONL" == "TRUE" || "$COMPACT_JSONL" == "yes" || "$COMPACT_JSONL" == "on" ]]; then
  export MULTIPLEX_COMPACT_JSONL=1
  COMPARE_ARGS+=(--compact-jsonl)
fi

if [[ -n "$REQUEST_BATCH_SIZE" ]]; then
  COMPARE_ARGS+=(--request-batch-size "$REQUEST_BATCH_SIZE")
fi

if [[ -n "$MAX_CONCURRENT_PROMPTS" ]]; then
  COMPARE_ARGS+=(--max-concurrent-prompts "$MAX_CONCURRENT_PROMPTS")
fi

if [[ -n "$PROMPT_INDICES" ]]; then
  COMPARE_ARGS+=(--prompt-indices "$PROMPT_INDICES")
fi

"$PYTHON_BIN" scripts/compare_passk_aime.py \
  "${COMPARE_ARGS[@]}"

echo "[setup] final summary: ${OUTPUT_DIR}/summary_overall.md"
