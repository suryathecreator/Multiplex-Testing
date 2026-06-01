#!/bin/bash
#SBATCH --job-name=shared4-obj
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=09:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint=h200
#SBATCH --output=/gscratch/scrubbed/suryadv/repos/Multiplex-Testing/slurm_logs/%x-%j.out
#SBATCH --chdir=/gscratch/scrubbed/suryadv/repos/Multiplex-Testing

set -euo pipefail

START_TIME="$(date +%s)"
REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"
CONFIG_SMOKE_ONLY="${CONFIG_SMOKE_ONLY:-False}"

truthy_early() {
  [[ "${1:-}" =~ ^(1|true|TRUE|True|yes|YES|on|ON)$ ]]
}

SCRATCH_ROOT="${SCRATCH_ROOT:-/gscratch/scrubbed/suryadv}"
SCRATCH_CACHE_ROOT="${SCRATCH_CACHE_ROOT:-${SCRATCH_ROOT}/.cache/multiplex-thinking}"
SCRATCH_TMP_ROOT="${SCRATCH_ROOT}/tmp/shared4-objective-${SLURM_JOB_ID:-local}"
SCRATCH_RUNTIME_ROOT="${SCRATCH_CACHE_ROOT}/runtime-envs"
JOB_OVERLAY_DIR="${SCRATCH_TMP_ROOT}/job-overlay"
JOB_BIN_DIR="${SCRATCH_TMP_ROOT}/bin"
SELECTED_RUNTIME_MANIFEST="${SCRATCH_TMP_ROOT}/runtime-manifest.json"
LOCAL_PYTHONPATH="${PWD}/verl-latest:${PWD}/sglang-0.4.9.post6:${PWD}/transformers-4.54.0/src"
ENV_PREFIX="/mmfs1/home/suryadv/.conda/envs/multiplex-thinking"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

mkdir -p slurm_logs final_eval_outputs
if ! truthy_early "$CONFIG_SMOKE_ONLY"; then
  mkdir -p "$SCRATCH_CACHE_ROOT" "$SCRATCH_TMP_ROOT" "$SCRATCH_RUNTIME_ROOT" "$JOB_OVERLAY_DIR" "$JOB_BIN_DIR"

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
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  fi
fi

ensure_host_compiler() {
  if command -v module >/dev/null 2>&1; then
    module load gcc/11.2.0 >/dev/null 2>&1 || module load gcc/12.3.0 >/dev/null 2>&1 || module load gcc/9.3.0 >/dev/null 2>&1 || true
  fi

  if ! command -v gcc >/dev/null 2>&1 || [[ "$(gcc -dumpversion | cut -d. -f1)" -lt 9 ]]; then
    local candidate
    for candidate in /mmfs1/sw/gcc/11.2.0 /mmfs1/sw/gcc/12.3.0 /mmfs1/sw/gcc/9.3.0 /sw/gcc/11.2.0 /sw/gcc/12.3.0 /sw/gcc/9.3.0; do
      if [[ -x "${candidate}/bin/gcc" && -x "${candidate}/bin/g++" ]]; then
        export PATH="${candidate}/bin:${PATH}"
        export LD_LIBRARY_PATH="${candidate}/lib64:${candidate}/lib:${LD_LIBRARY_PATH:-}"
        break
      fi
    done
  fi

  if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    export CC="$(command -v gcc)"
    export CXX="$(command -v g++)"
    export CUDAHOSTCXX="$CXX"
    echo "[setup] gcc=$("$CC" --version | head -n 1)"
    echo "[setup] gxx=$("$CXX" --version | head -n 1)"
  else
    echo "[WARN] gcc/g++ not found; FlashInfer may fail if it needs to JIT kernels." >&2
  fi
}

ensure_cuda_nvcc() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    return 0
  fi

  if command -v module >/dev/null 2>&1; then
    module load cuda/12.8.1 >/dev/null 2>&1 || module load cuda/12.4.1 >/dev/null 2>&1 || true
  fi

  if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    local candidate
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
    echo "[setup] nvcc=$("${CUDA_HOME}/bin/nvcc" --version | tail -n 1)"
  else
    echo "[WARN] nvcc not found; FlashInfer may fail if it needs to JIT kernels." >&2
  fi
}

if ! truthy_early "$CONFIG_SMOKE_ONLY"; then
  ensure_host_compiler
  ensure_cuda_nvcc
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export VERL_DISABLE_EXPANDABLE_SEGMENTS=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export OMPI_COMM_WORLD_RANK=0

if ! truthy_early "$CONFIG_SMOKE_ONLY"; then
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
    "$PYTHONPYCACHEPREFIX" \
    "$RAY_TMPDIR"

  BOOTSTRAP_LOG="${SCRATCH_CACHE_ROOT}/bootstrap-runtime-shared4-objective-${SLURM_JOB_ID:-local}.log"
  BOOTSTRAP_LOCK="${SCRATCH_CACHE_ROOT}/bootstrap-runtime.lock"
  echo "[setup] bootstrapping scratch-managed runtime"
  echo "[setup] bootstrap_log=${BOOTSTRAP_LOG}"
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
  for runtime_lib_dir in "$MANAGED_RUNTIME_SITE"/torch/lib "$MANAGED_RUNTIME_SITE"/nvidia/*/lib; do
    if [[ -d "$runtime_lib_dir" ]]; then
      export LD_LIBRARY_PATH="${runtime_lib_dir}:${LD_LIBRARY_PATH:-}"
    fi
  done
  if [[ -n "$LIGHTWEIGHT_OVERLAY_CACHE_DIR" ]]; then
    export PYTHONPATH="${MANAGED_RUNTIME_SITE}:${JOB_OVERLAY_DIR}:${LIGHTWEIGHT_OVERLAY_CACHE_DIR}:${LOCAL_PYTHONPATH}"
    export PATH="${JOB_OVERLAY_DIR}/bin:${LIGHTWEIGHT_OVERLAY_CACHE_DIR}/bin:${JOB_BIN_DIR}:${PATH}"
  else
    export PYTHONPATH="${MANAGED_RUNTIME_SITE}:${JOB_OVERLAY_DIR}:${LOCAL_PYTHONPATH}"
    export PATH="${JOB_OVERLAY_DIR}/bin:${JOB_BIN_DIR}:${PATH}"
  fi
else
  export PYTHONPATH="${LOCAL_PYTHONPATH}"
fi

echo "[setup] python=${PYTHON_BIN}"
echo "[setup] pythonpath=${PYTHONPATH}"
echo "[setup] slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
echo "[setup] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[setup] cuda_launch_blocking=${CUDA_LAUNCH_BLOCKING}"

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
RUN_STAMP="${RUN_TAG:-shared4-objective-${SLURM_JOB_ID:-local}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/${RUN_STAMP}}"
DATA_DIR="${OUTPUT_DIR}/data"
LOG_DIR="${OUTPUT_DIR}/logs"
CHECKPOINT_ROOT="${OUTPUT_DIR}/checkpoints"
PLOT_DIR="${OUTPUT_DIR}/training_plots"
EVAL_ROOT="${OUTPUT_DIR}/evals"
mkdir -p "$DATA_DIR" "$LOG_DIR" "$CHECKPOINT_ROOT" "$PLOT_DIR" "$EVAL_ROOT"

TASK_MODE="${TASK_MODE:-train}"
if [[ "$TASK_MODE" == "soft_smoke" ]]; then
  SOFT_SMOKE_BACKEND="${SOFT_SMOKE_BACKEND:-sglang-vanilla}"
  SOFT_SMOKE_LOG="${LOG_DIR}/soft_smoke_${SOFT_SMOKE_BACKEND}.log"
  echo "[soft-smoke] backend=${SOFT_SMOKE_BACKEND} log=${SOFT_SMOKE_LOG}"
  "$PYTHON_BIN" scripts/smoke_soft_thinking_generation.py \
    --backend "$SOFT_SMOKE_BACKEND" \
    --model "$MODEL_PATH" \
    --multiplex-width "${MULTIPLEX_WIDTH_OVERRIDE:-${MULTIPLEX_WIDTH:-3}}" \
    --thinking-tokens "${SOFT_SMOKE_THINKING_TOKENS:-${BRANCH_ROLLOUT_THINKING_TOKENS:-8}}" \
    --continuation-tokens "${SOFT_SMOKE_CONTINUATION_TOKENS:-${BRANCH_ROLLOUT_CONTINUATION_TOKENS:-8}}" \
    --traces "${SOFT_SMOKE_TRACES:-4}" \
    --answers-per-trace "${SOFT_SMOKE_ANSWERS_PER_TRACE:-4}" \
    --timeout-seconds "${SOFT_SMOKE_TIMEOUT_SECONDS:-240}" \
    --generation-timeout-seconds "${SOFT_SMOKE_GENERATION_TIMEOUT_SECONDS:-180}" \
    --gpu-mem-util "${GPU_MEM_UTIL:-0.50}" \
    --attention-backend "${SGLANG_ATTENTION_BACKEND:-flashinfer}" \
    --sampling-backend "${SGLANG_SAMPLING_BACKEND:-pytorch}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    2>&1 | tee "$SOFT_SMOKE_LOG"
  exit "${PIPESTATUS[0]}"
fi
TRAIN_EXAMPLE_COUNT="${TRAIN_EXAMPLES:-10000}"
USE_FULL_TRAIN_SET="${USE_FULL_TRAIN_SET:-True}"
if truthy_early "$USE_FULL_TRAIN_SET"; then
  TRAIN_EXAMPLE_COUNT="ALL"
fi
TOTAL_STEPS_PER_VARIANT="${TOTAL_STEPS_PER_VARIANT:-64}"
SHARD_STEPS="${SHARD_STEPS:-4}"
TRAIN_TARGET_STEPS="${TRAIN_TARGET_STEPS:-$TOTAL_STEPS_PER_VARIANT}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
RUN_CONDITIONS="${RUN_CONDITIONS:-shared4_joint,shared4_thinking_only,shared4_answer_only}"
TRAIN_BATCH_SIZE_OVERRIDE="${TRAIN_BATCH_SIZE:-}"
PPO_MINI_BATCH_SIZE_OVERRIDE="${PPO_MINI_BATCH_SIZE:-}"
ROLLOUT_N_OVERRIDE="${ROLLOUT_N:-}"
MAX_RESPONSE_LENGTH_OVERRIDE="${MAX_RESPONSE_LENGTH:-}"
MAX_TOKEN_LEN_PER_GPU_OVERRIDE="${MAX_TOKEN_LEN_PER_GPU:-}"
ROLLOUT_MAX_NUM_SEQS_OVERRIDE="${ROLLOUT_MAX_NUM_SEQS:-}"
SAVE_FREQ_OVERRIDE="${SAVE_FREQ:-}"
GPU_MEM_UTIL_OVERRIDE="${GPU_MEM_UTIL:-}"
ROLLOUT_N="${ROLLOUT_N:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-256}"
AUTO_OOM_RETRY="${AUTO_OOM_RETRY:-True}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-False}"
ACCELERATOR_LABEL="${ACCELERATOR_LABEL:-auto}"
CONFIG_SMOKE_ONLY="${CONFIG_SMOKE_ONLY:-False}"
EVAL_TASK="${EVAL_TASK:-}"
EVAL_SEED="${EVAL_SEED:-26010808}"
EVAL_REPEAT_LABEL="${EVAL_REPEAT_LABEL:-seed_${EVAL_SEED}}"
EVAL_MAX_K="${EVAL_MAX_K:-8}"
EVAL_MAX_PROMPTS="${EVAL_MAX_PROMPTS:-30}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-5120}"
REASONING_PREFIX_TOKENS="${REASONING_PREFIX_TOKENS:-4096}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-16}"
EVAL_MAX_CONCURRENT_PROMPTS="${EVAL_MAX_CONCURRENT_PROMPTS:-2}"
EVAL_SERVER_TIMEOUT_SECONDS="${EVAL_SERVER_TIMEOUT_SECONDS:-900}"
EVAL_DISABLE_CUDA_GRAPH="${EVAL_DISABLE_CUDA_GRAPH:-True}"
EVAL_MEM_FRACTION_STATIC="${EVAL_MEM_FRACTION_STATIC:-0.78}"
AGGREGATE_BOOTSTRAP_RUNS="${AGGREGATE_BOOTSTRAP_RUNS:-1000}"
AGGREGATE_SEED="${AGGREGATE_SEED:-26010808}"
AGGREGATE_LABEL="${AGGREGATE_LABEL:-}"

if [[ "$TRAIN_TARGET_STEPS" -gt "$TOTAL_STEPS_PER_VARIANT" ]]; then
  TRAIN_TARGET_STEPS="$TOTAL_STEPS_PER_VARIANT"
fi

if [[ "${FLASH_ATTN_RUNTIME_SMOKE_ONLY:-False}" =~ ^(1|true|TRUE|True|yes|YES|on|ON)$ ]]; then
  echo "[smoke] validating FlashAttention runtime"
  echo "[smoke] attn_implementation=${ATTN_IMPLEMENTATION}"
  FLASH_ATTN_SMOKE_ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" "$PYTHON_BIN" - <<'PY'
import importlib.metadata
import importlib.util
import os

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

attn_implementation = os.environ["FLASH_ATTN_SMOKE_ATTN_IMPLEMENTATION"]
if attn_implementation != "flash_attention_2":
    raise RuntimeError(f"smoke expected flash_attention_2, got {attn_implementation!r}")

flash_spec = importlib.util.find_spec("flash_attn")
flash_version = importlib.metadata.version("flash_attn")
if flash_spec is None or flash_spec.origin is None:
    raise RuntimeError("flash_attn import spec is missing")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available for FlashAttention smoke test")

device = torch.device("cuda")
config = Qwen2Config(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=128,
)
config.torch_dtype = torch.bfloat16
config._attn_implementation = attn_implementation
model = Qwen2ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
model.eval()
input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
with torch.no_grad():
    logits = model(input_ids=input_ids).logits
torch.cuda.synchronize()

print(f"[smoke] flash_attn_version={flash_version}")
print(f"[smoke] flash_attn_origin={flash_spec.origin}")
print(f"[smoke] cuda_device={torch.cuda.get_device_name(0)}")
print(f"[smoke] logits_shape={tuple(logits.shape)}")
print("[smoke] FlashAttention forward pass succeeded")
PY
  echo "[smoke] FlashAttention runtime validation finished"
  exit 0
fi

stop_ray() {
  if truthy "${ENABLE_RAY_STOP:-False}" && command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  else
    echo "[ray] skipping global ray stop; ENABLE_RAY_STOP=${ENABLE_RAY_STOP:-False}"
  fi
}

truthy() {
  [[ "${1:-}" =~ ^(1|true|TRUE|True|yes|YES|on|ON)$ ]]
}

detect_accelerator_label() {
  local names
  if command -v nvidia-smi >/dev/null 2>&1; then
    names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)"
    case "${names,,}" in
      *h200*) echo "H200"; return 0 ;;
      *a100*) echo "A100"; return 0 ;;
    esac
  fi

  case "${SLURM_JOB_CONSTRAINTS:-}${SLURM_JOB_FEATURES:-}${SLURM_JOB_GRES:-}${SLURM_STEP_GRES:-}" in
    *h200*|*H200*) echo "H200"; return 0 ;;
    *a100*|*A100*) echo "A100"; return 0 ;;
  esac

  echo "unknown"
}

resolve_accelerator_label() {
  local requested="${ACCELERATOR_LABEL:-auto}"
  local detected
  detected="$(detect_accelerator_label)"
  case "${requested,,}" in
    auto|any|anygpu|unknown|"")
      if [[ "$detected" != "unknown" ]]; then
        echo "$detected"
      else
        echo "A100"
      fi
      ;;
    *)
      if [[ "$detected" != "unknown" ]]; then
        echo "$detected"
      else
        echo "$requested"
      fi
      ;;
  esac
}

accelerator_class() {
  case "${ACCELERATOR_LABEL,,}" in
    *h200*) echo "H200" ;;
    *) echo "A100" ;;
  esac
}

ACCELERATOR_LABEL="$(resolve_accelerator_label)"
echo "[config] resolved_accelerator=${ACCELERATOR_LABEL}"

find_hf_checkpoint() {
  local checkpoint_dir="$1"
  find "$checkpoint_dir" -type f -path "*/actor/huggingface/config.json" -printf "%h\n" 2>/dev/null | sort -V | tail -n 1 || true
}

model_for_eval_task() {
  case "$1" in
    discrete_untrained|multiplex_untrained|shared4_untrained)
      echo "$MODEL_PATH"
      ;;
    discrete_trained|discrete_trained_multiplex_eval)
      find_hf_checkpoint "${CHECKPOINT_ROOT}/discrete_rl"
      ;;
    multiplex_trained|multiplex_trained_discrete_eval|shared4_multiplex_trained)
      find_hf_checkpoint "${CHECKPOINT_ROOT}/multiplex_thinking"
      ;;
    shared4_joint_trained)
      find_hf_checkpoint "${CHECKPOINT_ROOT}/shared4_joint"
      ;;
    shared4_thinking_trained)
      find_hf_checkpoint "${CHECKPOINT_ROOT}/shared4_thinking_only"
      ;;
    shared4_answer_trained)
      find_hf_checkpoint "${CHECKPOINT_ROOT}/shared4_answer_only"
      ;;
    *)
      echo "[config] unknown EVAL_TASK: $1" >&2
      return 1
      ;;
  esac
}

run_eval_task() {
  if [[ -z "$EVAL_TASK" ]]; then
    echo "[config] TASK_MODE=eval requires EVAL_TASK" >&2
    exit 1
  fi

  local model_path
  model_path="$(model_for_eval_task "$EVAL_TASK")"
  if [[ -z "$model_path" ]]; then
    echo "[eval] missing model checkpoint for task=${EVAL_TASK}" >&2
    exit 1
  fi

  local eval_dir="${EVAL_ROOT}/${EVAL_REPEAT_LABEL}/${EVAL_TASK}"
  mkdir -p "$eval_dir"
  stop_ray
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export OMPI_COMM_WORLD_RANK=0

  local common_args=(
    --model "$model_path"
    --benchmark aime2024
    --port "${EVAL_PORT:-36101}"
    --dp-size "$GPUS_PER_NODE"
    --tp-size 1
    --resource-profile auto
    --throughput-profile safe_auto
    --request-batch-size "$EVAL_REQUEST_BATCH_SIZE"
    --max-concurrent-prompts "$EVAL_MAX_CONCURRENT_PROMPTS"
    --prompts-per-rank 1
    --rank-scheduler dynamic
    --server-timeout-seconds "$EVAL_SERVER_TIMEOUT_SECONDS"
    --max-k "$EVAL_MAX_K"
    --max-prompts "$EVAL_MAX_PROMPTS"
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS"
    --seed "$EVAL_SEED"
    --compact-jsonl
    --resume
  )
  if truthy "$EVAL_DISABLE_CUDA_GRAPH"; then
    common_args+=(--disable-cuda-graph)
  fi
  if [[ -n "$EVAL_MEM_FRACTION_STATIC" ]]; then
    common_args+=(--mem-fraction-static "$EVAL_MEM_FRACTION_STATIC")
  fi

  case "$EVAL_TASK" in
    discrete_untrained|discrete_trained|multiplex_trained_discrete_eval)
      "$PYTHON_BIN" scripts/compare_passk_aime.py \
        --experiment-mode passk_sweep \
        --methods standard_generation \
        "${common_args[@]}" \
        --reasoning-prefix-token-values "$REASONING_PREFIX_TOKENS" \
        --output-dir "$eval_dir"
      ;;
    multiplex_untrained|multiplex_trained|discrete_trained_multiplex_eval)
      "$PYTHON_BIN" scripts/compare_passk_aime.py \
        --experiment-mode passk_sweep \
        --methods baseline \
        "${common_args[@]}" \
        --reasoning-prefix-token-values "$REASONING_PREFIX_TOKENS" \
        --output-dir "$eval_dir"
      ;;
    shared4_untrained|shared4_multiplex_trained|shared4_joint_trained|shared4_thinking_trained|shared4_answer_trained)
      "$PYTHON_BIN" scripts/compare_passk_aime.py \
        --experiment-mode branch_ablation \
        --methods baseline,shared_trace \
        "${common_args[@]}" \
        --branch-ablation-reasoning-prefix-tokens "$REASONING_PREFIX_TOKENS" \
        --branch-ablation-group-sizes 4 \
        --output-dir "$eval_dir"
      ;;
  esac
  stop_ray

  local expected_summary
  case "$EVAL_TASK" in
    shared4_*) expected_summary="${eval_dir}/summary_ablation.md" ;;
    *) expected_summary="${eval_dir}/summary_overall.md" ;;
  esac
  if [[ ! -f "$expected_summary" ]]; then
    echo "[eval] missing expected summary=${expected_summary}" >&2
    exit 1
  fi
  echo "[eval] finished task=${EVAL_TASK} seed=${EVAL_SEED} summary=${expected_summary}"
}

run_aggregate_task() {
  local label_suffix=""
  if [[ -n "$AGGREGATE_LABEL" ]]; then
    label_suffix="_${AGGREGATE_LABEL}"
  fi
  local aggregate_dir="${OUTPUT_DIR}/aggregate_bootstrap${label_suffix}"
  local training_plot_dir="${PLOT_DIR}"
  if [[ -n "$AGGREGATE_LABEL" ]]; then
    training_plot_dir="${PLOT_DIR}/${AGGREGATE_LABEL}"
  fi
  mkdir -p "$aggregate_dir" "$training_plot_dir"
  "$PYTHON_BIN" scripts/plot_eval_bootstrap_comparisons.py \
    --eval-root "$EVAL_ROOT" \
    --output-dir "$aggregate_dir" \
    --bootstrap-runs "$AGGREGATE_BOOTSTRAP_RUNS" \
    --seed "$AGGREGATE_SEED"
  "$PYTHON_BIN" scripts/plot_training_logs.py \
    --run "discrete_rl=${LOG_DIR}/discrete_rl_train.log" \
    --run "multiplex_thinking=${LOG_DIR}/multiplex_thinking_train.log" \
    --run "shared4_joint=${LOG_DIR}/shared4_joint_train.log" \
    --run "shared4_thinking_only=${LOG_DIR}/shared4_thinking_only_train.log" \
    --run "shared4_answer_only=${LOG_DIR}/shared4_answer_only_train.log" \
    --output-dir "$training_plot_dir" \
    --summary-json "${training_plot_dir}/training_plot_summary.json"
  echo "[aggregate] finished output_dir=${aggregate_dir}"
}

if [[ "$TASK_MODE" == "aggregate" ]]; then
  run_aggregate_task
  END_TIME="$(date +%s)"
  WALL_SECONDS="$((END_TIME - START_TIME))"
  SUMMARY_PATH="${OUTPUT_DIR}/aggregate_summary.txt"
  if [[ -n "$AGGREGATE_LABEL" ]]; then
    SUMMARY_PATH="${OUTPUT_DIR}/aggregate_${AGGREGATE_LABEL}_summary.txt"
  fi
  {
    echo "wall_seconds=${WALL_SECONDS}"
    echo "eval_root=${EVAL_ROOT}"
    echo "aggregate_bootstrap_runs=${AGGREGATE_BOOTSTRAP_RUNS}"
    echo "aggregate_seed=${AGGREGATE_SEED}"
    echo "aggregate_label=${AGGREGATE_LABEL:-none}"
  } > "$SUMMARY_PATH"
  echo "[summary] ${SUMMARY_PATH}"
  exit 0
fi

if [[ "$TASK_MODE" == "eval" ]]; then
  run_eval_task
  END_TIME="$(date +%s)"
  WALL_SECONDS="$((END_TIME - START_TIME))"
  SUMMARY_PATH="${OUTPUT_DIR}/eval_${EVAL_TASK}_${EVAL_REPEAT_LABEL}_summary.txt"
  {
    echo "task=${EVAL_TASK}"
    echo "seed=${EVAL_SEED}"
    echo "repeat_label=${EVAL_REPEAT_LABEL}"
    echo "wall_seconds=${WALL_SECONDS}"
    echo "eval_root=${EVAL_ROOT}"
    echo "accelerator=${ACCELERATOR_LABEL}"
    echo "gpus_per_node=${GPUS_PER_NODE}"
    echo "eval_disable_cuda_graph=${EVAL_DISABLE_CUDA_GRAPH}"
    echo "eval_mem_fraction_static=${EVAL_MEM_FRACTION_STATIC}"
  } > "$SUMMARY_PATH"
  echo "[summary] ${SUMMARY_PATH}"
  exit 0
fi

TRAIN_SUBSET="${DATA_DIR}/deepscaler_train_subset${TRAIN_EXAMPLE_COUNT}.parquet"
TRAIN_SUBSET_MANIFEST="${DATA_DIR}/deepscaler_train_subset${TRAIN_EXAMPLE_COUNT}_manifest.json"
if ! truthy "$CONFIG_SMOKE_ONLY"; then
  "$PYTHON_BIN" scripts/create_deepscaler_subset.py \
    --input deepscaler/hdfs_data/train.parquet \
    --output "$TRAIN_SUBSET" \
    --manifest "$TRAIN_SUBSET_MANIFEST" \
    --num-examples "$TRAIN_EXAMPLE_COUNT" \
    --seed "${TRAIN_SUBSET_SEED:-26010808}"
fi

build_train_common_args() {
  TRAIN_COMMON_ARGS=(
    --model "$MODEL_PATH"
    --train_file "$TRAIN_SUBSET"
    --val_file deepscaler/hdfs_data/aime.parquet
    --n_gpus_per_node "$GPUS_PER_NODE"
    --total_training_steps "$TOTAL_STEPS_PER_VARIANT"
    --stop_at_step "$TRAIN_TARGET_STEPS"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --ppo_mini_batch_size "$PPO_MINI_BATCH_SIZE"
    --rollout_n "$ROLLOUT_N"
    --rollout_name "${ROLLOUT_NAME:-sglang}"
    --rollout_max_num_seqs "$ROLLOUT_MAX_NUM_SEQS"
    --shared4_advantage "$CONDITION_SHARED4_ADVANTAGE"
    --shared4_thinking_rollouts 4
    --shared4_answers_per_trace 4
    --branch_rollout "$CONDITION_BRANCH_ROLLOUT"
    --branch_rollout_thinking_traces "$CONDITION_BRANCH_ROLLOUT_THINKING_TRACES"
    --branch_rollout_answers_per_trace "$CONDITION_BRANCH_ROLLOUT_ANSWERS_PER_TRACE"
    --branch_rollout_thinking_tokens "$CONDITION_BRANCH_ROLLOUT_THINKING_TOKENS"
    --branch_rollout_continuation_tokens "$CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --max_response_length "$MAX_RESPONSE_LENGTH"
    --max_token_len_per_gpu "$MAX_TOKEN_LEN_PER_GPU"
    --save_freq "$SAVE_FREQ"
    --test_freq -1
    --val_before_train False
    --val_rollout_n 16
    --val_batch_size 64
    --logger "['console']"
    --save_hf_model "$SAVE_HF_MODEL"
    --enforce_eager "$ENFORCE_EAGER"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --gpu_mem_util "$GPU_MEM_UTIL"
    --top_p 1.0
    --temp 1.0
    --early_stopping_length_threshold "$EARLY_STOPPING_LENGTH_THRESHOLD"
    --enable_unweighting False
    --wandb_project shared4-objective
  )
  if [[ -n "${CHECKPOINT_SAVE_CONTENTS:-}" ]]; then
    TRAIN_COMMON_ARGS+=(--checkpoint_save_contents "$CHECKPOINT_SAVE_CONTENTS")
  fi
  if [[ -n "${CHECKPOINT_LOAD_CONTENTS:-}" ]]; then
    TRAIN_COMMON_ARGS+=(--checkpoint_load_contents "$CHECKPOINT_LOAD_CONTENTS")
  fi
}

stop_ray() {
  if truthy "${ENABLE_RAY_STOP:-False}" && command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  else
    echo "[ray] skipping global ray stop; ENABLE_RAY_STOP=${ENABLE_RAY_STOP:-False}"
  fi
}

truthy() {
  [[ "${1:-}" =~ ^(1|true|TRUE|True|yes|YES|on|ON)$ ]]
}

latest_checkpoint_step() {
  local checkpoint_dir="$1"
  if [[ ! -d "$checkpoint_dir" ]]; then
    echo 0
    return 0
  fi
  local tracker="${checkpoint_dir}/latest_checkpointed_iteration.txt"
  if [[ -f "$tracker" ]]; then
    local tracked_step
    tracked_step="$(tr -dc '0-9' < "$tracker" || true)"
    if [[ -n "$tracked_step" ]]; then
      echo "$tracked_step"
      return 0
    fi
  fi
  find "$checkpoint_dir" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null \
    | sed -E 's/^global_step_//' \
    | sort -n \
    | tail -n 1
}

condition_requested() {
  local condition="$1"
  local requested="${RUN_CONDITIONS// /}"
  requested="${requested//+/,}"
  requested="${requested//:/,}"
  local normalized=",${requested},"
  [[ "$normalized" == *",${condition},"* ]]
}

condition_loss_mode() {
  case "$1" in
    discrete_rl) echo "vanilla" ;;
    multiplex_thinking) echo "multiplex_thinking" ;;
    shared4_joint) echo "shared4_joint" ;;
    shared4_thinking_only) echo "shared4_thinking_only" ;;
    shared4_answer_only) echo "shared4_answer_only" ;;
    *)
      echo "[config] unknown condition: $1" >&2
      return 1
      ;;
  esac
}

configure_condition() {
  case "$1" in
    discrete_rl)
      CONDITION_ENABLE_SOFT_THINK=False
      CONDITION_MULTIPLEX_WIDTH=1
      CONDITION_SHARED4_ADVANTAGE=False
      CONDITION_BRANCH_ROLLOUT=False
      CONDITION_BRANCH_ROLLOUT_THINKING_TRACES=4
      CONDITION_BRANCH_ROLLOUT_ANSWERS_PER_TRACE=4
      CONDITION_BRANCH_ROLLOUT_THINKING_TOKENS="$EARLY_STOPPING_LENGTH_THRESHOLD"
      CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS=0
      ;;
    multiplex_thinking)
      CONDITION_ENABLE_SOFT_THINK=True
      CONDITION_MULTIPLEX_WIDTH="${MULTIPLEX_WIDTH_OVERRIDE:-3}"
      CONDITION_SHARED4_ADVANTAGE=False
      CONDITION_BRANCH_ROLLOUT=False
      CONDITION_BRANCH_ROLLOUT_THINKING_TRACES="${BRANCH_ROLLOUT_THINKING_TRACES:-4}"
      CONDITION_BRANCH_ROLLOUT_ANSWERS_PER_TRACE="${BRANCH_ROLLOUT_ANSWERS_PER_TRACE:-4}"
      CONDITION_BRANCH_ROLLOUT_THINKING_TOKENS="${BRANCH_ROLLOUT_THINKING_TOKENS:-$EARLY_STOPPING_LENGTH_THRESHOLD}"
      CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS="${BRANCH_ROLLOUT_CONTINUATION_TOKENS:-0}"
      ;;
    shared4_joint|shared4_thinking_only|shared4_answer_only)
      CONDITION_ENABLE_SOFT_THINK=True
      CONDITION_MULTIPLEX_WIDTH="${SHARED4_MULTIPLEX_WIDTH_OVERRIDE:-${MULTIPLEX_WIDTH_OVERRIDE:-4}}"
      CONDITION_SHARED4_ADVANTAGE=True
      CONDITION_BRANCH_ROLLOUT="${BRANCH_ROLLOUT:-False}"
      CONDITION_BRANCH_ROLLOUT_THINKING_TRACES="${BRANCH_ROLLOUT_THINKING_TRACES:-4}"
      CONDITION_BRANCH_ROLLOUT_ANSWERS_PER_TRACE="${BRANCH_ROLLOUT_ANSWERS_PER_TRACE:-4}"
      CONDITION_BRANCH_ROLLOUT_THINKING_TOKENS="${BRANCH_ROLLOUT_THINKING_TOKENS:-$EARLY_STOPPING_LENGTH_THRESHOLD}"
      CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS="${BRANCH_ROLLOUT_CONTINUATION_TOKENS:-0}"
      if [[ "$ROLLOUT_N" -ne 16 ]]; then
        echo "[config] ${1} requires ROLLOUT_N=16 for 4 shared traces x 4 answers; got ${ROLLOUT_N}" >&2
        exit 1
      fi
      ;;
    *)
      echo "[config] unknown condition: $1" >&2
      exit 1
      ;;
  esac
  if truthy "$CONDITION_BRANCH_ROLLOUT"; then
    if [[ "$CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS" -le 0 ]]; then
      CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS="$((MAX_RESPONSE_LENGTH - CONDITION_BRANCH_ROLLOUT_THINKING_TOKENS - 1))"
    fi
    if [[ "$((CONDITION_BRANCH_ROLLOUT_THINKING_TOKENS + 1 + CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS))" -gt "$MAX_RESPONSE_LENGTH" ]]; then
      CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS="$((MAX_RESPONSE_LENGTH - CONDITION_BRANCH_ROLLOUT_THINKING_TOKENS - 1))"
      echo "[config] ${1} branch_rollout continuation clamped to ${CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS} to fit max_response=${MAX_RESPONSE_LENGTH}" >&2
    fi
    if [[ "$CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS" -le 0 ]]; then
      echo "[config] ${1} branch_rollout leaves no discrete continuation tokens: max_response=${MAX_RESPONSE_LENGTH}, thinking_tokens=${CONDITION_BRANCH_ROLLOUT_THINKING_TOKENS}" >&2
      exit 1
    fi
    if [[ "$ROLLOUT_N" -ne "$((CONDITION_BRANCH_ROLLOUT_THINKING_TRACES * CONDITION_BRANCH_ROLLOUT_ANSWERS_PER_TRACE))" ]]; then
      echo "[config] ${1} branch_rollout requires rollout_n=traces*answers=$((CONDITION_BRANCH_ROLLOUT_THINKING_TRACES * CONDITION_BRANCH_ROLLOUT_ANSWERS_PER_TRACE)); got ${ROLLOUT_N}" >&2
      exit 1
    fi
  fi
}

configure_training_knobs() {
  local condition="$1"
  local accel
  local default_train_batch_size
  local default_ppo_mini_batch_size
  local default_rollout_n
  local default_max_response_length
  local default_max_token_len_per_gpu
  local default_rollout_max_num_seqs
  local default_gpu_mem_util
  local default_save_freq

  accel="$(accelerator_class)"
  case "$condition:$accel" in
    discrete_rl:A100|multiplex_thinking:A100)
      default_train_batch_size=32
      default_ppo_mini_batch_size=16
      default_rollout_n=4
      default_max_response_length=1024
      default_max_token_len_per_gpu=8192
      default_rollout_max_num_seqs=192
      default_gpu_mem_util=0.70
      default_save_freq="$SHARD_STEPS"
      ;;
    discrete_rl:H200|multiplex_thinking:H200)
      default_train_batch_size=64
      default_ppo_mini_batch_size=32
      default_rollout_n=4
      default_max_response_length=1024
      default_max_token_len_per_gpu=16384
      default_rollout_max_num_seqs=128
      default_gpu_mem_util=0.74
      default_save_freq="$SHARD_STEPS"
      ;;
    shared4_joint:A100|shared4_thinking_only:A100|shared4_answer_only:A100)
      default_train_batch_size=8
      default_ppo_mini_batch_size=4
      default_rollout_n=16
      default_max_response_length=4096
      default_max_token_len_per_gpu=12288
      default_rollout_max_num_seqs=128
      default_gpu_mem_util=0.70
      default_save_freq="$SHARD_STEPS"
      ;;
    shared4_joint:H200|shared4_thinking_only:H200|shared4_answer_only:H200)
      default_train_batch_size=32
      default_ppo_mini_batch_size=16
      default_rollout_n=16
      default_max_response_length=4096
      default_max_token_len_per_gpu=24576
      default_rollout_max_num_seqs=128
      default_gpu_mem_util=0.72
      default_save_freq="$SHARD_STEPS"
      ;;
    *)
      echo "[config] no training knobs for condition=${condition} accelerator=${accel}" >&2
      exit 1
      ;;
  esac

  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE_OVERRIDE:-$default_train_batch_size}"
  PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE_OVERRIDE:-$default_ppo_mini_batch_size}"
  ROLLOUT_N="${ROLLOUT_N_OVERRIDE:-$default_rollout_n}"
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH_OVERRIDE:-$default_max_response_length}"
  MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU_OVERRIDE:-$default_max_token_len_per_gpu}"
  ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS_OVERRIDE:-$default_rollout_max_num_seqs}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL_OVERRIDE:-$default_gpu_mem_util}"
  SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-$default_save_freq}"

  if [[ "$TRAIN_BATCH_SIZE" -lt 2 ]]; then
    echo "[config] train batch size must be at least 2; got ${TRAIN_BATCH_SIZE}" >&2
    exit 1
  fi
  if [[ "$((TRAIN_BATCH_SIZE / 2))" -ne "$PPO_MINI_BATCH_SIZE" ]]; then
    echo "[config] PPO_MINI_BATCH_SIZE should be exactly half of TRAIN_BATCH_SIZE; got ${PPO_MINI_BATCH_SIZE}/${TRAIN_BATCH_SIZE}" >&2
    exit 1
  fi
}

is_oom_log() {
  local log_path="$1"
  [[ -f "$log_path" ]] && grep -Eiq 'out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|Allocation failed|memory allocation' "$log_path"
}

apply_oom_retry_knobs() {
  if [[ "$TRAIN_BATCH_SIZE" -gt 32 ]]; then
    TRAIN_BATCH_SIZE=32
    PPO_MINI_BATCH_SIZE=16
  fi
  if [[ "$MAX_TOKEN_LEN_PER_GPU" -gt 12288 ]]; then
    MAX_TOKEN_LEN_PER_GPU="$((MAX_TOKEN_LEN_PER_GPU * 3 / 4))"
    if [[ "$MAX_TOKEN_LEN_PER_GPU" -lt 12288 ]]; then
      MAX_TOKEN_LEN_PER_GPU=12288
    fi
  fi
  if [[ "$ROLLOUT_MAX_NUM_SEQS" -gt 64 ]]; then
    ROLLOUT_MAX_NUM_SEQS="$((ROLLOUT_MAX_NUM_SEQS / 2))"
    if [[ "$ROLLOUT_MAX_NUM_SEQS" -lt 64 ]]; then
      ROLLOUT_MAX_NUM_SEQS=64
    fi
  fi
  if [[ "$PPO_MINI_BATCH_SIZE" -gt 32 ]]; then
    PPO_MINI_BATCH_SIZE=32
  fi
  ENFORCE_EAGER=True
  GPU_MEM_UTIL="${OOM_RETRY_GPU_MEM_UTIL:-0.66}"
  SAVE_FREQ=1
}

run_training() {
  local condition="$1"
  local loss_mode="$2"
  local target_steps="$3"
  local condition_checkpoint_dir="${CHECKPOINT_ROOT}/${condition}"
  local condition_log="${LOG_DIR}/${condition}_train_step${target_steps}.log"
  local aggregate_log="${LOG_DIR}/${condition}_train.log"
  local done_marker="${condition_checkpoint_dir}/.done"
  local shard_marker="${condition_checkpoint_dir}/.target_step_${target_steps}.done"
  local latest_step

  configure_training_knobs "$condition"
  configure_condition "$condition"
  latest_step="$(latest_checkpoint_step "$condition_checkpoint_dir")"
  latest_step="${latest_step:-0}"

  if [[ -f "$done_marker" && "$latest_step" -ge "$TOTAL_STEPS_PER_VARIANT" ]]; then
    echo "[train] skipping condition=${condition}; done_marker=${done_marker} latest_step=${latest_step}"
    return 0
  fi
  if [[ "$latest_step" -ge "$target_steps" ]]; then
    echo "[train] skipping condition=${condition}; latest_step=${latest_step} >= target_steps=${target_steps}"
    date --iso-8601=seconds > "$shard_marker"
    if [[ "$latest_step" -ge "$TOTAL_STEPS_PER_VARIANT" ]]; then
      date --iso-8601=seconds > "$done_marker"
    fi
    return 0
  fi
  echo "[train] starting condition=${condition} loss_mode=${loss_mode} target_steps=${target_steps} latest_step=${latest_step}"
  echo "[train] checkpoint_dir=${condition_checkpoint_dir}"
  echo "[train] log=${condition_log}"
  stop_ray
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export OMPI_COMM_WORLD_RANK=0
  build_train_common_args
  set +e
  {
    echo
    echo "===== $(date --iso-8601=seconds) condition=${condition} target_steps=${target_steps} job=${SLURM_JOB_ID:-local} ====="
    echo "gpus_per_node=${GPUS_PER_NODE} train_batch_size=${TRAIN_BATCH_SIZE} ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    echo "max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} gpu_mem_util=${GPU_MEM_UTIL} enforce_eager=${ENFORCE_EAGER}"
  } >> "$aggregate_log"
  bash scripts/train.sh \
    "${TRAIN_COMMON_ARGS[@]}" \
    --exp_name "$condition" \
    --default_local_dir "$condition_checkpoint_dir" \
    --enable_soft_think "$CONDITION_ENABLE_SOFT_THINK" \
    --loss_mode "$loss_mode" \
    --multiplex_width "$CONDITION_MULTIPLEX_WIDTH" \
    2>&1 | tee "$condition_log" | tee -a "$aggregate_log"
  local train_status="${PIPESTATUS[0]}"
  set -e
  stop_ray
  if [[ "$train_status" -ne 0 ]] && truthy "$AUTO_OOM_RETRY" && is_oom_log "$condition_log"; then
    echo "[train] condition=${condition} hit an OOM-like failure; retrying once with safer memory knobs" >&2
    apply_oom_retry_knobs
    build_train_common_args
    local retry_log="${LOG_DIR}/${condition}_train_step${target_steps}_oom_retry.log"
    set +e
    {
      echo
      echo "===== $(date --iso-8601=seconds) OOM retry condition=${condition} target_steps=${target_steps} job=${SLURM_JOB_ID:-local} ====="
      echo "gpus_per_node=${GPUS_PER_NODE} train_batch_size=${TRAIN_BATCH_SIZE} ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
      echo "max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} gpu_mem_util=${GPU_MEM_UTIL} enforce_eager=${ENFORCE_EAGER}"
    } >> "$aggregate_log"
    bash scripts/train.sh \
      "${TRAIN_COMMON_ARGS[@]}" \
      --exp_name "$condition" \
      --default_local_dir "$condition_checkpoint_dir" \
      --enable_soft_think "$CONDITION_ENABLE_SOFT_THINK" \
      --loss_mode "$loss_mode" \
      --multiplex_width "$CONDITION_MULTIPLEX_WIDTH" \
      2>&1 | tee "$retry_log" | tee -a "$aggregate_log"
    train_status="${PIPESTATUS[0]}"
    set -e
    stop_ray
  fi
  if [[ "$train_status" -ne 0 ]]; then
    echo "[train] condition=${condition} failed with status=${train_status}" >&2
    exit "$train_status"
  fi
  mkdir -p "$condition_checkpoint_dir"
  date --iso-8601=seconds > "$shard_marker"
  latest_step="$(latest_checkpoint_step "$condition_checkpoint_dir")"
  latest_step="${latest_step:-0}"
  if [[ "$latest_step" -ge "$TOTAL_STEPS_PER_VARIANT" ]]; then
    date --iso-8601=seconds > "$done_marker"
  fi
  echo "[train] finished condition=${condition} target_steps=${target_steps} latest_step=${latest_step}"
}

if truthy "$CONFIG_SMOKE_ONLY"; then
  for condition in discrete_rl multiplex_thinking shared4_joint shared4_thinking_only shared4_answer_only; do
    if condition_requested "$condition"; then
      configure_training_knobs "$condition"
      configure_condition "$condition"
      echo "[config-smoke] condition=${condition} accelerator=${ACCELERATOR_LABEL} gpus_per_node=${GPUS_PER_NODE}"
      echo "[config-smoke] train_batch_size=${TRAIN_BATCH_SIZE} ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N}"
      echo "[config-smoke] multiplex_width=${CONDITION_MULTIPLEX_WIDTH}"
      echo "[config-smoke] effective_trajectory_batch=$((TRAIN_BATCH_SIZE * ROLLOUT_N)) effective_trajectory_minibatch=$((PPO_MINI_BATCH_SIZE * ROLLOUT_N))"
      echo "[config-smoke] max_response_length=${MAX_RESPONSE_LENGTH} max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
      echo "[config-smoke] rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} gpu_mem_util=${GPU_MEM_UTIL} enforce_eager=${ENFORCE_EAGER} save_freq=${SAVE_FREQ}"
      echo "[config-smoke] branch_rollout=${CONDITION_BRANCH_ROLLOUT} traces=${CONDITION_BRANCH_ROLLOUT_THINKING_TRACES} answers_per_trace=${CONDITION_BRANCH_ROLLOUT_ANSWERS_PER_TRACE} thinking_tokens=${CONDITION_BRANCH_ROLLOUT_THINKING_TOKENS} continuation_tokens=${CONDITION_BRANCH_ROLLOUT_CONTINUATION_TOKENS}"
    fi
  done
  exit 0
fi

for condition in discrete_rl multiplex_thinking shared4_joint shared4_thinking_only shared4_answer_only; do
  if condition_requested "$condition"; then
    run_training "$condition" "$(condition_loss_mode "$condition")" "$TRAIN_TARGET_STEPS"
  else
    echo "[train] condition=${condition} not requested by RUN_CONDITIONS=${RUN_CONDITIONS}"
  fi
done

if [[ -f "${CHECKPOINT_ROOT}/discrete_rl/.done" \
      && -f "${CHECKPOINT_ROOT}/multiplex_thinking/.done" \
      && -f "${CHECKPOINT_ROOT}/shared4_joint/.done" \
      && -f "${CHECKPOINT_ROOT}/shared4_thinking_only/.done" \
      && -f "${CHECKPOINT_ROOT}/shared4_answer_only/.done" ]]; then
  "$PYTHON_BIN" scripts/plot_training_logs.py \
    --run "discrete_rl=${LOG_DIR}/discrete_rl_train.log" \
    --run "multiplex_thinking=${LOG_DIR}/multiplex_thinking_train.log" \
    --run "shared4_joint=${LOG_DIR}/shared4_joint_train.log" \
    --run "shared4_thinking_only=${LOG_DIR}/shared4_thinking_only_train.log" \
    --run "shared4_answer_only=${LOG_DIR}/shared4_answer_only_train.log" \
    --output-dir "$PLOT_DIR" \
    --summary-json "${PLOT_DIR}/training_plot_summary.json"
else
  echo "[plot] skipping final plot until all conditions have .done markers"
fi

END_TIME="$(date +%s)"
WALL_SECONDS="$((END_TIME - START_TIME))"
SUMMARY_PATH="${OUTPUT_DIR}/shared4_objective_summary.md"
{
  echo "# Shared4 Objective Suite"
  echo
  echo "- Run stamp: ${RUN_STAMP}"
  echo "- Chain chunk: ${CHAIN_CHUNK_INDEX:-1}/${CHAIN_CHUNKS:-1}"
  echo "- Requested conditions: ${RUN_CONDITIONS}"
  echo "- Wall time seconds: ${WALL_SECONDS}"
  echo "- Model: ${MODEL_PATH}"
  echo "- Hardware: ${GPUS_PER_NODE}x ${ACCELERATOR_LABEL}"
  echo "- Training examples file: ${TRAIN_SUBSET}"
  echo "- Training subset manifest: ${TRAIN_SUBSET_MANIFEST}"
  echo "- Target steps this job: ${TRAIN_TARGET_STEPS}"
  echo "- Final steps per variant: ${TOTAL_STEPS_PER_VARIANT}"
  echo "- Prompt batch size: ${TRAIN_BATCH_SIZE}"
  echo "- Rollouts per prompt: ${ROLLOUT_N}"
  echo "- Effective trajectory batch: $((TRAIN_BATCH_SIZE * ROLLOUT_N))"
  echo "- PPO mini batch size: ${PPO_MINI_BATCH_SIZE}"
  echo "- Effective trajectory mini batch: $((PPO_MINI_BATCH_SIZE * ROLLOUT_N))"
  echo "- Max response length: ${MAX_RESPONSE_LENGTH}"
  echo "- Actor max token length per GPU: ${MAX_TOKEN_LEN_PER_GPU}"
  echo "- Rollout max num seqs: ${ROLLOUT_MAX_NUM_SEQS}"
  echo "- GPU memory utilization: ${GPU_MEM_UTIL}"
  echo "- Enforce eager rollout: ${ENFORCE_EAGER}"
  echo "- Checkpoint save frequency: every ${SAVE_FREQ} steps"
  echo "- Save HF model: ${SAVE_HF_MODEL}"
  echo "- Checkpoint save contents: ${CHECKPOINT_SAVE_CONTENTS:-default}"
  echo "- Checkpoint load contents: ${CHECKPOINT_LOAD_CONTENTS:-default}"
  echo "- Output dir: ${OUTPUT_DIR}"
  echo
  echo "## Training logs"
  echo "- discrete_rl: ${LOG_DIR}/discrete_rl_train.log"
  echo "- multiplex_thinking: ${LOG_DIR}/multiplex_thinking_train.log"
  echo "- shared4_joint: ${LOG_DIR}/shared4_joint_train.log"
  echo "- shared4_thinking_only: ${LOG_DIR}/shared4_thinking_only_train.log"
  echo "- shared4_answer_only: ${LOG_DIR}/shared4_answer_only_train.log"
  echo
  echo "## Training plots"
  find "$PLOT_DIR" -maxdepth 1 -type f | sort | sed 's#^#- #'
} > "$SUMMARY_PATH"

echo "[summary] ${SUMMARY_PATH}"
