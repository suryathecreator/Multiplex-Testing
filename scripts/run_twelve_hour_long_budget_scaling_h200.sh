#!/bin/bash
#SBATCH --job-name=mux12h-long
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --output=/gscratch/scrubbed/suryadv/repos/Multiplex-Testing/slurm_logs/%x-%j.out
#SBATCH --chdir=/gscratch/scrubbed/suryadv/repos/Multiplex-Testing

set -euo pipefail

START_TIME="$(date +%s)"
REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

SCRATCH_ROOT="${SCRATCH_ROOT:-/gscratch/scrubbed/suryadv}"
SCRATCH_CACHE_ROOT="${SCRATCH_ROOT}/.cache/multiplex-thinking"
SCRATCH_TMP_ROOT="${SCRATCH_ROOT}/tmp/multiplex-thinking-twelve-hour-long-${SLURM_JOB_ID:-local}"
SCRATCH_RUNTIME_ROOT="${SCRATCH_CACHE_ROOT}/runtime-envs"
JOB_OVERLAY_DIR="${SCRATCH_TMP_ROOT}/job-overlay"
JOB_BIN_DIR="${SCRATCH_TMP_ROOT}/bin"
SELECTED_RUNTIME_MANIFEST="${SCRATCH_TMP_ROOT}/runtime-manifest.json"
LOCAL_PYTHONPATH="${PWD}/verl-latest:${PWD}/sglang-0.4.9.post6:${PWD}/transformers-4.54.0/src"
ENV_PREFIX="/mmfs1/home/suryadv/.conda/envs/multiplex-thinking"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

mkdir -p slurm_logs final_eval_outputs
mkdir -p "$SCRATCH_CACHE_ROOT" "$SCRATCH_TMP_ROOT" "$SCRATCH_RUNTIME_ROOT" "$JOB_OVERLAY_DIR" "$JOB_BIN_DIR"

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

ensure_host_compiler
ensure_cuda_nvcc

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VERL_DISABLE_EXPANDABLE_SEGMENTS=1

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

mkdir -p "$RAY_TMPDIR"

BOOTSTRAP_LOG="${SCRATCH_CACHE_ROOT}/bootstrap-runtime-twelve-hour-long-${SLURM_JOB_ID:-local}.log"
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
if [[ -n "$LIGHTWEIGHT_OVERLAY_CACHE_DIR" ]]; then
  export PYTHONPATH="${MANAGED_RUNTIME_SITE}:${JOB_OVERLAY_DIR}:${LIGHTWEIGHT_OVERLAY_CACHE_DIR}:${LOCAL_PYTHONPATH}"
  export PATH="${JOB_OVERLAY_DIR}/bin:${LIGHTWEIGHT_OVERLAY_CACHE_DIR}/bin:${JOB_BIN_DIR}:${PATH}"
else
  export PYTHONPATH="${MANAGED_RUNTIME_SITE}:${JOB_OVERLAY_DIR}:${LOCAL_PYTHONPATH}"
  export PATH="${JOB_OVERLAY_DIR}/bin:${JOB_BIN_DIR}:${PATH}"
fi

echo "[setup] python=${PYTHON_BIN}"
echo "[setup] pythonpath=${PYTHONPATH}"
echo "[setup] slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
echo "[setup] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

resolve_local_deepseek_1p5b_model() {
  local local_model_dir="${SCRATCH_CACHE_ROOT}/local-models/DeepSeek-R1-Distill-Qwen-1.5B"
  local metadata_snapshot="${TRANSFORMERS_CACHE}/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
  local weights_snapshot="${HF_HOME}/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

  if [[ ! -f "${metadata_snapshot}/config.json" || ! -f "${weights_snapshot}/model.safetensors" ]]; then
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
  if MODEL_PATH="$(resolve_local_deepseek_1p5b_model)"; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
  else
    MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  fi
fi

RUN_STAMP="${RUN_TAG:-twelve-hour-long-budget-scaling-${SLURM_JOB_ID:-local}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/final_eval_outputs/${RUN_STAMP}}"
DATA_DIR="${OUTPUT_DIR}/data"
LOG_DIR="${OUTPUT_DIR}/logs"
CHECKPOINT_ROOT="${OUTPUT_DIR}/checkpoints"
EVAL_ROOT="${OUTPUT_DIR}/eval"
PLOT_DIR="${OUTPUT_DIR}/training_plots"
mkdir -p "$DATA_DIR" "$LOG_DIR" "$CHECKPOINT_ROOT" "$EVAL_ROOT" "$PLOT_DIR"

TRAIN_EXAMPLE_COUNT="${TRAIN_EXAMPLES:-128}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-48}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
EVAL_MAX_K="${EVAL_MAX_K:-8}"
EVAL_MAX_PROMPTS="${EVAL_MAX_PROMPTS:-30}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-5120}"
REASONING_PREFIX_TOKENS="${REASONING_PREFIX_TOKENS:-4096}"
EVAL_SEEDS_CSV="${EVAL_SEEDS_CSV:-26010808,26010809,26010810}"
TRAIN_SUBSET="${DATA_DIR}/deepscaler_train_subset${TRAIN_EXAMPLE_COUNT}.parquet"
TRAIN_SUBSET_MANIFEST="${DATA_DIR}/deepscaler_train_subset${TRAIN_EXAMPLE_COUNT}_manifest.json"
"$PYTHON_BIN" scripts/create_deepscaler_subset.py \
  --input deepscaler/hdfs_data/train.parquet \
  --output "$TRAIN_SUBSET" \
  --manifest "$TRAIN_SUBSET_MANIFEST" \
  --num-examples "$TRAIN_EXAMPLE_COUNT" \
  --seed "${TRAIN_SUBSET_SEED:-26010808}"

TRAIN_COMMON_ARGS=(
  --model "$MODEL_PATH"
  --train_file "$TRAIN_SUBSET"
  --val_file deepscaler/hdfs_data/aime.parquet
  --n_gpus_per_node 2
  --total_training_steps "$TOTAL_TRAINING_STEPS"
  --train_batch_size 8
  --rollout_n 4
  --rollout_max_num_seqs 192
  --max_prompt_length 1024
  --max_response_length 4096
  --max_token_len_per_gpu "$MAX_TOKEN_LEN_PER_GPU"
  --save_freq "$TOTAL_TRAINING_STEPS"
  --test_freq -1
  --val_before_train False
  --val_rollout_n 4
  --val_batch_size 64
  --logger "['console']"
  --save_hf_model True
  --enforce_eager True
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --gpu_mem_util 0.78
  --top_p 1.0
  --temp 1.0
  --enable_unweighting False
  --wandb_project twelve-hour-long-budget-scaling
)

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}

run_training() {
  local condition="$1"
  local enable_soft_think="$2"
  local loss_mode="$3"
  local multiplex_width="$4"
  local condition_checkpoint_dir="${CHECKPOINT_ROOT}/${condition}"
  local condition_log="${LOG_DIR}/${condition}_train.log"

  echo "[train] starting condition=${condition}"
  stop_ray
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export OMPI_COMM_WORLD_RANK=0
  set +e
  bash scripts/train.sh \
    "${TRAIN_COMMON_ARGS[@]}" \
    --exp_name "$condition" \
    --default_local_dir "$condition_checkpoint_dir" \
    --enable_soft_think "$enable_soft_think" \
    --loss_mode "$loss_mode" \
    --multiplex_width "$multiplex_width" \
    2>&1 | tee "$condition_log"
  local train_status="${PIPESTATUS[0]}"
  set -e
  stop_ray
  if [[ "$train_status" -ne 0 ]]; then
    echo "[train] condition=${condition} failed with status=${train_status}" >&2
    exit "$train_status"
  fi
  echo "[train] finished condition=${condition} log=${condition_log}"
}

find_hf_checkpoint() {
  local checkpoint_dir="$1"
  find "$checkpoint_dir" -type f -path "*/actor/huggingface/config.json" -printf "%h\n" 2>/dev/null | sort -V | tail -n 1 || true
}

if [[ -n "${PRETRAINED_DISCRETE_HF_CHECKPOINT:-}" ]]; then
  DISCRETE_HF_CHECKPOINT="${PRETRAINED_DISCRETE_HF_CHECKPOINT}"
  echo "[checkpoint] reusing pretrained discrete_rl=${DISCRETE_HF_CHECKPOINT}"
else
  run_training discrete_rl False vanilla 1
  DISCRETE_HF_CHECKPOINT="$(find_hf_checkpoint "${CHECKPOINT_ROOT}/discrete_rl")"
fi
if [[ -z "$DISCRETE_HF_CHECKPOINT" || ! -f "${DISCRETE_HF_CHECKPOINT}/config.json" ]]; then
  echo "[checkpoint] missing discrete_rl HF checkpoint under ${CHECKPOINT_ROOT}/discrete_rl" >&2
  exit 1
fi
if [[ -n "${PRETRAINED_DISCRETE_TRAIN_LOG:-}" && -f "${PRETRAINED_DISCRETE_TRAIN_LOG}" && ! -f "${LOG_DIR}/discrete_rl_train.log" ]]; then
  cp "${PRETRAINED_DISCRETE_TRAIN_LOG}" "${LOG_DIR}/discrete_rl_train.log"
  echo "[train] copied pretrained discrete_rl log=${LOG_DIR}/discrete_rl_train.log"
fi
echo "[checkpoint] discrete_rl=${DISCRETE_HF_CHECKPOINT}"

run_training multiplex_thinking True multiplex_thinking 3
MULTIPLEX_HF_CHECKPOINT="$(find_hf_checkpoint "${CHECKPOINT_ROOT}/multiplex_thinking")"
if [[ -z "$MULTIPLEX_HF_CHECKPOINT" || ! -f "${MULTIPLEX_HF_CHECKPOINT}/config.json" ]]; then
  echo "[checkpoint] missing multiplex_thinking HF checkpoint under ${CHECKPOINT_ROOT}/multiplex_thinking" >&2
  exit 1
fi
echo "[checkpoint] multiplex_thinking=${MULTIPLEX_HF_CHECKPOINT}"

common_eval_args() {
  local model_path="$1"
  local seed="$2"
  local port="$3"
  echo \
    --model "$model_path" \
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
    --seed "$seed" \
    --compact-jsonl \
    --resume
}

run_passk_eval() {
  local condition="$1"
  local model_path="$2"
  local methods="$3"
  local seed="$4"
  local port="$5"
  local eval_dir="$6"

  mkdir -p "$eval_dir"
  echo "[eval] starting condition=${condition} methods=${methods} seed=${seed} model=${model_path}"
  "$PYTHON_BIN" scripts/compare_passk_aime.py \
    --experiment-mode passk_sweep \
    --methods "$methods" \
    $(common_eval_args "$model_path" "$seed" "$port") \
    --reasoning-prefix-token-values "$REASONING_PREFIX_TOKENS" \
    --output-dir "$eval_dir"
  if [[ ! -f "${eval_dir}/summary_overall.md" ]]; then
    echo "[eval] missing summary_overall.md for ${condition}" >&2
    exit 1
  fi
  echo "[eval] finished condition=${condition} summary=${eval_dir}/summary_overall.md"
}

run_branch_shared4_eval() {
  local seed="$1"
  local port="$2"
  local eval_dir="$3"

  mkdir -p "$eval_dir"
  echo "[eval] starting multiplex branch_ablation shared4 seed=${seed}"
  "$PYTHON_BIN" scripts/compare_passk_aime.py \
    --experiment-mode branch_ablation \
    --methods baseline,shared_trace \
    $(common_eval_args "$MULTIPLEX_HF_CHECKPOINT" "$seed" "$port") \
    --branch-ablation-reasoning-prefix-tokens "$REASONING_PREFIX_TOKENS" \
    --branch-ablation-group-sizes 4 \
    --output-dir "$eval_dir"
  if [[ ! -f "${eval_dir}/summary_ablation.md" ]]; then
    echo "[eval] missing summary_ablation.md for shared4 branch eval" >&2
    exit 1
  fi
  echo "[eval] finished shared4 branch eval summary=${eval_dir}/summary_ablation.md"
}

run_shared4_shared2_topup_eval() {
  local seed="$1"
  local port="$2"
  local source_root="$3"
  local eval_dir="$4"

  mkdir -p "$eval_dir"
  echo "[eval] starting shared4 + shared2 memory-match top-up seed=${seed}"
  "$PYTHON_BIN" scripts/compare_passk_aime.py \
    --experiment-mode memory_match_topup \
    --methods baseline,shared_trace \
    $(common_eval_args "$MULTIPLEX_HF_CHECKPOINT" "$seed" "$port") \
    --branch-ablation-reasoning-prefix-tokens "$REASONING_PREFIX_TOKENS" \
    --memory-match-source-root "$source_root" \
    --memory-match-shared-groups 4 \
    --memory-match-topup-generator shared_group \
    --memory-match-topup-shared-group-size 2 \
    --output-dir "$eval_dir"
  if [[ ! -f "${eval_dir}/summary_memory_match.md" ]]; then
    echo "[eval] missing summary_memory_match.md for shared4/shared2 top-up" >&2
    exit 1
  fi
  echo "[eval] finished shared4/shared2 top-up summary=${eval_dir}/summary_memory_match.md"
}

IFS=',' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_CSV"
repeat_index=0
for seed in "${EVAL_SEEDS[@]}"; do
  repeat_label="$(printf 'repeat_%02d' "$repeat_index")"
  repeat_dir="${EVAL_ROOT}/${repeat_label}"
  port_base="$((36100 + repeat_index * 20))"
  mkdir -p "$repeat_dir"
  echo "[eval] repeat=${repeat_label} seed=${seed} output=${repeat_dir}"

  run_passk_eval \
    discrete_cot_untrained \
    "$MODEL_PATH" \
    standard_generation \
    "$seed" \
    "$((port_base + 1))" \
    "${repeat_dir}/discrete_cot_untrained"

  run_passk_eval \
    discrete_rl \
    "$DISCRETE_HF_CHECKPOINT" \
    standard_generation \
    "$seed" \
    "$((port_base + 2))" \
    "${repeat_dir}/discrete_rl"

  run_passk_eval \
    multiplex_thinking_fixed \
    "$MULTIPLEX_HF_CHECKPOINT" \
    baseline \
    "$seed" \
    "$((port_base + 3))" \
    "${repeat_dir}/multiplex_thinking_fixed"

  run_branch_shared4_eval \
    "$seed" \
    "$((port_base + 4))" \
    "${repeat_dir}/multiplex_branch_shared4"

  run_shared4_shared2_topup_eval \
    "$seed" \
    "$((port_base + 5))" \
    "${repeat_dir}/multiplex_branch_shared4" \
    "${repeat_dir}/multiplex_shared4_shared2_topup"

  repeat_index="$((repeat_index + 1))"
done

"$PYTHON_BIN" scripts/plot_training_logs.py \
  --run "discrete_rl=${LOG_DIR}/discrete_rl_train.log" \
  --run "multiplex_thinking=${LOG_DIR}/multiplex_thinking_train.log" \
  --output-dir "$PLOT_DIR" \
  --summary-json "${PLOT_DIR}/training_plot_summary.json"

"$PYTHON_BIN" scripts/aggregate_four_hour_scaling.py \
  --eval-root "$EVAL_ROOT" \
  --output-dir "${OUTPUT_DIR}/final_scaling_plots" \
  --max-k "$EVAL_MAX_K"

END_TIME="$(date +%s)"
WALL_SECONDS="$((END_TIME - START_TIME))"
SUMMARY_PATH="${OUTPUT_DIR}/twelve_hour_long_budget_scaling_summary.md"
{
  echo "# Twelve-Hour Long-Budget Multiplex Scaling Experiment"
  echo
  echo "- Run stamp: ${RUN_STAMP}"
  echo "- Wall time seconds: ${WALL_SECONDS}"
  echo "- Model: ${MODEL_PATH}"
  echo "- Hardware: 2x H200"
  echo "- Training subset manifest: ${TRAIN_SUBSET_MANIFEST}"
  echo "- Training examples: ${TRAIN_EXAMPLE_COUNT}"
  echo "- Training steps per condition: ${TOTAL_TRAINING_STEPS}"
  echo "- Training max response length: 4096"
  echo "- Training actor max token length per GPU: ${MAX_TOKEN_LEN_PER_GPU}"
  echo "- KL penalty: 0.0 (use_kl_loss=False, kl_loss_coef=0.0, algorithm.kl_ctrl.kl_coef=0.0)"
  echo "- Entropy penalty: 0.0"
  echo "- Eval benchmark: AIME 2024"
  echo "- Eval max prompts: ${EVAL_MAX_PROMPTS}"
  echo "- Eval seeds: ${EVAL_SEEDS_CSV}"
  echo "- Eval max k: ${EVAL_MAX_K}"
  echo "- Eval max new tokens: ${EVAL_MAX_NEW_TOKENS}"
  echo "- Eval reasoning prefix tokens: ${REASONING_PREFIX_TOKENS}"
  echo "- Discrete RL checkpoint: ${DISCRETE_HF_CHECKPOINT}"
  echo "- Multiplex Thinking checkpoint: ${MULTIPLEX_HF_CHECKPOINT}"
  echo
  echo "## Training Plots"
  if [[ -f "${PLOT_DIR}/training_plot_summary.json" ]]; then
    echo "- Plot summary: ${PLOT_DIR}/training_plot_summary.json"
    find "$PLOT_DIR" -maxdepth 1 -name '*.png' -print | sort | sed 's#^#- #'
  else
    echo "- Plot summary missing."
  fi
  echo
  echo "## Final Scaling Plots"
  find "${OUTPUT_DIR}/final_scaling_plots" -maxdepth 1 -type f | sort | sed 's#^#- #'
  echo
  echo "## Eval Repeats"
  for repeat_dir in "${EVAL_ROOT}"/repeat_*; do
    [[ -d "$repeat_dir" ]] || continue
    echo "- $(basename "$repeat_dir")"
    echo "  - discrete CoT untrained: ${repeat_dir}/discrete_cot_untrained/summary_overall.md"
    echo "  - discrete RL trained: ${repeat_dir}/discrete_rl/summary_overall.md"
    echo "  - multiplex fixed independent: ${repeat_dir}/multiplex_thinking_fixed/summary_overall.md"
    echo "  - shared4 base: ${repeat_dir}/multiplex_branch_shared4/summary_ablation.md"
    echo "  - shared4 + shared2 top-up: ${repeat_dir}/multiplex_shared4_shared2_topup/summary_memory_match.md"
  done
} > "$SUMMARY_PATH"

echo "[done] output_dir=${OUTPUT_DIR}"
echo "[done] summary=${SUMMARY_PATH}"
