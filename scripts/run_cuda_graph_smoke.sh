#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT"

if ! type module >/dev/null 2>&1; then
  # Batch shells on Hyak usually have Lmod already, but keep this robust.
  source /etc/profile.d/modules.sh 2>/dev/null || true
fi

gcc_major_version() {
  local version
  version="$(gcc -dumpfullversion -dumpversion 2>/dev/null || true)"
  version="${version%%$'\n'*}"
  version="${version%%.*}"
  if [[ "$version" =~ ^[0-9]+$ ]]; then
    echo "$version"
  else
    echo 0
  fi
}

select_modern_gcc() {
  local major
  if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    major="$(gcc_major_version)"
    if (( major >= 9 )); then
      return 0
    fi
  fi

  if ! type module >/dev/null 2>&1; then
    return 1
  fi

  local candidate
  for candidate in \
    gcc/13.2.0 gcc/13.1.0 gcc/12.3.0 gcc/12.2.0 gcc/12.1.0 \
    gcc/11.3.0 gcc/11.2.0 gcc/10.3.0 gcc/10.2.0 gcc/9.4.0 gcc/9.3.0 \
    gcc/13 gcc/12 gcc/11 gcc/10 gcc/9 \
    GCC/13.2.0 GCC/12.3.0 GCC/11.3.0 GCC/10.3.0 GCC/9.4.0; do
    if module load "$candidate" >/dev/null 2>&1; then
      if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
        major="$(gcc_major_version)"
        if (( major >= 9 )); then
          echo "[cuda-smoke] loaded compiler module ${candidate}"
          return 0
        fi
      fi
      module unload "$candidate" >/dev/null 2>&1 || true
    fi
  done

  return 1
}

if type module >/dev/null 2>&1; then
  select_modern_gcc || {
    echo "[cuda-smoke] GCC >= 9 is required for PyTorch/FlashInfer CUDA JIT." >&2
    echo "[cuda-smoke] Current gcc: $(command -v gcc || echo missing)" >&2
    command -v gcc >/dev/null 2>&1 && gcc --version || true
    module list || true
    exit 2
  }

  module load cuda/12.8.1 2>/dev/null || \
    module load cuda/12.8 2>/dev/null || \
    module load cuda/12.6 2>/dev/null || \
    module load cuda/12.4 2>/dev/null || \
    module load cuda/12.2 2>/dev/null || \
    module load cuda/12.1 2>/dev/null || \
    module load cuda 2>/dev/null || true
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[cuda-smoke] nvcc was not found after module load attempts; cannot test CUDA graph." >&2
  echo "[cuda-smoke] PATH=${PATH}" >&2
  type module >/dev/null 2>&1 && module list || true
  exit 2
fi

if ! command -v gcc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1 || (( "$(gcc_major_version)" < 9 )); then
  echo "[cuda-smoke] GCC >= 9 is required for CUDA graph smoke, but the active compiler is too old." >&2
  echo "[cuda-smoke] gcc=$(command -v gcc || echo missing)" >&2
  command -v gcc >/dev/null 2>&1 && gcc --version || true
  exit 2
fi

GCC_PATH="$(command -v gcc)"
GXX_PATH="$(command -v g++)"
GCC_BIN_DIR="$(dirname "$GCC_PATH")"
GCC_ROOT="$(dirname "$GCC_BIN_DIR")"
NVCC_PATH="$(command -v nvcc)"
CUDA_BIN_DIR="$(dirname "$NVCC_PATH")"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$CUDA_BIN_DIR")}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export CC="$GCC_PATH"
export CXX="$GXX_PATH"
export CUDAHOSTCXX="$GXX_PATH"
export CMAKE_CUDA_HOST_COMPILER="$GXX_PATH"
export NVCC_PREPEND_FLAGS="--compiler-bindir=${GCC_BIN_DIR} ${NVCC_PREPEND_FLAGS:-}"
export CUDACXX="$NVCC_PATH"
export PATH="${GCC_BIN_DIR}:${PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
if [[ -d "${GCC_ROOT}/lib64" ]]; then
  export LD_LIBRARY_PATH="${GCC_ROOT}/lib64:${LD_LIBRARY_PATH}"
fi
if [[ -d "${GCC_ROOT}/lib" ]]; then
  export LD_LIBRARY_PATH="${GCC_ROOT}/lib:${LD_LIBRARY_PATH}"
fi
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/gscratch/scrubbed/suryadv/tmp/multiplex-thinking-${SLURM_JOB_ID:-local}/flashinfer}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${FLASHINFER_WORKSPACE_BASE}/torch_extensions}"

echo "[cuda-smoke] gcc=${GCC_PATH}"
gcc --version
echo "[cuda-smoke] g++=${GXX_PATH}"
g++ --version
echo "[cuda-smoke] nvcc=${NVCC_PATH}"
nvcc --version
echo "[cuda-smoke] CUDA_HOME=${CUDA_HOME}"
echo "[cuda-smoke] CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "[cuda-smoke] NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS}"
echo "[cuda-smoke] FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}"

export EXPERIMENT_MODE="${EXPERIMENT_MODE:-branch_ablation}"
export BENCHMARK="${BENCHMARK:-deepscaler_aime_train}"
export RUN_TAG="${RUN_TAG:-cuda-graph-smoke}"
export MAX_PROMPTS="${MAX_PROMPTS:-2}"
export MAX_K="${MAX_K:-2}"
export METHODS="${METHODS:-baseline,shared_trace}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4608}"
export BRANCH_ABLATION_REASONING_PREFIX_TOKENS="${BRANCH_ABLATION_REASONING_PREFIX_TOKENS:-4096}"
export BRANCH_ABLATION_GROUP_SIZES="${BRANCH_ABLATION_GROUP_SIZES:-2}"
export BRANCH_ABLATION_NO_BASELINE="${BRANCH_ABLATION_NO_BASELINE:-0}"
export CHECKPOINT_MATCHED_PROMPTS_STEP="${CHECKPOINT_MATCHED_PROMPTS_STEP:-5}"
export COMPACT_JSONL="${COMPACT_JSONL:-1}"
export DP_SIZE="${DP_SIZE:-2}"
export TP_SIZE="${TP_SIZE:-1}"
export SERVER_TIMEOUT_SECONDS="${SERVER_TIMEOUT_SECONDS:-900}"
export FALLBACK_CONFIGS="${FALLBACK_CONFIGS:-64:4:8:aggressive;48:3:6:aggressive;32:2:4:aggressive;16:1:2:aggressive}"

echo "[cuda-smoke] output_dir=${OUTPUT_DIR:-unset}"
echo "[cuda-smoke] fallback_configs=${FALLBACK_CONFIGS}"
bash "${REPO_ROOT}/scripts/run_aime_job_with_fallback.sh"
