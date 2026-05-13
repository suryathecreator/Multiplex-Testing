#!/bin/bash
set -u -o pipefail

REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/suryadv/repos/Multiplex-Testing}"
cd "$REPO_ROOT" || exit 1

FALLBACK_CONFIGS="${FALLBACK_CONFIGS:-16:1:2:aggressive}"
SERVER_TIMEOUT_SECONDS="${SERVER_TIMEOUT_SECONDS:-900}"
export SERVER_TIMEOUT_SECONDS

IFS=';' read -r -a CONFIGS <<< "$FALLBACK_CONFIGS"
last_status=0
attempt=0

for config in "${CONFIGS[@]}"; do
  if [[ -z "$config" ]]; then
    continue
  fi
  attempt=$((attempt + 1))
  IFS=':' read -r request_batch_size prompts_per_rank max_concurrent_prompts throughput_profile <<< "$config"
  request_batch_size="${request_batch_size:-16}"
  prompts_per_rank="${prompts_per_rank:-1}"
  max_concurrent_prompts="${max_concurrent_prompts:-$((2 * prompts_per_rank))}"
  throughput_profile="${throughput_profile:-aggressive}"

  export REQUEST_BATCH_SIZE="$request_batch_size"
  export PROMPTS_PER_RANK="$prompts_per_rank"
  export MAX_CONCURRENT_PROMPTS="$max_concurrent_prompts"
  export THROUGHPUT_PROFILE="$throughput_profile"

  echo "[fallback] attempt=${attempt} config=${config} request_batch_size=${REQUEST_BATCH_SIZE} prompts_per_rank=${PROMPTS_PER_RANK} max_concurrent_prompts=${MAX_CONCURRENT_PROMPTS} throughput_profile=${THROUGHPUT_PROFILE} server_timeout_seconds=${SERVER_TIMEOUT_SECONDS}"
  bash "${REPO_ROOT}/run.sh"
  last_status=$?
  if [[ "$last_status" -eq 0 ]]; then
    echo "[fallback] attempt=${attempt} succeeded"
    exit 0
  fi

  echo "[fallback] attempt=${attempt} failed with exit_status=${last_status}; trying next fallback config if available"
  sleep 45
done

echo "[fallback] all configs failed; final_exit_status=${last_status}" >&2
exit "$last_status"
