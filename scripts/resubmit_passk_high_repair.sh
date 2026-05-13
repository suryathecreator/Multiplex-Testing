#!/bin/bash
set -euo pipefail

REPO_ROOT="/gscratch/scrubbed/suryadv/repos/Multiplex-Testing"
cd "$REPO_ROOT"

PASSK_DIR="${PASSK_DIR:-${REPO_ROOT}/final_eval_outputs/passk-memory-raivn-20260424-101750}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"
mkdir -p "$PASSK_DIR" slurm_logs

SBATCH_COMMON=(
  --account=raivn-ckpt
  --partition=ckpt-all
  --cpus-per-task=4
  --export=ALL
)

echo "[repair] passk_dir=${PASSK_DIR}"
echo "[repair] repairing high prefixes 4096,6144 only"

repair_job="$(
  env \
    BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
    BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
    MPLCONFIGDIR=/tmp \
    PORT=30104 \
    EXPERIMENT_MODE=passk_sweep \
    RUN_TAG=raivn-passk-high-repair-20260426 \
    OUTPUT_DIR="$PASSK_DIR" \
    MAX_K=64 \
    MAX_PROMPTS=30 \
    METHODS=baseline,shared_trace \
    REASONING_PREFIX_TOKEN_VALUES=4096,6144 \
    sbatch --parsable \
      --job-name=passk-high-r \
      --gres="$SLURM_GPU_GRES" \
      --constraint="$SLURM_GPU_CONSTRAINT" \
      --time=96:00:00 \
      --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
      "${SBATCH_COMMON[@]}" \
      --wrap="cd '${REPO_ROOT}' && bash run.sh && /mmfs1/home/suryadv/.conda/envs/multiplex-thinking/bin/python - <<'PY'
import csv
from pathlib import Path

root = Path('${PASSK_DIR}')
missing = []
for prefix in (4096, 6144):
    summary = root / f'prefix_{prefix:04d}' / 'summary.csv'
    if not summary.exists():
        missing.append(str(summary))
        continue
    rows = list(csv.DictReader(summary.open(newline='', encoding='utf-8')))
    if not rows:
        missing.append(f'{summary}: no rows')
        continue
    bad = [row for row in rows if int(float(row.get('num_prompts', 0))) < 30]
    if bad:
        missing.append(f'{summary}: incomplete num_prompts')
if missing:
    raise SystemExit('passk-high repair incomplete: ' + '; '.join(missing))
print('passk-high repair completeness check passed')
PY"
)"
repair_job="${repair_job%%;*}"
echo "[repair] passk-high-r: ${repair_job}"
