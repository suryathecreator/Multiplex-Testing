#!/bin/bash
set -euo pipefail

REPO_ROOT="/gscratch/scrubbed/suryadv/repos/Multiplex-Testing"
cd "$REPO_ROOT"

PASSK_DIR="${PASSK_DIR:-${REPO_ROOT}/final_eval_outputs/passk-memory-raivn-20260424-101750}"
AB32_ROOT="${AB32_ROOT:-${REPO_ROOT}/final_eval_outputs/passk-ablation32-raivn-20260425-022604}"
AB32_2048_DIR="${AB32_2048_DIR:-${AB32_ROOT}/budget_2048}"

AB32_1024_JOB="${AB32_1024_JOB:-34834764}"
AB32_4096_JOB="${AB32_4096_JOB:-34834766}"
STALE_AB32_PLOT_JOBS="${STALE_AB32_PLOT_JOBS:-34834767 34844428 34844488}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_PLOT_GPU_GRES="${SLURM_PLOT_GPU_GRES:-gpu:1}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"

mkdir -p "$PASSK_DIR" "$AB32_2048_DIR" slurm_logs

SBATCH_COMMON=(
  --account=raivn-ckpt
  --partition=ckpt-all
  --cpus-per-task=4
  --export=ALL
)

echo "[resubmit] passk_dir=${PASSK_DIR}"
echo "[resubmit] ab32_root=${AB32_ROOT}"
echo "[resubmit] ab32_2048_dir=${AB32_2048_DIR}"
echo "[resubmit] cancelling stale aggregate plot jobs ${STALE_AB32_PLOT_JOBS} if still pending"
for stale_plot_job in ${STALE_AB32_PLOT_JOBS}; do
  scancel "$stale_plot_job" 2>/dev/null || true
done

passk_low_job="$(
  env \
    BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
    BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
    MPLCONFIGDIR=/tmp \
    PORT=30101 \
    EXPERIMENT_MODE=passk_sweep \
    RUN_TAG=raivn-passk-low-resume-20260425 \
    OUTPUT_DIR="$PASSK_DIR" \
    MAX_K=64 \
    MAX_PROMPTS=30 \
    METHODS=baseline,shared_trace \
    REASONING_PREFIX_TOKEN_VALUES=256,512,1024 \
    sbatch --parsable \
      --job-name=passk-low-r \
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
for prefix in (256, 512, 1024):
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
    raise SystemExit('passk-low resume incomplete: ' + '; '.join(missing))
print('passk-low resume completeness check passed')
PY"
)"
passk_low_job="${passk_low_job%%;*}"
echo "[resubmit] passk-low-r: ${passk_low_job}"

ab32_2048_job="$(
  env \
    BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
    BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
    MPLCONFIGDIR=/tmp \
    PORT=30102 \
    EXPERIMENT_MODE=branch_ablation \
    RUN_TAG=raivn-ablation32-t2048-resume-20260425 \
    OUTPUT_DIR="$AB32_2048_DIR" \
    MAX_K=32 \
    MAX_PROMPTS=30 \
    METHODS=baseline,shared_trace \
    BRANCH_ABLATION_REASONING_PREFIX_TOKENS=2048 \
    BRANCH_ABLATION_GROUP_SIZES=2,4,8,16,32 \
    sbatch --parsable \
      --job-name=ab32-t2048-r \
      --gres="$SLURM_GPU_GRES" \
      --constraint="$SLURM_GPU_CONSTRAINT" \
      --time=96:00:00 \
      --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
      "${SBATCH_COMMON[@]}" \
      --wrap="cd '${REPO_ROOT}' && bash run.sh && /mmfs1/home/suryadv/.conda/envs/multiplex-thinking/bin/python - <<'PY'
import csv
from pathlib import Path

root = Path('${AB32_2048_DIR}')
summary = root / 'summary_ablation.csv'
if not summary.exists():
    raise SystemExit(f'missing {summary}')
rows = list(csv.DictReader(summary.open(newline='', encoding='utf-8')))
groups = {int(row['branch_group_size']) for row in rows}
expected_groups = {1, 2, 4, 8, 16, 32}
if groups != expected_groups:
    raise SystemExit(f'unexpected groups: {sorted(groups)}')
bad = [row for row in rows if int(float(row.get('num_prompts', 0))) < 30]
if bad:
    raise SystemExit('ab32 t2048 resume incomplete: some rows have num_prompts < 30')
print('ab32 t2048 resume completeness check passed')
PY"
)"
ab32_2048_job="${ab32_2048_job%%;*}"
echo "[resubmit] ab32-t2048-r: ${ab32_2048_job}"

plot_dependency="afterok:${AB32_1024_JOB}:${ab32_2048_job}:${AB32_4096_JOB}"
plot_job="$(
  sbatch --parsable \
    --job-name=ab32-plots-r \
    --gres="$SLURM_PLOT_GPU_GRES" \
    --constraint="$SLURM_GPU_CONSTRAINT" \
    --time=01:00:00 \
    --dependency="$plot_dependency" \
    --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
    "${SBATCH_COMMON[@]}" \
    --wrap="cd '${REPO_ROOT}' && MPLCONFIGDIR=/tmp /mmfs1/home/suryadv/.conda/envs/multiplex-thinking/bin/python scripts/plot_branch_ablation_budget_sweep.py --root '${AB32_ROOT}'"
)"
plot_job="${plot_job%%;*}"
echo "[resubmit] ab32-plots-r: ${plot_job} (${plot_dependency})"
