#!/bin/bash
set -euo pipefail

REPO_ROOT="/gscratch/scrubbed/suryadv/repos/Multiplex-Testing"
cd "$REPO_ROOT"

PASSK_DIR="${PASSK_DIR:-${REPO_ROOT}/final_eval_outputs/passk-memory-raivn-20260424-101750}"
AB32_ROOT="${AB32_ROOT:-${REPO_ROOT}/final_eval_outputs/passk-ablation32-raivn-20260425-022604}"
AB32_4096_DIR="${AB32_4096_DIR:-${AB32_ROOT}/budget_4096}"
AB32_1024_JOB="${AB32_1024_JOB:-34834764}"
AB32_2048_JOB="${AB32_2048_JOB:-34844505}"
STALE_AB32_PLOT_JOBS="${STALE_AB32_PLOT_JOBS:-34867016}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:2}"
SLURM_PLOT_GPU_GRES="${SLURM_PLOT_GPU_GRES:-gpu:1}"
SLURM_GPU_CONSTRAINT="${SLURM_GPU_CONSTRAINT:-a40|a100|l40|l40s|h200}"

mkdir -p "$PASSK_DIR" "$AB32_4096_DIR" slurm_logs

SBATCH_COMMON=(
  --account=raivn-ckpt
  --partition=ckpt-all
  --cpus-per-task=4
  --export=ALL
)

echo "[repair] passk_dir=${PASSK_DIR}"
echo "[repair] ab32_root=${AB32_ROOT}"
echo "[repair] cancelling stale plot jobs: ${STALE_AB32_PLOT_JOBS}"
for stale_plot_job in ${STALE_AB32_PLOT_JOBS}; do
  scancel "$stale_plot_job" 2>/dev/null || true
done

passk_high_job="$(
  env \
    BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
    BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
    MPLCONFIGDIR=/tmp \
    PORT=30114 \
    EXPERIMENT_MODE=passk_sweep \
    RUN_TAG=raivn-passk-high-targeted-repair-20260427 \
    OUTPUT_DIR="$PASSK_DIR" \
    MAX_K=64 \
    MAX_PROMPTS=30 \
    PROMPT_INDICES=0,2,8,9,10,12,14,29 \
    METHODS=baseline,shared_trace \
    REASONING_PREFIX_TOKEN_VALUES=4096,6144 \
    sbatch --parsable \
      --job-name=passk-high-r2 \
      --gres="$SLURM_GPU_GRES" \
      --constraint="$SLURM_GPU_CONSTRAINT" \
      --time=24:00:00 \
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
    raise SystemExit('passk-high targeted repair incomplete: ' + '; '.join(missing))
print('passk-high targeted repair completeness check passed')
PY"
)"
passk_high_job="${passk_high_job%%;*}"
echo "[repair] passk-high-r2: ${passk_high_job}"

ab32_4096_job="$(
  env \
    BOOTSTRAP_HELPER_TIMEOUT_SECONDS=900 \
    BOOTSTRAP_OVERLAY_CACHE_MAX_COPY_BYTES=268435456 \
    MPLCONFIGDIR=/tmp \
    PORT=30113 \
    EXPERIMENT_MODE=branch_ablation \
    RUN_TAG=raivn-ablation32-t4096-targeted-repair-20260427 \
    OUTPUT_DIR="$AB32_4096_DIR" \
    MAX_K=32 \
    MAX_PROMPTS=30 \
    PROMPT_INDICES=2,12 \
    METHODS=baseline,shared_trace \
    BRANCH_ABLATION_REASONING_PREFIX_TOKENS=4096 \
    BRANCH_ABLATION_GROUP_SIZES=2,4 \
    sbatch --parsable \
      --job-name=ab32-t4096-r2 \
      --gres="$SLURM_GPU_GRES" \
      --constraint="$SLURM_GPU_CONSTRAINT" \
      --time=12:00:00 \
      --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
      "${SBATCH_COMMON[@]}" \
      --wrap="cd '${REPO_ROOT}' && bash run.sh && /mmfs1/home/suryadv/.conda/envs/multiplex-thinking/bin/python - <<'PY'
import csv
from pathlib import Path

root = Path('${AB32_4096_DIR}')
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
    raise SystemExit('ab32 t4096 targeted repair incomplete: some rows have num_prompts < 30')
print('ab32 t4096 targeted repair completeness check passed')
PY"
)"
ab32_4096_job="${ab32_4096_job%%;*}"
echo "[repair] ab32-t4096-r2: ${ab32_4096_job}"

plot_dependency="afterany:${AB32_1024_JOB}:${AB32_2048_JOB}:${ab32_4096_job}"
plot_job="$(
  sbatch --parsable \
    --job-name=ab32-plots-r2 \
    --gres="$SLURM_PLOT_GPU_GRES" \
    --constraint="$SLURM_GPU_CONSTRAINT" \
    --time=01:00:00 \
    --dependency="$plot_dependency" \
    --output="${REPO_ROOT}/slurm_logs/%x-%j.out" \
    "${SBATCH_COMMON[@]}" \
    --wrap="set -euo pipefail && cd '${REPO_ROOT}' && MPLCONFIGDIR=/tmp /mmfs1/home/suryadv/.conda/envs/multiplex-thinking/bin/python - <<'PY'
import csv
from pathlib import Path

root = Path('${AB32_ROOT}')
expected_groups = {1, 2, 4, 8, 16, 32}
problems = []
for budget in (1024, 2048, 4096):
    summary = root / f'budget_{budget}' / 'summary_ablation.csv'
    if not summary.exists():
        problems.append(f'missing {summary}')
        continue
    rows = list(csv.DictReader(summary.open(newline='', encoding='utf-8')))
    groups = {int(row['branch_group_size']) for row in rows}
    if groups != expected_groups:
        problems.append(f'{summary}: groups={sorted(groups)}')
    if any(int(float(row.get('num_prompts', 0))) < 30 for row in rows):
        problems.append(f'{summary}: incomplete num_prompts')
if problems:
    raise SystemExit('ab32 plot precheck failed: ' + '; '.join(problems))
PY
MPLCONFIGDIR=/tmp /mmfs1/home/suryadv/.conda/envs/multiplex-thinking/bin/python scripts/plot_branch_ablation_budget_sweep.py --root '${AB32_ROOT}'"
)"
plot_job="${plot_job%%;*}"
echo "[repair] ab32-plots-r2: ${plot_job} (${plot_dependency})"
