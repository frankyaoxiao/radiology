#!/bin/bash
# Smarter Round 2 driver: wait for "enough" CSVs, then act on whatever's available.
# Doesn't hang forever if some R1 variants get killed.

set -uo pipefail
cd /home/fxiao/misc/156
SUB_DIR="submissions/2026-05-11"
mkdir -p "${SUB_DIR}"

# Wait for at least N _va.csv files to land in submissions/2026-05-11/
# (any pattern omnirad_*_va.csv counts; covers killed-and-replaced experiments)
TARGET=10  # at least 10 of the original 13 (replacements add more)
TIMEOUT=$((9 * 3600))  # 9 hours absolute timeout

start_t=$(date +%s)
while true; do
  n=$(ls "${SUB_DIR}"/omnirad_*_va.csv 2>/dev/null | grep -v "_5split_mean\|_3family\|_3seed_mean" | wc -l)
  elapsed=$(( $(date +%s) - start_t ))
  echo "$(date -Iseconds): n_va_csvs=${n}/${TARGET}  elapsed=${elapsed}s" >&2
  if [ "${n}" -ge "${TARGET}" ] || [ "${elapsed}" -ge "${TIMEOUT}" ]; then
    break
  fi
  sleep 120
done

echo "" >&2
echo "=== Round 1 sufficient (n=${n} CSVs); summarizing ===" >&2
uv run python summarize_experiments.py > "${SUB_DIR}/round1_summary.txt" 2>&1 || true
cat "${SUB_DIR}/round1_summary.txt" >&2

# Identify the BEST val NMSE among landed variants
# Parse summarize output: rows with 0.xxxx values
winner=$(awk '
  /^[ \t]*[0-9]+  / {
    name=$2
    val=$4 + 0
    if (name ~ /^(aug|focal|mse|pseudo)_/ && val > 0.5 && val < 1.0) {
      if (best == 0 || val < best) { best = val; w = name; bv = val }
    }
  }
  END { print w; print bv > "/dev/stderr" }
' "${SUB_DIR}/round1_summary.txt" 2>&1)

# More robust: search slurm logs for best val per experiment
best_var=""
best_val=99
for f in "${SUB_DIR}"/omnirad_*_va.csv; do
  short=$(basename "$f" _va.csv | sed 's/^omnirad_//')
  # Map short → full variant name
  case "$short" in
    aug_rlight) variant="aug_randlight" ;;
    aug_rstd)   variant="aug_randstd" ;;
    aug_rstrong) variant="aug_randstrong" ;;
    aug_triv)   variant="aug_trivial" ;;
    *) variant="$short" ;;
  esac
  log=""
  for sf in $(ls -t /home/fxiao/misc/156/slurm_logs/*.out 2>/dev/null); do
    if grep -q "config     : configs/v1_3class_omnirad_b14_s0_${variant}.yaml" "$sf" 2>/dev/null; then
      log="$sf"; break
    fi
  done
  [ -z "$log" ] && continue
  val=$(grep "new best" "$log" 2>/dev/null | tail -1 | grep -oE "nmse=[0-9.]+" | cut -d= -f2)
  [ -z "$val" ] && continue
  echo "  ${variant}: val=${val}" >&2
  if (( $(echo "${val} < ${best_val}" | bc -l) )); then
    best_val=${val}
    best_var=${variant}
  fi
done

if [ -z "${best_var:-}" ]; then
  echo "FATAL: no variant identified" >&2
  exit 1
fi

echo "" >&2
echo "=== WINNER: ${best_var}  val=${best_val} ===" >&2

# Save winner info for downstream
echo "${best_var}" > "${SUB_DIR}/round1_winner.txt"
echo "${best_val}" >> "${SUB_DIR}/round1_winner.txt"

# Generate 4 multi-split configs of the winner
winner_cfg="configs/v1_3class_omnirad_b14_s0_${best_var}.yaml"
declare -a r2_short_names=()
for seed in 7 13 29 101; do
  short_name="${best_var}_s${seed}"
  out_cfg="configs/v1_3class_omnirad_b14_s0_${short_name}.yaml"
  sed -e "s|^split_seed:.*|split_seed: ${seed}|" \
      -e "s|^run_name:.*|run_name: v1_3class_omnirad_b14_s0_${short_name}|" \
      "${winner_cfg}" > "${out_cfg}"
  train_id=$(sbatch -J "infer_${short_name}" slurm/train_single.sh "${out_cfg}" | awk '{print $4}')
  pred_id=$(sbatch --dependency=afterok:${train_id} \
    -J "pred_${short_name}" \
    slurm/infer_and_va.sh \
      "/data/artifacts/frank/misc/runs/v1_3class_omnirad_b14_s0_${short_name}/ckpts/ckpt_best.pt" \
      "${SUB_DIR}/omnirad_${short_name}" \
    | awk '{print $4}')
  echo "  R2 ${short_name}: train ${train_id}, pred ${pred_id}" >&2
  r2_short_names+=($short_name)
done

# Wait for all 4 R2 CSVs
for short_name in "${r2_short_names[@]}"; do
  until [ -f "${SUB_DIR}/omnirad_${short_name}_va.csv" ]; do
    sleep 90
  done
  echo "  $(date -Iseconds): omnirad_${short_name}_va.csv ready" >&2
done

# Final assembly
echo "" >&2
echo "=== Assembling final ensemble ===" >&2

uv run python -c "
import pandas as pd
import numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

winner = '${best_var}'
short_map = {
    'aug_randlight': 'aug_rlight', 'aug_randstd': 'aug_rstd',
    'aug_randstrong': 'aug_rstrong', 'aug_trivial': 'aug_triv',
}
s0_short = short_map.get(winner, winner)
om_csvs = [f'{SUB11}/omnirad_{s0_short}.csv'] + [f'{SUB11}/omnirad_{winner}_s{s}.csv' for s in [7, 13, 29, 101]]

oms = [pd.read_csv(p).sort_values('Id').reset_index(drop=True) for p in om_csvs]
ids = oms[0]['Id'].to_numpy()
for d in oms[1:]:
    assert (d['Id'].to_numpy() == ids).all()
om_pool = pd.DataFrame({'Id': ids})
for lab in LABELS:
    om_pool[lab] = np.mean([d[lab].to_numpy() for d in oms], axis=0)
om_pool.to_csv(f'{SUB11}/omnirad_{winner}_5split_mean.csv', index=False, float_format='%.6f')
print(f'wrote omnirad_{winner}_5split_mean.csv')

hp_pool = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cn      = pd.read_csv(f'{SUB10}/dinov3_cnxl.csv').sort_values('Id').reset_index(drop=True)

ens = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens[lab] = 0.4*om_pool[lab].to_numpy() + 0.4*hp_pool[lab].to_numpy() + 0.2*cn[lab].to_numpy()
ens.to_csv(f'{SUB11}/ensemble_winner_3family_w442.csv', index=False, float_format='%.6f')
print('wrote ensemble_winner_3family_w442.csv')
"

uv run python view_average.py \
  --in "${SUB_DIR}/ensemble_winner_3family_w442.csv" \
  --out "${SUB_DIR}/ensemble_winner_3family_w442_va.csv" --force

echo "" >&2
echo "=== ROUND 2 COMPLETE — final CSV: ${SUB_DIR}/ensemble_winner_3family_w442_va.csv ===" >&2
ls -la "${SUB_DIR}/" >&2
