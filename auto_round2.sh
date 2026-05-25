#!/bin/bash
# Autonomous Round 2 driver:
# 1. Wait for all 13 Round 1 _va.csv files to land
# 2. Run summarize_experiments.py, write to submissions/2026-05-11/round1_summary.txt
# 3. Identify winner (lowest val NMSE) — also pick top 2
# 4. For winner: generate 4 multi-split configs (seeds 7/13/29/101), submit train+pred+VA
# 5. When all 4 multi-split finish: assemble ensemble (5-split pool of winner + H+ pool + ConvNeXt-L), VA, output

set -euo pipefail
cd /home/fxiao/misc/156

R1_VARIANTS=(
  "aug_randlight aug_rlight"
  "aug_randstd aug_rstd"
  "aug_randstrong aug_rstrong"
  "aug_trivial aug_triv"
  "focal_g1 focal_g1"
  "focal_g2 focal_g2"
  "focal_g3 focal_g3"
  "mse_w30 mse_w30"
  "mse_w50 mse_w50"
  "mse_w70 mse_w70"
  "pseudo_t03 pseudo_t03"
  "pseudo_t05 pseudo_t05"
  "pseudo_t07 pseudo_t07"
)

SUB_DIR="submissions/2026-05-11"
mkdir -p "${SUB_DIR}"

echo "=== Waiting for Round 1 CSVs (13 _va.csv files) ===" >&2
# Wait for each CSV separately
for entry in "${R1_VARIANTS[@]}"; do
  short=$(echo "$entry" | awk '{print $2}')
  until [ -f "${SUB_DIR}/omnirad_${short}_va.csv" ]; do
    sleep 60
  done
  echo "  $(date -Iseconds): omnirad_${short}_va.csv ready" >&2
done

echo "=== All Round 1 CSVs landed; running summarize ===" >&2
uv run python summarize_experiments.py > "${SUB_DIR}/round1_summary.txt" 2>&1
echo "" >&2
echo "=== Round 1 summary ===" >&2
cat "${SUB_DIR}/round1_summary.txt" >&2

# Identify winner (lowest val NMSE among the 13 variants)
# Parse summarize output: rank lines have "rank  variant  state  val_nmse  ..."
winner=$(awk '
  /^[ \t]*[0-9]+  / && $2 ~ /^(aug|focal|mse|pseudo)_/ && $4 ~ /^0\./ {
    val = $4 + 0
    if (val > 0 && (best == 0 || val < best)) { best = val; w = $2 }
  }
  END { print w }
' "${SUB_DIR}/round1_summary.txt")

if [ -z "${winner:-}" ]; then
  echo "FATAL: could not identify winner from round 1 summary" >&2
  exit 1
fi
echo "" >&2
echo "=== Round 1 winner: ${winner} ===" >&2

# Read winner's overrides from its config (relative to baseline)
winner_cfg="configs/v1_3class_omnirad_b14_s0_${winner}.yaml"

# Generate 4 multi-split configs
declare -a r2_train_ids=()
declare -a r2_short_names=()
for seed in 7 13 29 101; do
  short_name="${winner}_s${seed}"
  out_cfg="configs/v1_3class_omnirad_b14_s0_${short_name}.yaml"
  # Substitute split_seed and run_name
  sed -e "s|^split_seed:.*|split_seed: ${seed}|" \
      -e "s|^run_name:.*|run_name: v1_3class_omnirad_b14_s0_${short_name}|" \
      "${winner_cfg}" > "${out_cfg}"
  echo "  generated ${out_cfg}" >&2

  # Submit train + chained pred+VA
  train_id=$(sbatch -J "infer_${short_name}" slurm/train_single.sh "${out_cfg}" | awk '{print $4}')
  pred_id=$(sbatch --dependency=afterok:${train_id} \
    -J "pred_${short_name}" \
    slurm/infer_and_va.sh \
      "/data/artifacts/frank/misc/runs/v1_3class_omnirad_b14_s0_${short_name}/ckpts/ckpt_best.pt" \
      "${SUB_DIR}/omnirad_${short_name}" \
    | awk '{print $4}')
  echo "  ${short_name}: train ${train_id}, pred ${pred_id}" >&2
  r2_train_ids+=($train_id)
  r2_short_names+=($short_name)
done

# Final assembly job: depends on all 4 round-2 inferences
# (we'd need their pred IDs but easier to depend on completion of CSVs via the file watcher)
# Submit a CPU-only assembly job that depends on the 4 train jobs (which causes the chained pred jobs to fire)

echo "" >&2
echo "=== Round 2 launched. Waiting for 4 multi-split CSVs to land. ===" >&2
for short_name in "${r2_short_names[@]}"; do
  until [ -f "${SUB_DIR}/omnirad_${short_name}_va.csv" ]; do
    sleep 60
  done
  echo "  $(date -Iseconds): omnirad_${short_name}_va.csv ready" >&2
done

echo "" >&2
echo "=== All 4 multi-split CSVs landed; building final ensemble ===" >&2

# Build the final ensemble: 5-split pool (winner s0 + 4 new splits) + H+ pool + ConvNeXt-L, w442 weighted, VA
uv run python -c "
import pandas as pd
import numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

winner = '${winner}'
short_names = ['${winner}_s7', '${winner}_s13', '${winner}_s29', '${winner}_s101']

# Need s0 (winner-recipe at split 42) too — use the round-1 winner CSV
short_map = {
    'aug_randlight': 'aug_rlight', 'aug_randstd': 'aug_rstd', 'aug_randstrong': 'aug_rstrong', 'aug_trivial': 'aug_triv',
}
s0_short = short_map.get(winner, winner)
om_csvs = [f'{SUB11}/omnirad_{s0_short}.csv'] + [f'{SUB11}/omnirad_{n}.csv' for n in short_names]

oms = [pd.read_csv(p).sort_values('Id').reset_index(drop=True) for p in om_csvs]
ids = oms[0]['Id'].to_numpy()
for d in oms[1:]:
    assert (d['Id'].to_numpy() == ids).all(), 'id mismatch'

om_pool = pd.DataFrame({'Id': ids})
for lab in LABELS:
    om_pool[lab] = np.mean([d[lab].to_numpy() for d in oms], axis=0)
om_pool.to_csv(f'{SUB11}/omnirad_{winner}_5split_mean.csv', index=False, float_format='%.6f')
print(f'wrote omnirad_{winner}_5split_mean.csv')

hp_pool = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cn      = pd.read_csv(f'{SUB10}/dinov3_cnxl.csv').sort_values('Id').reset_index(drop=True)
assert (hp_pool['Id'].to_numpy() == ids).all() and (cn['Id'].to_numpy() == ids).all()

# Weighted .4 / .4 / .2 — same recipe that gave 0.657 SOTA
ens = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens[lab] = 0.4*om_pool[lab].to_numpy() + 0.4*hp_pool[lab].to_numpy() + 0.2*cn[lab].to_numpy()
ens.to_csv(f'{SUB11}/ensemble_winner_3family_w442.csv', index=False, float_format='%.6f')
print(f'wrote ensemble_winner_3family_w442.csv')
"

# View-average it
uv run python view_average.py \
  --in "${SUB_DIR}/ensemble_winner_3family_w442.csv" \
  --out "${SUB_DIR}/ensemble_winner_3family_w442_va.csv" --force

echo "" >&2
echo "=== ROUND 2 COMPLETE ===" >&2
echo "Final winner-based ensemble: ${SUB_DIR}/ensemble_winner_3family_w442_va.csv" >&2
ls -la "${SUB_DIR}/" >&2
