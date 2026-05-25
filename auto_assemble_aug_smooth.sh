#!/bin/bash
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

EXPECTED=(
  "${SUB}/omnirad_aug_smooth_s7_va.csv"
  "${SUB}/omnirad_aug_smooth_s13_va.csv"
  "${SUB}/omnirad_aug_smooth_s29_va.csv"
  "${SUB}/omnirad_aug_smooth_s101_va.csv"
  "${SUB}/omnirad_aug_smooth_va.csv"
)

echo "Waiting for 5 aug_smooth CSVs..." >&2
for f in "${EXPECTED[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $f) ready" >&2
done

uv run python -c "
import pandas as pd, numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

csvs = [
    f'{SUB11}/omnirad_aug_smooth.csv',
    f'{SUB11}/omnirad_aug_smooth_s7.csv',
    f'{SUB11}/omnirad_aug_smooth_s13.csv',
    f'{SUB11}/omnirad_aug_smooth_s29.csv',
    f'{SUB11}/omnirad_aug_smooth_s101.csv',
]
oms = [pd.read_csv(p).sort_values('Id').reset_index(drop=True) for p in csvs]
ids = oms[0]['Id'].to_numpy()
om_pool = pd.DataFrame({'Id': ids})
for lab in LABELS: om_pool[lab] = np.mean([d[lab].to_numpy() for d in oms], axis=0)
om_pool.to_csv(f'{SUB11}/omnirad_aug_smooth_5split_mean.csv', index=False, float_format='%.6f')
print('wrote omnirad_aug_smooth_5split_mean.csv')

hp_pool = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cn      = pd.read_csv(f'{SUB10}/dinov3_cnxl.csv').sort_values('Id').reset_index(drop=True)

ens = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens[lab] = 0.4*om_pool[lab].to_numpy() + 0.4*hp_pool[lab].to_numpy() + 0.2*cn[lab].to_numpy()
ens.to_csv(f'{SUB11}/ensemble_aug_smooth_w442_hp_pool.csv', index=False, float_format='%.6f')
print('wrote ensemble_aug_smooth_w442_hp_pool.csv')
"

uv run python view_average.py \
  --in "${SUB}/ensemble_aug_smooth_w442_hp_pool.csv" \
  --out "${SUB}/ensemble_aug_smooth_w442_hp_pool_va.csv" --force

echo "" >&2
echo "=== AUG_SMOOTH ASSEMBLY DONE ===" >&2
ls -la "${SUB}/ensemble_aug_smooth"* >&2
