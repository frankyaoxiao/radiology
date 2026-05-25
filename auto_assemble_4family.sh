#!/bin/bash
# Final 4-family ensemble: aug_trivial pool + aug_smooth pool + H+ pool + ConvNeXt-L.
# Waits for both 5-split mean CSVs to be assembled.
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

REQ=(
  "${SUB}/omnirad_aug_trivial_5split_mean.csv"
  "${SUB}/omnirad_aug_smooth_5split_mean.csv"
)

echo "Waiting for both 5-split mean CSVs..." >&2
for f in "${REQ[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $f) ready" >&2
done

uv run python -c "
import pandas as pd, numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

triv = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
smth = pd.read_csv(f'{SUB11}/omnirad_aug_smooth_5split_mean.csv').sort_values('Id').reset_index(drop=True)
hp_pool = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cn      = pd.read_csv(f'{SUB10}/dinov3_cnxl.csv').sort_values('Id').reset_index(drop=True)

ids = triv['Id'].to_numpy()

# Combine the two aug pools (these are the winning OmniRad voice with diversity)
# Equal weight between them since both beat baseline
om_combined = pd.DataFrame({'Id': ids})
for lab in LABELS:
    om_combined[lab] = (triv[lab].to_numpy() + smth[lab].to_numpy()) / 2.0

# 4-family ensemble (treating aug_trivial+aug_smooth as one OmniRad family at .4)
# weighted .4 / .4 / .2
ens = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens[lab] = 0.4*om_combined[lab].to_numpy() + 0.4*hp_pool[lab].to_numpy() + 0.2*cn[lab].to_numpy()
ens.to_csv(f'{SUB11}/ensemble_4family_combined_pool_w442.csv', index=False, float_format='%.6f')
print('wrote ensemble_4family_combined_pool_w442.csv')

# Alternative: literal 4-family (each aug variant is its own voice)
# .35 trivial + .35 smooth + .2 hp_pool + .1 cnxl  (boost the winning OmniRad family more)
ens2 = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens2[lab] = 0.35*triv[lab].to_numpy() + 0.35*smth[lab].to_numpy() + 0.20*hp_pool[lab].to_numpy() + 0.10*cn[lab].to_numpy()
ens2.to_csv(f'{SUB11}/ensemble_4family_literal_w35_35_20_10.csv', index=False, float_format='%.6f')
print('wrote ensemble_4family_literal_w35_35_20_10.csv')
"

# View-average both
for v in 4family_combined_pool_w442 4family_literal_w35_35_20_10; do
  uv run python view_average.py \
    --in "${SUB}/ensemble_${v}.csv" \
    --out "${SUB}/ensemble_${v}_va.csv" --force 2>&1 | tail -1
done

echo "" >&2
echo "=== 4-FAMILY ASSEMBLY DONE ===" >&2
ls -la "${SUB}/ensemble_4family"* >&2
