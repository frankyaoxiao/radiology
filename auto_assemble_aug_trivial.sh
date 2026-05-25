#!/bin/bash
# Wait for the 4 aug_trivial multi-split CSVs + s0 CSV, then assemble final ensemble.
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

EXPECTED=(
  "${SUB}/omnirad_aug_trivial_s7_va.csv"
  "${SUB}/omnirad_aug_trivial_s13_va.csv"
  "${SUB}/omnirad_aug_trivial_s29_va.csv"
  "${SUB}/omnirad_aug_trivial_s101_va.csv"
  "${SUB}/omnirad_aug_triv_va.csv"
)

echo "Waiting for 5 aug_trivial CSVs..." >&2
for f in "${EXPECTED[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $f) ready" >&2
done

echo "" >&2
echo "All 5 aug_trivial CSVs landed; assembling final ensemble..." >&2

uv run python -c "
import pandas as pd, numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

csvs = [
    f'{SUB11}/omnirad_aug_triv.csv',
    f'{SUB11}/omnirad_aug_trivial_s7.csv',
    f'{SUB11}/omnirad_aug_trivial_s13.csv',
    f'{SUB11}/omnirad_aug_trivial_s29.csv',
    f'{SUB11}/omnirad_aug_trivial_s101.csv',
]
oms = [pd.read_csv(p).sort_values('Id').reset_index(drop=True) for p in csvs]
ids = oms[0]['Id'].to_numpy()
om_pool = pd.DataFrame({'Id': ids})
for lab in LABELS: om_pool[lab] = np.mean([d[lab].to_numpy() for d in oms], axis=0)
om_pool.to_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv', index=False, float_format='%.6f')
print('wrote omnirad_aug_trivial_5split_mean.csv')

hp_pool = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
hp_single = pd.read_csv(f'{SUB10}/dinov3_hplus.csv').sort_values('Id').reset_index(drop=True)
cn = pd.read_csv(f'{SUB10}/dinov3_cnxl.csv').sort_values('Id').reset_index(drop=True)

# Variant A: with H+ 3-seed pool, .4/.4/.2
ens_a = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens_a[lab] = 0.4*om_pool[lab].to_numpy() + 0.4*hp_pool[lab].to_numpy() + 0.2*cn[lab].to_numpy()
ens_a.to_csv(f'{SUB11}/ensemble_aug_trivial_w442_hp_pool.csv', index=False, float_format='%.6f')

# Variant B: with H+ single, .4/.4/.2
ens_b = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens_b[lab] = 0.4*om_pool[lab].to_numpy() + 0.4*hp_single[lab].to_numpy() + 0.2*cn[lab].to_numpy()
ens_b.to_csv(f'{SUB11}/ensemble_aug_trivial_w442_hp_single.csv', index=False, float_format='%.6f')

# Variant C: equal weights .333 each, with H+ pool
ens_c = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens_c[lab] = (om_pool[lab].to_numpy() + hp_pool[lab].to_numpy() + cn[lab].to_numpy()) / 3.0
ens_c.to_csv(f'{SUB11}/ensemble_aug_trivial_eq_hp_pool.csv', index=False, float_format='%.6f')

print('wrote ensemble_aug_trivial_{w442_hp_pool,w442_hp_single,eq_hp_pool}.csv')
"

# View-average all 3 variants
for v in w442_hp_pool w442_hp_single eq_hp_pool; do
  uv run python view_average.py \
    --in "${SUB}/ensemble_aug_trivial_${v}.csv" \
    --out "${SUB}/ensemble_aug_trivial_${v}_va.csv" --force 2>&1 | tail -1
done

echo "" >&2
echo "=== AUG_TRIVIAL ENSEMBLE ASSEMBLY COMPLETE ===" >&2
ls -la "${SUB}/ensemble_aug_trivial"* >&2
