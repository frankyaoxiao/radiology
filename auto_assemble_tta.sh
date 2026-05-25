#!/bin/bash
# Wait for 11 TTA CSVs, then build TTA pool replacements + ensemble combos.
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

EXPECTED=(
  "${SUB}/tta_om_s0_va.csv"
  "${SUB}/tta_om_s7_va.csv"
  "${SUB}/tta_om_s13_va.csv"
  "${SUB}/tta_om_s29_va.csv"
  "${SUB}/tta_om_s101_va.csv"
  "${SUB}/tta_hp_s0_va.csv"
  "${SUB}/tta_hp_s1_va.csv"
  "${SUB}/tta_hp_s2_va.csv"
  "${SUB}/tta_cnxl_s0_va.csv"
  "${SUB}/tta_cnxl_s1_va.csv"
  "${SUB}/tta_cnxl_s2_va.csv"
)

echo "Waiting for 11 TTA CSVs..." >&2
for f in "${EXPECTED[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $f) ready" >&2
done

echo "" >&2
echo "All TTA CSVs landed. Building pools and ensembles..." >&2

uv run python -c "
import pandas as pd, numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

# Build TTA-ed pools
def pool_mean(csvs):
    dfs = [pd.read_csv(p).sort_values('Id').reset_index(drop=True) for p in csvs]
    ids = dfs[0]['Id'].to_numpy()
    out = pd.DataFrame({'Id': ids})
    for lab in LABELS:
        out[lab] = np.mean([d[lab].to_numpy() for d in dfs], axis=0)
    return out, ids

om_tta_pool, ids = pool_mean([f'{SUB11}/tta_om_s{s}_va.csv' for s in (0, 7, 13, 29, 101)])
om_tta_pool.to_csv(f'{SUB11}/omnirad_aug_trivial_5split_TTA3_mean.csv', index=False, float_format='%.6f')

hp_tta_pool, _ = pool_mean([f'{SUB11}/tta_hp_s{s}_va.csv' for s in (0, 1, 2)])
hp_tta_pool.to_csv(f'{SUB11}/hplus_aug_trivial_3seed_TTA3_mean.csv', index=False, float_format='%.6f')

cnxl_tta_pool, _ = pool_mean([f'{SUB11}/tta_cnxl_s{s}_va.csv' for s in (0, 1, 2)])
cnxl_tta_pool.to_csv(f'{SUB11}/cnxl_aug_trivial_3seed_TTA3_mean.csv', index=False, float_format='%.6f')

print('wrote TTA pool CSVs')

# Build PP-equivalent with TTA pools and 5-seed CXR
hp_pool_no_aug = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cxr5 = pd.read_csv(f'{SUB11}/cxr_foundation_5seed_pool.csv').sort_values('Id').reset_index(drop=True)
ii   = pd.read_csv(f'{SUB11}/ladder/I_D_with_cnxl_boost40.csv').sort_values('Id').reset_index(drop=True)

# TTA H+ vs no-aug H+ — we saw earlier H+ aug_trivial pool TIED with no-aug pool on public.
# So try both: TTA-ed H+ aug and original no-aug H+.

# Variant 1: PP recipe using TTA OmniRad + no-aug H+ + TTA cnxl + 5-seed CXR
ens = pd.DataFrame({'Id': ids})
for lab in LABELS:
    if lab == 'Cardiomegaly':
        ens[lab] = ii[lab].to_numpy()
    elif lab == 'Fracture':
        ens[lab] = 0.30*cnxl_tta_pool[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
    else:
        ens[lab] = (0.20*om_tta_pool[lab].to_numpy() + 0.20*hp_pool_no_aug[lab].to_numpy() +
                    0.35*cnxl_tta_pool[lab].to_numpy() + 0.25*cxr5[lab].to_numpy())
ens.to_csv(f'{SUB11}/ladder/PP_TTA_w20_20_35_25.csv', index=False, float_format='%.6f')
print('wrote PP_TTA_w20_20_35_25.csv')

# Variant 2: PP recipe using TTA OmniRad + TTA H+ aug + TTA cnxl + 5-seed CXR
ens2 = pd.DataFrame({'Id': ids})
for lab in LABELS:
    if lab == 'Cardiomegaly':
        ens2[lab] = ii[lab].to_numpy()
    elif lab == 'Fracture':
        ens2[lab] = 0.30*cnxl_tta_pool[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
    else:
        ens2[lab] = (0.20*om_tta_pool[lab].to_numpy() + 0.20*hp_tta_pool[lab].to_numpy() +
                     0.35*cnxl_tta_pool[lab].to_numpy() + 0.25*cxr5[lab].to_numpy())
ens2.to_csv(f'{SUB11}/ladder/PP_TTA_hpAUG_w20_20_35_25.csv', index=False, float_format='%.6f')
print('wrote PP_TTA_hpAUG_w20_20_35_25.csv')
"

# View-average outputs
for v in PP_TTA_w20_20_35_25 PP_TTA_hpAUG_w20_20_35_25; do
  uv run python view_average.py \
    --in submissions/2026-05-11/ladder/${v}.csv \
    --out submissions/2026-05-11/ladder/${v}_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/${v}_va.csv submissions/2026-05-11/ladder/${v}.csv
done

echo "" >&2
echo "=== TTA assembly DONE ===" >&2
ls -la submissions/2026-05-11/ladder/PP_TTA_* >&2
