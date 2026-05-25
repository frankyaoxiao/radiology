#!/bin/bash
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"
SUB10="submissions/2026-05-10"

# Wait for all 9 new SWA CSVs (eva SWA already exists)
EXPECTED=(
  "${SUB}/cnxl_aug_trivial_s0_swa_va.csv"
  "${SUB}/cnxl_aug_trivial_s1_swa_va.csv"
  "${SUB}/cnxl_aug_trivial_s2_swa_va.csv"
  "${SUB}/hplus_s0_swa_va.csv"
  "${SUB}/hplus_s1_swa_va.csv"
  "${SUB}/hplus_s2_swa_va.csv"
  "${SUB}/siglip2_p14_384_s0_swa_va.csv"
  "${SUB}/siglip2_p14_384_s1_swa_va.csv"
  "${SUB}/siglip2_p14_384_s2_swa_va.csv"
)

echo "Waiting for 9 SWA CSVs (CNXL/H+/SigLIP)..." >&2
for f in "${EXPECTED[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $f)" >&2
done
sleep 30

uv run python -c "
import pandas as pd, numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

def pool_mean(csvs):
    dfs = [pd.read_csv(p).sort_values('Id').reset_index(drop=True) for p in csvs]
    ids = dfs[0]['Id'].to_numpy()
    out = pd.DataFrame({'Id': ids})
    for lab in LABELS: out[lab] = np.mean([d[lab].to_numpy() for d in dfs], axis=0)
    return out, ids

# Pool the SWA predictions for each backbone family
cnxl_swa, ids = pool_mean([f'{SUB11}/cnxl_aug_trivial_s{s}_swa_va.csv' for s in (0,1,2)])
cnxl_swa.to_csv(f'{SUB11}/cnxl_swa_3seed_mean.csv', index=False, float_format='%.6f')

hp_swa, _ = pool_mean([f'{SUB11}/hplus_s{s}_swa_va.csv' for s in (0,1,2)])
hp_swa.to_csv(f'{SUB11}/hplus_swa_3seed_mean.csv', index=False, float_format='%.6f')

sg_swa, _ = pool_mean([f'{SUB11}/siglip2_p14_384_s{s}_swa_va.csv' for s in (0,1,2)])
sg_swa.to_csv(f'{SUB11}/siglip2_swa_3seed_mean.csv', index=False, float_format='%.6f')

# Already-built EVA SWA pool
eva_swa = pd.read_csv(f'{SUB11}/eva02_swa_3seed_mean.csv').sort_values('Id').reset_index(drop=True)

# All-SWA 6-leg PP (EVA + CNXL + H+ + SigLIP all SWA-averaged; OmniRad and CXR original)
om   = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
cxr5 = pd.read_csv(f'{SUB11}/cxr_foundation_5seed_pool.csv').sort_values('Id').reset_index(drop=True)
ii   = pd.read_csv(f'{SUB11}/ladder/I_D_with_cnxl_boost40.csv').sort_values('Id').reset_index(drop=True)

ens = pd.DataFrame({'Id': ids})
for lab in LABELS:
    if lab == 'Cardiomegaly':
        ens[lab] = ii[lab].to_numpy()
    elif lab == 'Fracture':
        ens[lab] = 0.30*cnxl_swa[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
    else:
        ens[lab] = (0.10*om[lab].to_numpy() + 0.10*hp_swa[lab].to_numpy() +
                    0.20*cnxl_swa[lab].to_numpy() + 0.14*cxr5[lab].to_numpy() +
                    0.16*sg_swa[lab].to_numpy() + 0.30*eva_swa[lab].to_numpy())
ens.to_csv(f'{SUB11}/ladder/PP_6leg_allSWA_30.csv', index=False, float_format='%.6f')

# CNXL-only and combined variants
for name, eva_csv, cnxl_csv, hp_csv, sg_csv in [
    ('PP_6leg_cnxlSWA_30',  pd.read_csv(f'{SUB11}/eva02_3seed_mean.csv'), cnxl_swa,
     pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv'),
     pd.read_csv(f'{SUB11}/siglip2_p14_384_3seed_mean.csv')),
]:
    eva_csv = eva_csv.sort_values('Id').reset_index(drop=True)
    hp_csv = hp_csv.sort_values('Id').reset_index(drop=True)
    sg_csv = sg_csv.sort_values('Id').reset_index(drop=True)
    ens = pd.DataFrame({'Id': ids})
    for lab in LABELS:
        if lab == 'Cardiomegaly':
            ens[lab] = ii[lab].to_numpy()
        elif lab == 'Fracture':
            ens[lab] = 0.30*cnxl_csv[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
        else:
            ens[lab] = (0.10*om[lab].to_numpy() + 0.10*hp_csv[lab].to_numpy() +
                        0.20*cnxl_csv[lab].to_numpy() + 0.14*cxr5[lab].to_numpy() +
                        0.16*sg_csv[lab].to_numpy() + 0.30*eva_csv[lab].to_numpy())
    ens.to_csv(f'{SUB11}/ladder/{name}.csv', index=False, float_format='%.6f')

print('wrote PP_6leg_allSWA_30 + PP_6leg_cnxlSWA_30')
"

for v in PP_6leg_allSWA_30 PP_6leg_cnxlSWA_30; do
  f="submissions/2026-05-11/ladder/${v}.csv"
  if [ -f "$f" ]; then
    uv run python view_average.py --in "$f" --out "submissions/2026-05-11/ladder/${v}_va.csv" --force 2>&1 | tail -1
    mv "submissions/2026-05-11/ladder/${v}_va.csv" "$f"
  fi
done
echo "=== ALL-SWA ASSEMBLY DONE ===" >&2
