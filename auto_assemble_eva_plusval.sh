#!/bin/bash
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"
SUB10="submissions/2026-05-10"

EXPECTED=(
  "${SUB}/eva02_s0_plusval_va.csv"
  "${SUB}/eva02_s1_plusval_va.csv"
  "${SUB}/eva02_s2_plusval_va.csv"
)

echo "Waiting for 3 EVA-02 plusval CSVs..." >&2
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

eva_pv, ids = pool_mean([f'{SUB11}/eva02_s{s}_plusval_va.csv' for s in (0,1,2)])
eva_pv.to_csv(f'{SUB11}/eva02_plusval_3seed_mean.csv', index=False, float_format='%.6f')
eva_pv.to_csv(f'{SUB11}/ladder/EVP_plusval_alone.csv', index=False, float_format='%.6f')

om   = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
hp   = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cn   = pd.read_csv(f'{SUB11}/cnxl_aug_trivial_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cxr5 = pd.read_csv(f'{SUB11}/cxr_foundation_5seed_pool.csv').sort_values('Id').reset_index(drop=True)
sg14 = pd.read_csv(f'{SUB11}/siglip2_p14_384_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
ii   = pd.read_csv(f'{SUB11}/ladder/I_D_with_cnxl_boost40.csv').sort_values('Id').reset_index(drop=True)

ens = pd.DataFrame({'Id': ids})
for lab in LABELS:
    if lab == 'Cardiomegaly':
        ens[lab] = ii[lab].to_numpy()
    elif lab == 'Fracture':
        ens[lab] = 0.30*cn[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
    else:
        ens[lab] = (0.10*om[lab].to_numpy() + 0.10*hp[lab].to_numpy() +
                    0.20*cn[lab].to_numpy() + 0.14*cxr5[lab].to_numpy() +
                    0.16*sg14[lab].to_numpy() + 0.30*eva_pv[lab].to_numpy())
ens.to_csv(f'{SUB11}/ladder/PP_6leg_evaPlusval_30.csv', index=False, float_format='%.6f')
print('wrote EVP_plusval_alone + PP_6leg_evaPlusval_30')
"

for v in EVP_plusval_alone PP_6leg_evaPlusval_30; do
  uv run python view_average.py --in submissions/2026-05-11/ladder/${v}.csv --out submissions/2026-05-11/ladder/${v}_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/${v}_va.csv submissions/2026-05-11/ladder/${v}.csv
done
echo "=== EVA PLUSVAL ASSEMBLY DONE ===" >&2
