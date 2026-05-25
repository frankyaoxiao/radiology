#!/bin/bash
# Wait for 3 CNXL split-diversified CSVs, build new CNXL pool + 6-leg ensembles.
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

EXPECTED=(
  "${SUB}/cnxl_s0_split13_va.csv"
  "${SUB}/cnxl_s1_split29_va.csv"
  "${SUB}/cnxl_s2_split101_va.csv"
)

echo "Waiting for 3 CNXL split-diversified CSVs..." >&2
for f in "${EXPECTED[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $f)" >&2
done
sleep 30

uv run python -c "
import pandas as pd, numpy as np, os
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

cnxl_split, ids = pool_mean([
    f'{SUB11}/cnxl_s0_split13_va.csv',
    f'{SUB11}/cnxl_s1_split29_va.csv',
    f'{SUB11}/cnxl_s2_split101_va.csv',
])
cnxl_split.to_csv(f'{SUB11}/cnxl_split_3seed_mean.csv', index=False, float_format='%.6f')

cnxl_orig = [f'{SUB11}/cnxl_aug_trivial_s{s}_va.csv' for s in (0,1,2)]
all_cnxl = cnxl_orig + [
    f'{SUB11}/cnxl_s0_split13_va.csv',
    f'{SUB11}/cnxl_s1_split29_va.csv',
    f'{SUB11}/cnxl_s2_split101_va.csv',
]
all_exist = all(os.path.exists(p) for p in all_cnxl)
if all_exist:
    cnxl_6, _ = pool_mean(all_cnxl)
    cnxl_6.to_csv(f'{SUB11}/cnxl_6seed_combined_mean.csv', index=False, float_format='%.6f')

cnxl_split.to_csv(f'{SUB11}/ladder/CNS_split_alone.csv', index=False, float_format='%.6f')
if all_exist:
    cnxl_6.to_csv(f'{SUB11}/ladder/CN6_combined_alone.csv', index=False, float_format='%.6f')

# Build PP_6leg replacing CNXL leg
om   = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
hp   = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cxr5 = pd.read_csv(f'{SUB11}/cxr_foundation_5seed_pool.csv').sort_values('Id').reset_index(drop=True)
sg14 = pd.read_csv(f'{SUB11}/siglip2_p14_384_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
eva  = pd.read_csv(f'{SUB11}/eva02_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
# Also use the new diverse EVA if it exists
eva_split_path = f'{SUB11}/eva02_split_3seed_mean.csv'
eva_split = pd.read_csv(eva_split_path).sort_values('Id').reset_index(drop=True) if os.path.exists(eva_split_path) else None
ii   = pd.read_csv(f'{SUB11}/ladder/I_D_with_cnxl_boost40.csv').sort_values('Id').reset_index(drop=True)

def make_pp(name, cnxl_csv, eva_csv):
    ens = pd.DataFrame({'Id': ids})
    for lab in LABELS:
        if lab == 'Cardiomegaly':
            ens[lab] = ii[lab].to_numpy()
        elif lab == 'Fracture':
            ens[lab] = 0.30*cnxl_csv[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
        else:
            ens[lab] = (0.10*om[lab].to_numpy() + 0.10*hp[lab].to_numpy() +
                        0.20*cnxl_csv[lab].to_numpy() + 0.14*cxr5[lab].to_numpy() +
                        0.16*sg14[lab].to_numpy() + 0.30*eva_csv[lab].to_numpy())
    ens.to_csv(f'{SUB11}/ladder/{name}.csv', index=False, float_format='%.6f')

# PP with new CNXL, original EVA
make_pp('PP_6leg_cnxlSplit_30', cnxl_split, eva)
# PP with new CNXL + new EVA (BOTH diversified)
if eva_split is not None:
    make_pp('PP_6leg_bothSplit_30', cnxl_split, eva_split)
# PP with 6-seed combined CNXL + original EVA
if all_exist:
    make_pp('PP_6leg_cnxl6Comb_30', cnxl_6, eva)
print('CNXL assembly done')
"

for v in CNS_split_alone CN6_combined_alone PP_6leg_cnxlSplit_30 PP_6leg_bothSplit_30 PP_6leg_cnxl6Comb_30; do
  f="submissions/2026-05-11/ladder/${v}.csv"
  if [ -f "$f" ]; then
    uv run python view_average.py --in "$f" --out "submissions/2026-05-11/ladder/${v}_va.csv" --force 2>&1 | tail -1
    mv "submissions/2026-05-11/ladder/${v}_va.csv" "$f"
  fi
done

echo "" >&2
echo "=== CNXL SPLIT ASSEMBLY DONE ===" >&2
ls submissions/2026-05-11/ladder/CNS_* submissions/2026-05-11/ladder/CN6_* submissions/2026-05-11/ladder/PP_6leg_{cnxl,both}*.csv 2>/dev/null >&2
