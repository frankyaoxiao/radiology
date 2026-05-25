#!/bin/bash
# Wait for 3 EVA-02 split-diversified CSVs, build new EVA pool + 6-leg ensemble.
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

EXPECTED=(
  "${SUB}/eva02_s0_split13_va.csv"
  "${SUB}/eva02_s1_split29_va.csv"
  "${SUB}/eva02_s2_split101_va.csv"
)

echo "Waiting for 3 EVA-02 split-diversified CSVs..." >&2
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

# Build new EVA split-diversified pool
eva_split, ids = pool_mean([
    f'{SUB11}/eva02_s0_split13_va.csv',
    f'{SUB11}/eva02_s1_split29_va.csv',
    f'{SUB11}/eva02_s2_split101_va.csv',
])
eva_split.to_csv(f'{SUB11}/eva02_split_3seed_mean.csv', index=False, float_format='%.6f')

# Also build a 6-seed EVA pool: original 3 seeds (split=42) + new 3 seeds (splits 13/29/101)
eva_orig_csvs = [f'{SUB11}/eva02_s0_va.csv', f'{SUB11}/eva02_s1_va.csv', f'{SUB11}/eva02_s2_va.csv']
import os
all_eva = eva_orig_csvs + [
    f'{SUB11}/eva02_s0_split13_va.csv',
    f'{SUB11}/eva02_s1_split29_va.csv',
    f'{SUB11}/eva02_s2_split101_va.csv',
]
all_exist = all(os.path.exists(p) for p in all_eva)
if all_exist:
    eva_6, _ = pool_mean(all_eva)
    eva_6.to_csv(f'{SUB11}/eva02_6seed_combined_mean.csv', index=False, float_format='%.6f')
    print('wrote eva02_split_3seed_mean.csv + eva02_6seed_combined_mean.csv')
else:
    print('wrote eva02_split_3seed_mean.csv only (original EVA CSVs missing)')

# Standalone diagnostic
eva_split.to_csv(f'{SUB11}/ladder/EVS_split_alone.csv', index=False, float_format='%.6f')
if all_exist:
    eva_6.to_csv(f'{SUB11}/ladder/EV6_combined_alone.csv', index=False, float_format='%.6f')

# Build PP_6leg_eva_30 with EVA replaced by split pool (and by 6-seed combined pool)
om   = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
hp   = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cn   = pd.read_csv(f'{SUB11}/cnxl_aug_trivial_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cxr5 = pd.read_csv(f'{SUB11}/cxr_foundation_5seed_pool.csv').sort_values('Id').reset_index(drop=True)
sg14 = pd.read_csv(f'{SUB11}/siglip2_p14_384_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
ii   = pd.read_csv(f'{SUB11}/ladder/I_D_with_cnxl_boost40.csv').sort_values('Id').reset_index(drop=True)

def make_pp(name, eva_csv):
    ens = pd.DataFrame({'Id': ids})
    for lab in LABELS:
        if lab == 'Cardiomegaly':
            ens[lab] = ii[lab].to_numpy()
        elif lab == 'Fracture':
            ens[lab] = 0.30*cn[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
        else:
            ens[lab] = (0.10*om[lab].to_numpy() + 0.10*hp[lab].to_numpy() +
                        0.20*cn[lab].to_numpy() + 0.14*cxr5[lab].to_numpy() +
                        0.16*sg14[lab].to_numpy() + 0.30*eva_csv[lab].to_numpy())
    ens.to_csv(f'{SUB11}/ladder/{name}.csv', index=False, float_format='%.6f')

make_pp('PP_6leg_evaSplit_30', eva_split)
if all_exist:
    make_pp('PP_6leg_eva6Comb_30', eva_6)
print('wrote PP ensembles')
"

# Run view-averaging on the new ladder files
for v in EVS_split_alone EV6_combined_alone PP_6leg_evaSplit_30 PP_6leg_eva6Comb_30; do
  f="submissions/2026-05-11/ladder/${v}.csv"
  if [ -f "$f" ]; then
    uv run python view_average.py --in "$f" --out "submissions/2026-05-11/ladder/${v}_va.csv" --force 2>&1 | tail -1
    mv "submissions/2026-05-11/ladder/${v}_va.csv" "$f"
  fi
done

echo "" >&2
echo "=== EVA SPLIT ASSEMBLY DONE ===" >&2
ls submissions/2026-05-11/ladder/EVS_* submissions/2026-05-11/ladder/EV6_* submissions/2026-05-11/ladder/PP_6leg_eva{Split,6Comb}_*.csv 2>/dev/null >&2
