#!/bin/bash
# Wait for both CLAHE and EVA-02 pools to finish, then build ensembles.
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

# CLAHE
CLAHE_FILES=(
  "${SUB}/cnxl_s0_clahe_va.csv"
  "${SUB}/cnxl_s1_clahe_va.csv"
  "${SUB}/cnxl_s2_clahe_va.csv"
)
echo "Waiting for 3 CLAHE CSVs..." >&2
for f in "${CLAHE_FILES[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $f)" >&2
done
sleep 30  # finalization safety

# EVA-02
EVA_FILES=(
  "${SUB}/eva02_s0_va.csv"
  "${SUB}/eva02_s1_va.csv"
  "${SUB}/eva02_s2_va.csv"
)
echo "Waiting for 3 EVA-02 CSVs..." >&2
for f in "${EVA_FILES[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $f)" >&2
done
sleep 30

echo "" >&2
echo "All landed. Building pools + ensembles..." >&2

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

clahe_pool, ids = pool_mean([f'{SUB11}/cnxl_s{s}_clahe_va.csv' for s in (0, 1, 2)])
eva_pool, _ = pool_mean([f'{SUB11}/eva02_s{s}_va.csv' for s in (0, 1, 2)])
clahe_pool.to_csv(f'{SUB11}/cnxl_clahe_3seed_mean.csv', index=False, float_format='%.6f')
eva_pool.to_csv(f'{SUB11}/eva02_3seed_mean.csv', index=False, float_format='%.6f')

# Standalone diagnostics
clahe_pool.to_csv(f'{SUB11}/ladder/CL_clahe_cnxl_alone.csv', index=False, float_format='%.6f')
eva_pool.to_csv(f'{SUB11}/ladder/EV_eva02_alone.csv', index=False, float_format='%.6f')

# Load existing legs
om   = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
hp   = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cn   = pd.read_csv(f'{SUB11}/cnxl_aug_trivial_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cxr5 = pd.read_csv(f'{SUB11}/cxr_foundation_5seed_pool.csv').sort_values('Id').reset_index(drop=True)
sg14 = pd.read_csv(f'{SUB11}/siglip2_p14_384_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
ii   = pd.read_csv(f'{SUB11}/ladder/I_D_with_cnxl_boost40.csv').sort_values('Id').reset_index(drop=True)

# Build 6-leg ensembles with each new addition
# Goal: see if CLAHE-cnxl or EVA-02 decorrelates from existing legs
def make_pp_6leg(name, weights):
    om_w, hp_w, cn_w, cxr_w, sg_w, new_w = weights
    ens = pd.DataFrame({'Id': ids})
    new = clahe_pool if 'clahe' in name else (eva_pool if 'eva' in name else None)
    for lab in LABELS:
        if lab == 'Cardiomegaly':
            ens[lab] = ii[lab].to_numpy()
        elif lab == 'Fracture':
            ens[lab] = 0.30*cn[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
        else:
            ens[lab] = (om_w*om[lab].to_numpy() + hp_w*hp[lab].to_numpy() +
                        cn_w*cn[lab].to_numpy() + cxr_w*cxr5[lab].to_numpy() +
                        sg_w*sg14[lab].to_numpy() + new_w*new[lab].to_numpy())
    ens.to_csv(f'{SUB11}/ladder/{name}.csv', index=False, float_format='%.6f')

# CLAHE variants: add CLAHE to PP_5leg_sg14_20pct (current best at 0.648)
# .16/.16/.28/.20/.20/0  +0.05 new -> rebalance
make_pp_6leg('PP_6leg_clahe_05', (0.15, 0.15, 0.26, 0.19, 0.20, 0.05))
make_pp_6leg('PP_6leg_clahe_10', (0.14, 0.14, 0.25, 0.18, 0.19, 0.10))
make_pp_6leg('PP_6leg_clahe_15', (0.13, 0.13, 0.24, 0.17, 0.18, 0.15))

# EVA-02 variants
make_pp_6leg('PP_6leg_eva_05', (0.15, 0.15, 0.26, 0.19, 0.20, 0.05))
make_pp_6leg('PP_6leg_eva_10', (0.14, 0.14, 0.25, 0.18, 0.19, 0.10))
make_pp_6leg('PP_6leg_eva_15', (0.13, 0.13, 0.24, 0.17, 0.18, 0.15))

# Both new legs at once (7 legs total)
ens = pd.DataFrame({'Id': ids})
for lab in LABELS:
    if lab == 'Cardiomegaly':
        ens[lab] = ii[lab].to_numpy()
    elif lab == 'Fracture':
        ens[lab] = 0.30*cn[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
    else:
        ens[lab] = (0.12*om[lab].to_numpy() + 0.12*hp[lab].to_numpy() +
                    0.22*cn[lab].to_numpy() + 0.16*cxr5[lab].to_numpy() +
                    0.18*sg14[lab].to_numpy() +
                    0.10*clahe_pool[lab].to_numpy() + 0.10*eva_pool[lab].to_numpy())
ens.to_csv(f'{SUB11}/ladder/PP_7leg_clahe_eva_10each.csv', index=False, float_format='%.6f')

print('wrote CL/EV (standalone) + 6 PP_6leg variants + PP_7leg')
"

# VA all
for v in CL_clahe_cnxl_alone EV_eva02_alone \
         PP_6leg_clahe_05 PP_6leg_clahe_10 PP_6leg_clahe_15 \
         PP_6leg_eva_05 PP_6leg_eva_10 PP_6leg_eva_15 \
         PP_7leg_clahe_eva_10each; do
  uv run python view_average.py \
    --in submissions/2026-05-11/ladder/${v}.csv \
    --out submissions/2026-05-11/ladder/${v}_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/${v}_va.csv submissions/2026-05-11/ladder/${v}.csv
done

echo "" >&2
echo "=== CLAHE/EVA-02 ENSEMBLES DONE ===" >&2
ls -la submissions/2026-05-11/ladder/CL_* submissions/2026-05-11/ladder/EV_* submissions/2026-05-11/ladder/PP_6leg_* submissions/2026-05-11/ladder/PP_7leg_* >&2
