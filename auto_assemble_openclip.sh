#!/bin/bash
# Wait for 3 OpenCLIP CSVs, build pool + 7-leg PP ensembles.
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

EXPECTED=(
  "${SUB}/openclip_s0_va.csv"
  "${SUB}/openclip_s1_va.csv"
  "${SUB}/openclip_s2_va.csv"
)

echo "Waiting for 3 OpenCLIP CSVs..." >&2
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

oc_pool, ids = pool_mean([f'{SUB11}/openclip_s{s}_va.csv' for s in (0,1,2)])
oc_pool.to_csv(f'{SUB11}/openclip_3seed_mean.csv', index=False, float_format='%.6f')

# Standalone
oc_pool.to_csv(f'{SUB11}/ladder/OC_openclip_alone.csv', index=False, float_format='%.6f')

om   = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
hp   = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cn   = pd.read_csv(f'{SUB11}/cnxl_aug_trivial_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cxr5 = pd.read_csv(f'{SUB11}/cxr_foundation_5seed_pool.csv').sort_values('Id').reset_index(drop=True)
sg14 = pd.read_csv(f'{SUB11}/siglip2_p14_384_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
eva  = pd.read_csv(f'{SUB11}/eva02_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
ii   = pd.read_csv(f'{SUB11}/ladder/I_D_with_cnxl_boost40.csv').sort_values('Id').reset_index(drop=True)

# 7-leg PP variants with OpenCLIP added (replacing some of the lower-weight legs proportionally)
# Base PP_6leg_eva_30: .10 om / .10 hp / .20 cn / .14 cxr / .16 sg / .30 eva
def make_pp_7leg(name, om_w, hp_w, cn_w, cxr_w, sg_w, eva_w, oc_w):
    assert abs(om_w + hp_w + cn_w + cxr_w + sg_w + eva_w + oc_w - 1.0) < 1e-6
    ens = pd.DataFrame({'Id': ids})
    for lab in LABELS:
        if lab == 'Cardiomegaly':
            ens[lab] = ii[lab].to_numpy()
        elif lab == 'Fracture':
            ens[lab] = 0.30*cn[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
        else:
            ens[lab] = (om_w*om[lab].to_numpy() + hp_w*hp[lab].to_numpy() +
                        cn_w*cn[lab].to_numpy() + cxr_w*cxr5[lab].to_numpy() +
                        sg_w*sg14[lab].to_numpy() + eva_w*eva[lab].to_numpy() +
                        oc_w*oc_pool[lab].to_numpy())
    ens.to_csv(f'{SUB11}/ladder/{name}.csv', index=False, float_format='%.6f')

# Add OpenCLIP at 10/15/20% by reducing other legs proportionally
make_pp_7leg('PP_7leg_oc_10', 0.09, 0.09, 0.18, 0.13, 0.14, 0.27, 0.10)
make_pp_7leg('PP_7leg_oc_15', 0.08, 0.08, 0.17, 0.12, 0.14, 0.26, 0.15)
make_pp_7leg('PP_7leg_oc_20', 0.08, 0.08, 0.16, 0.11, 0.13, 0.24, 0.20)
make_pp_7leg('PP_7leg_oc_25', 0.07, 0.07, 0.15, 0.10, 0.12, 0.24, 0.25)
make_pp_7leg('PP_7leg_oc_30', 0.07, 0.07, 0.14, 0.09, 0.11, 0.22, 0.30)
print('wrote OC_openclip_alone + 5 PP_7leg_oc variants')
"

for v in OC_openclip_alone PP_7leg_oc_10 PP_7leg_oc_15 PP_7leg_oc_20 PP_7leg_oc_25 PP_7leg_oc_30; do
  uv run python view_average.py \
    --in submissions/2026-05-11/ladder/${v}.csv \
    --out submissions/2026-05-11/ladder/${v}_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/${v}_va.csv submissions/2026-05-11/ladder/${v}.csv
done

echo "" >&2
echo "=== OPENCLIP ASSEMBLY DONE ===" >&2
ls submissions/2026-05-11/ladder/OC_* submissions/2026-05-11/ladder/PP_7leg_oc_*.csv >&2
