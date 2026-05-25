#!/bin/bash
# Wait for 6 SigLIP 2 CSVs (3 seeds × 2 variants), build pools + ensembles.
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

EXPECTED=(
  "${SUB}/siglip2_p14_384_s0_va.csv"
  "${SUB}/siglip2_p14_384_s1_va.csv"
  "${SUB}/siglip2_p14_384_s2_va.csv"
  "${SUB}/siglip2_naflex_512_s0_va.csv"
  "${SUB}/siglip2_naflex_512_s1_va.csv"
  "${SUB}/siglip2_naflex_512_s2_va.csv"
)

echo "Waiting for 6 SigLIP 2 CSVs..." >&2
for f in "${EXPECTED[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $f) ready" >&2
done

# Extra safety
sleep 30

uv run python -c "
import pandas as pd, numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

# Per-variant pools (3 seeds)
def pool_mean(csvs):
    dfs = [pd.read_csv(p).sort_values('Id').reset_index(drop=True) for p in csvs]
    ids = dfs[0]['Id'].to_numpy()
    out = pd.DataFrame({'Id': ids})
    for lab in LABELS: out[lab] = np.mean([d[lab].to_numpy() for d in dfs], axis=0)
    return out, ids

p14_pool, ids = pool_mean([f'{SUB11}/siglip2_p14_384_s{s}_va.csv' for s in (0, 1, 2)])
naflex_pool, _ = pool_mean([f'{SUB11}/siglip2_naflex_512_s{s}_va.csv' for s in (0, 1, 2)])
p14_pool.to_csv(f'{SUB11}/siglip2_p14_384_3seed_mean.csv', index=False, float_format='%.6f')
naflex_pool.to_csv(f'{SUB11}/siglip2_naflex_512_3seed_mean.csv', index=False, float_format='%.6f')

# Combined SigLIP 2 pool (both variants averaged, 6 models)
both = pd.DataFrame({'Id': ids})
for lab in LABELS:
    both[lab] = (p14_pool[lab].to_numpy() + naflex_pool[lab].to_numpy()) / 2.0
both.to_csv(f'{SUB11}/siglip2_combined_pool.csv', index=False, float_format='%.6f')

# Standalone diagnostics
p14_pool.to_csv(f'{SUB11}/ladder/SG1_siglip2_p14_alone.csv', index=False, float_format='%.6f')
naflex_pool.to_csv(f'{SUB11}/ladder/SG2_siglip2_naflex_alone.csv', index=False, float_format='%.6f')
both.to_csv(f'{SUB11}/ladder/SG3_siglip2_combined_alone.csv', index=False, float_format='%.6f')

# Load all other backbones
om_pool   = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
hp_pool   = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cnxl_pool = pd.read_csv(f'{SUB11}/cnxl_aug_trivial_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cxr5      = pd.read_csv(f'{SUB11}/cxr_foundation_5seed_pool.csv').sort_values('Id').reset_index(drop=True)
ii        = pd.read_csv(f'{SUB11}/ladder/I_D_with_cnxl_boost40.csv').sort_values('Id').reset_index(drop=True)

# Ensemble variants: 5-leg ensembles with SigLIP 2 added
# SG_A: PP recipe + siglip2 at 15%, redistribute proportionally
def make_5leg(name, om_w, hp_w, cn_w, cxr_w, sg_w, sg_csv):
    ens = pd.DataFrame({'Id': ids})
    for lab in LABELS:
        if lab == 'Cardiomegaly':
            ens[lab] = ii[lab].to_numpy()
        elif lab == 'Fracture':
            ens[lab] = 0.30*cnxl_pool[lab].to_numpy() + 0.70*cxr5[lab].to_numpy()
        else:
            ens[lab] = (om_w*om_pool[lab].to_numpy() + hp_w*hp_pool[lab].to_numpy() +
                        cn_w*cnxl_pool[lab].to_numpy() + cxr_w*cxr5[lab].to_numpy() +
                        sg_w*sg_csv[lab].to_numpy())
    ens.to_csv(f'{SUB11}/ladder/{name}.csv', index=False, float_format='%.6f')

# Add 15% SigLIP 2 (p14 only)
make_5leg('PP_5leg_p14_15pct', 0.18, 0.18, 0.30, 0.20, 0.15, p14_pool)
# Add 15% SigLIP 2 (naflex only)
make_5leg('PP_5leg_naflex_15pct', 0.18, 0.18, 0.30, 0.20, 0.15, naflex_pool)
# Add 15% SigLIP 2 (combined)
make_5leg('PP_5leg_sigcombined_15pct', 0.18, 0.18, 0.30, 0.20, 0.15, both)
# Add 20% SigLIP 2 combined
make_5leg('PP_5leg_sigcombined_20pct', 0.16, 0.16, 0.28, 0.20, 0.20, both)

print('wrote SigLIP 2 pools + 4 ensembles + 3 standalones')
"

# VA all
for v in SG1_siglip2_p14_alone SG2_siglip2_naflex_alone SG3_siglip2_combined_alone \
         PP_5leg_p14_15pct PP_5leg_naflex_15pct PP_5leg_sigcombined_15pct PP_5leg_sigcombined_20pct; do
  uv run python view_average.py \
    --in submissions/2026-05-11/ladder/${v}.csv \
    --out submissions/2026-05-11/ladder/${v}_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/${v}_va.csv submissions/2026-05-11/ladder/${v}.csv
done

echo "" >&2
echo "=== SIGLIP 2 ENSEMBLE BUILD DONE ===" >&2
ls -la submissions/2026-05-11/ladder/SG* submissions/2026-05-11/ladder/PP_5leg* >&2
