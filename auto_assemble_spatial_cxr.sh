#!/bin/bash
# Wait for 8 spatial CXR shards, then train heads and build ensembles.
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"
SHARDS_DIR=/data/artifacts/frank/misc/cxr_embeds_spatial

echo "Waiting for 8 spatial CXR shards..." >&2
for r in 0 1 2 3 4 5 6 7; do
  f="${SHARDS_DIR}/shard_0${r}_of_08.npz"
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): shard_${r} ready" >&2
done

echo "" >&2
echo "All 8 shards landed. Training spatial heads (3 seeds in parallel)..." >&2

# Train 3 spatial heads with different seeds
for s in 0 1 2; do
  out="${SUB}/ladder/cxr_spatial_head_s${s}.csv"
  (uv run python cxr_spatial_head_train.py --out-csv "${out}" --seed ${s} > /tmp/cxr_spatial_s${s}.log 2>&1) &
done
wait

echo "Heads trained. Building pool + ensembles..." >&2

uv run python -c "
import pandas as pd, numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

# Spatial CXR pool (3 seeds)
heads = [pd.read_csv(f'{SUB11}/ladder/cxr_spatial_head_s{s}.csv').sort_values('Id').reset_index(drop=True) for s in range(3)]
ids = heads[0]['Id'].to_numpy()
spatial_pool = pd.DataFrame({'Id': ids})
for lab in LABELS: spatial_pool[lab] = np.mean([h[lab].to_numpy() for h in heads], axis=0)
spatial_pool.to_csv(f'{SUB11}/cxr_spatial_3seed_pool.csv', index=False, float_format='%.6f')
print('wrote cxr_spatial_3seed_pool.csv')

# Standalone spatial (diagnostic)
spatial_pool.to_csv(f'{SUB11}/ladder/SP_spatial_cxr_pool_alone.csv', index=False, float_format='%.6f')

# Build PP-equivalent with spatial CXR pool
om_pool   = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
hp_pool   = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cnxl_pool = pd.read_csv(f'{SUB11}/cnxl_aug_trivial_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
ii        = pd.read_csv(f'{SUB11}/ladder/I_D_with_cnxl_boost40.csv').sort_values('Id').reset_index(drop=True)

# Variant: PP recipe with spatial CXR
ens = pd.DataFrame({'Id': ids})
for lab in LABELS:
    if lab == 'Cardiomegaly':
        ens[lab] = ii[lab].to_numpy()
    elif lab == 'Fracture':
        ens[lab] = 0.30*cnxl_pool[lab].to_numpy() + 0.70*spatial_pool[lab].to_numpy()
    else:
        ens[lab] = (0.20*om_pool[lab].to_numpy() + 0.20*hp_pool[lab].to_numpy() +
                    0.35*cnxl_pool[lab].to_numpy() + 0.25*spatial_pool[lab].to_numpy())
ens.to_csv(f'{SUB11}/ladder/PP_spatial_FFbase.csv', index=False, float_format='%.6f')

# Variant: FF (.20/.20/.35/.25) with spatial CXR
ens2 = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ens2[lab] = 0.20*om_pool[lab].to_numpy() + 0.20*hp_pool[lab].to_numpy() + 0.35*cnxl_pool[lab].to_numpy() + 0.25*spatial_pool[lab].to_numpy()
ens2.to_csv(f'{SUB11}/ladder/FF_spatial_w20_20_35_25.csv', index=False, float_format='%.6f')

print('wrote SP/PP_spatial/FF_spatial')
"

for v in SP_spatial_cxr_pool_alone PP_spatial_FFbase FF_spatial_w20_20_35_25; do
  uv run python view_average.py \
    --in submissions/2026-05-11/ladder/${v}.csv \
    --out submissions/2026-05-11/ladder/${v}_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/${v}_va.csv submissions/2026-05-11/ladder/${v}.csv
done

echo "" >&2
echo "=== SPATIAL ASSEMBLY DONE ===" >&2
ls -la submissions/2026-05-11/ladder/*spatial* >&2
