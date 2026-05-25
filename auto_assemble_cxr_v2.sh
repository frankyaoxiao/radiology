#!/bin/bash
# Wait for 8 v2 shards (with bit-identical TF preprocessing), then train heads:
# 1. Plain GAP head on the new shards — should match original 0.2942 val
# 2. Attention-pool head on 8x8 spatial features — the real test
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"
SHARDS_DIR=/data/artifacts/frank/misc/cxr_embeds_v2

echo "Waiting for 8 v2 shards..." >&2
for r in 0 1 2 3 4 5 6 7; do
  f="${SHARDS_DIR}/shard_0${r}_of_08.npz"
  until [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 100000 ]; do sleep 60; done
  echo "  $(date -Iseconds): shard_${r} ready" >&2
done

# Extra safety: wait for file modification times to be stable
sleep 30

echo "" >&2
echo "All 8 shards landed. Training spatial-attention heads (3 seeds in parallel)..." >&2

for s in 0 1 2; do
  out="${SUB}/ladder/cxr_attn_head_s${s}.csv"
  (uv run python cxr_attn_head_train.py --out-csv "${out}" --seed ${s} > /tmp/cxr_attn_s${s}.log 2>&1) &
done
wait

echo "Heads trained. Logging val NMSE per seed..." >&2
for s in 0 1 2; do
  v=$(grep 'best val_nmse' /tmp/cxr_attn_s${s}.log | tail -1)
  echo "  seed ${s}: ${v}" >&2
done

# Build 3-seed attention pool
uv run python -c "
import pandas as pd, numpy as np
from config import LABEL_NAMES
LABELS = list(LABEL_NAMES)
SUB10 = 'submissions/2026-05-10'
SUB11 = 'submissions/2026-05-11'

heads = [pd.read_csv(f'{SUB11}/ladder/cxr_attn_head_s{s}.csv').sort_values('Id').reset_index(drop=True) for s in range(3)]
ids = heads[0]['Id'].to_numpy()
attn_pool = pd.DataFrame({'Id': ids})
for lab in LABELS: attn_pool[lab] = np.mean([h[lab].to_numpy() for h in heads], axis=0)
attn_pool.to_csv(f'{SUB11}/cxr_attn_3seed_pool.csv', index=False, float_format='%.6f')

# Standalone diagnostic
attn_pool.to_csv(f'{SUB11}/ladder/AT_attn_cxr_alone.csv', index=False, float_format='%.6f')

# PP-equivalent with attention pool
om = pd.read_csv(f'{SUB11}/omnirad_aug_trivial_5split_mean.csv').sort_values('Id').reset_index(drop=True)
hp = pd.read_csv(f'{SUB10}/dinov3_hplus_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
cn = pd.read_csv(f'{SUB11}/cnxl_aug_trivial_3seed_mean.csv').sort_values('Id').reset_index(drop=True)
ii = pd.read_csv(f'{SUB11}/ladder/I_D_with_cnxl_boost40.csv').sort_values('Id').reset_index(drop=True)

# PP with attention CXR
ens = pd.DataFrame({'Id': ids})
for lab in LABELS:
    if lab == 'Cardiomegaly':
        ens[lab] = ii[lab].to_numpy()
    elif lab == 'Fracture':
        ens[lab] = 0.30*cn[lab].to_numpy() + 0.70*attn_pool[lab].to_numpy()
    else:
        ens[lab] = (0.20*om[lab].to_numpy() + 0.20*hp[lab].to_numpy() +
                    0.35*cn[lab].to_numpy() + 0.25*attn_pool[lab].to_numpy())
ens.to_csv(f'{SUB11}/ladder/PP_attn_FFbase.csv', index=False, float_format='%.6f')

# FF with attention CXR
ff = pd.DataFrame({'Id': ids})
for lab in LABELS:
    ff[lab] = (0.20*om[lab].to_numpy() + 0.20*hp[lab].to_numpy() +
               0.35*cn[lab].to_numpy() + 0.25*attn_pool[lab].to_numpy())
ff.to_csv(f'{SUB11}/ladder/FF_attn_w20_20_35_25.csv', index=False, float_format='%.6f')

print('wrote AT/PP_attn/FF_attn')
"

for v in AT_attn_cxr_alone PP_attn_FFbase FF_attn_w20_20_35_25; do
  uv run python view_average.py \
    --in submissions/2026-05-11/ladder/${v}.csv \
    --out submissions/2026-05-11/ladder/${v}_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/${v}_va.csv submissions/2026-05-11/ladder/${v}.csv
done

echo "" >&2
echo "=== ATTENTION CXR ASSEMBLY DONE ===" >&2
ls -la submissions/2026-05-11/ladder/*attn* submissions/2026-05-11/ladder/AT_* >&2
