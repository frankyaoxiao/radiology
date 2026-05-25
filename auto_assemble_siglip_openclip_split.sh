#!/bin/bash
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"

EXPECTED=(
  "${SUB}/siglip2_p14_384_s0_split13_va.csv"
  "${SUB}/siglip2_p14_384_s1_split29_va.csv"
  "${SUB}/siglip2_p14_384_s2_split101_va.csv"
  "${SUB}/openclip_s0_split13_va.csv"
  "${SUB}/openclip_s1_split29_va.csv"
  "${SUB}/openclip_s2_split101_va.csv"
)

echo "Waiting for 6 SigLIP+OpenCLIP split CSVs..." >&2
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

sg_split, ids = pool_mean([
    f'{SUB11}/siglip2_p14_384_s0_split13_va.csv',
    f'{SUB11}/siglip2_p14_384_s1_split29_va.csv',
    f'{SUB11}/siglip2_p14_384_s2_split101_va.csv',
])
sg_split.to_csv(f'{SUB11}/siglip2_split_3seed_mean.csv', index=False, float_format='%.6f')
sg_split.to_csv(f'{SUB11}/ladder/SGS_split_alone.csv', index=False, float_format='%.6f')

oc_split, _ = pool_mean([
    f'{SUB11}/openclip_s0_split13_va.csv',
    f'{SUB11}/openclip_s1_split29_va.csv',
    f'{SUB11}/openclip_s2_split101_va.csv',
])
oc_split.to_csv(f'{SUB11}/openclip_split_3seed_mean.csv', index=False, float_format='%.6f')
oc_split.to_csv(f'{SUB11}/ladder/OCS_split_alone.csv', index=False, float_format='%.6f')
print('siglip+openclip split pools written')
"

for v in SGS_split_alone OCS_split_alone; do
  uv run python view_average.py --in submissions/2026-05-11/ladder/${v}.csv --out submissions/2026-05-11/ladder/${v}_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/${v}_va.csv submissions/2026-05-11/ladder/${v}.csv
done
echo "=== SIGLIP+OPENCLIP ASSEMBLY DONE ===" >&2
