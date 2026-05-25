#!/bin/bash
set -uo pipefail
cd /home/fxiao/misc/156
SUB="submissions/2026-05-11"
SUB10="submissions/2026-05-10"

EXPECTED=(
  "${SUB}/hplus_s0_split13_va.csv"
  "${SUB}/hplus_s1_split29_va.csv"
  "${SUB}/hplus_s2_split101_va.csv"
)

echo "Waiting for 3 H+ split-diversified CSVs..." >&2
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

hp_split, ids = pool_mean([
    f'{SUB11}/hplus_s0_split13_va.csv',
    f'{SUB11}/hplus_s1_split29_va.csv',
    f'{SUB11}/hplus_s2_split101_va.csv',
])
hp_split.to_csv(f'{SUB11}/hplus_split_3seed_mean.csv', index=False, float_format='%.6f')
hp_split.to_csv(f'{SUB11}/ladder/HPS_split_alone.csv', index=False, float_format='%.6f')
print('hplus split pool written')
"

uv run python view_average.py --in submissions/2026-05-11/ladder/HPS_split_alone.csv --out submissions/2026-05-11/ladder/HPS_split_alone_va.csv --force 2>&1 | tail -1
mv submissions/2026-05-11/ladder/HPS_split_alone_va.csv submissions/2026-05-11/ladder/HPS_split_alone.csv

echo "=== H+ ASSEMBLY DONE ===" >&2
