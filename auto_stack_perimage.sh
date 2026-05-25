#!/bin/bash
# Wait for per-image MV val_preds, then re-run stack_expanded with corrected MV data.
set -uo pipefail
cd /home/fxiao/misc/156

EXPECTED=(
  "/data/artifacts/frank/misc/runs/v1_3class_multiview_omnirad_s0/val_preds_perimage.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_multiview_omnirad_s0_linear/val_preds_perimage.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_multiview_omnirad_s0_sum/val_preds_perimage.npz"
)

echo "Waiting for per-image MV val_preds..." >&2
for f in "${EXPECTED[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  ready: $(basename $(dirname $f))" >&2
done
sleep 30

# Build pooled per-image MV val_preds
uv run python -c "
import numpy as np
files = [
    '/data/artifacts/frank/misc/runs/v1_3class_multiview_omnirad_s0/val_preds_perimage.npz',
    '/data/artifacts/frank/misc/runs/v1_3class_multiview_omnirad_s0_linear/val_preds_perimage.npz',
    '/data/artifacts/frank/misc/runs/v1_3class_multiview_omnirad_s0_sum/val_preds_perimage.npz',
]
ds = [np.load(f, allow_pickle=True) for f in files]
preds = np.mean([d['preds'] for d in ds], axis=0).astype(np.float32)
np.savez('/data/artifacts/frank/misc/runs/v1_3class_multiview_omnirad_pooled/val_preds_perimage.npz',
         preds=preds, paths=ds[0]['paths'], raw_labels=ds[0]['raw_labels'])
print(f'wrote per-image pooled MV val_preds: shape {preds.shape}')
"

# Patch stack_expanded.py temporarily to use perimage
cp stack_expanded.py stack_expanded_perimage.py
sed -i "s|v1_3class_multiview_omnirad_pooled/val_preds.npz|v1_3class_multiview_omnirad_pooled/val_preds_perimage.npz|" stack_expanded_perimage.py
sed -i "s|STK_expanded.csv|STK_expanded_perimage.csv|" stack_expanded_perimage.py

uv run python stack_expanded_perimage.py 2>&1 | tail -30
if [ -f submissions/2026-05-11/ladder/STK_expanded_perimage.csv ]; then
  uv run python view_average.py --in submissions/2026-05-11/ladder/STK_expanded_perimage.csv \
    --out submissions/2026-05-11/ladder/STK_expanded_perimage_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/STK_expanded_perimage_va.csv submissions/2026-05-11/ladder/STK_expanded_perimage.csv
fi
echo "=== PER-IMAGE EXPANDED STACK DONE ===" >&2
