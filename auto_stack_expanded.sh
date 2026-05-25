#!/bin/bash
set -uo pipefail
cd /home/fxiao/misc/156

# Wait until we have enough new pools. Just poll periodically and re-run.
echo "expanded stack watcher starting..." >&2

EXPECTED_NEW=(
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s0_ema/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s1_ema/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s2_ema/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s0/val_preds_swa.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s1/val_preds_swa.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s2/val_preds_swa.npz"
)
for f in "${EXPECTED_NEW[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  ready: $(basename $(dirname $f))" >&2
done
sleep 30

uv run python stack_expanded.py 2>&1 | tail -30
if [ -f submissions/2026-05-11/ladder/STK_expanded.csv ]; then
  uv run python view_average.py --in submissions/2026-05-11/ladder/STK_expanded.csv \
    --out submissions/2026-05-11/ladder/STK_expanded_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/STK_expanded_va.csv submissions/2026-05-11/ladder/STK_expanded.csv
fi
echo "=== EXPANDED STACK DONE ===" >&2
