#!/bin/bash
# Wait for ALL val_preds.npz to land, then run stacking analysis.
set -uo pipefail
cd /home/fxiao/misc/156

EXPECTED=(
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s0/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s1/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s2/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_cnxl_s0_aug_trivial/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_cnxl_s1_aug_trivial/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_cnxl_s2_aug_trivial/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_hplus_s0_aug_trivial/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_hplus_s1_aug_trivial/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_hplus_s2_aug_trivial/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_siglip2_p14_384_s0/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_siglip2_p14_384_s1/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_siglip2_p14_384_s2/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_openclip_s0/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_openclip_s1/val_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_openclip_s2/val_preds.npz"
)

echo "Waiting for ${#EXPECTED[@]} val_preds.npz files..." >&2
for f in "${EXPECTED[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  $(date -Iseconds): $(basename $(dirname $f))" >&2
done
sleep 30

uv run python stack_per_label.py 2>&1
uv run python view_average.py --in submissions/2026-05-11/ladder/STK_optimal_stacked.csv \
  --out submissions/2026-05-11/ladder/STK_optimal_stacked_va.csv --force 2>&1 | tail -1
mv submissions/2026-05-11/ladder/STK_optimal_stacked_va.csv submissions/2026-05-11/ladder/STK_optimal_stacked.csv
echo "=== LINEAR STACKING DONE ===" >&2

# Also try XGBoost stacking (non-linear interactions)
uv run --with xgboost python stack_xgb.py 2>&1 | tail -30
if [ -f submissions/2026-05-11/ladder/STKX_xgb_stacked.csv ]; then
  uv run python view_average.py --in submissions/2026-05-11/ladder/STKX_xgb_stacked.csv \
    --out submissions/2026-05-11/ladder/STKX_xgb_stacked_va.csv --force 2>&1 | tail -1
  mv submissions/2026-05-11/ladder/STKX_xgb_stacked_va.csv submissions/2026-05-11/ladder/STKX_xgb_stacked.csv
fi
echo "=== XGBOOST STACKING DONE ===" >&2
