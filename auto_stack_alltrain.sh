#!/bin/bash
set -uo pipefail
cd /home/fxiao/misc/156

EXPECTED=(
  "/data/artifacts/frank/misc/runs/v1_3class_omnirad_b14_s0/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s0/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s1/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_eva02_s2/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_cnxl_s0_aug_trivial/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_cnxl_s1_aug_trivial/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_cnxl_s2_aug_trivial/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_hplus_s0_aug_trivial/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_hplus_s1_aug_trivial/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_hplus_s2_aug_trivial/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_siglip2_p14_384_s0/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_siglip2_p14_384_s1/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_siglip2_p14_384_s2/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_openclip_s0/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_openclip_s1/train_preds.npz"
  "/data/artifacts/frank/misc/runs/v1_3class_openclip_s2/train_preds.npz"
)

echo "Waiting for ${#EXPECTED[@]} train_preds..." >&2
for f in "${EXPECTED[@]}"; do
  until [ -f "$f" ]; do sleep 60; done
  echo "  ready: $(basename $(dirname $f))" >&2
done
sleep 30

echo "=== MLP stacker ===" >&2
uv run python stack_mlp_train.py 2>&1 | tail -25

echo "=== Ridge stacker ===" >&2
uv run python stack_ridge_train.py 2>&1 | tail -25

echo "=== XGBoost stacker ===" >&2
uv run --with xgboost python stack_xgb_train.py 2>&1 | tail -25

for v in STK_mlp_trainfit STK_ridge_trainfit STK_xgb_trainfit; do
  f="submissions/2026-05-16/ladder/${v}.csv"
  if [ -f "$f" ]; then
    uv run python view_average.py --in "$f" --out "submissions/2026-05-16/ladder/${v}_va.csv" --force 2>&1 | tail -1
    mv "submissions/2026-05-16/ladder/${v}_va.csv" "$f"
  fi
done
echo "=== ALL TRAIN STACKERS DONE ===" >&2
