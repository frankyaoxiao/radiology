"""Per-label ridge regression stacking over (OmniRad, ViT-H+, ConvNeXt-L) val predictions.

For each of the 9 labels, fit:
    y_label = beta_0 + beta_1 * pred_omnirad + beta_2 * pred_hplus + beta_3 * pred_cnxl

with L2 (ridge) regularization. Alpha auto-selected via 5-fold CV. Apply learned
coefficients to test predictions to produce a stacked ensemble CSV.

Usage:
    uv run python fit_ridge.py \\
        --val-omnirad /data/.../v1_3class_omnirad_b14_s0/val_preds.npz \\
        --val-hplus   /data/.../v1_3class_hplus_s0/val_preds.npz \\
        --val-cnxl    /data/.../v1_3class_cnxl_s0/val_preds.npz \\
        --test-omnirad submissions/2026-05-10/omnirad_5split_mean.csv \\
        --test-hplus   submissions/2026-05-10/dinov3_hplus.csv \\
        --test-cnxl    submissions/2026-05-10/dinov3_cnxl.csv \\
        --out-csv      submissions/2026-05-10/ensemble_ridge.csv \\
        --out-coef     ridge_coefs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from config import LABEL_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-omnirad", required=True, type=Path)
    ap.add_argument("--val-hplus", required=True, type=Path)
    ap.add_argument("--val-cnxl", required=True, type=Path)
    ap.add_argument("--test-omnirad", required=True, type=Path)
    ap.add_argument("--test-hplus", required=True, type=Path)
    ap.add_argument("--test-cnxl", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-coef", default=Path("ridge_coefs.json"), type=Path)
    ap.add_argument("--clip", action="store_true", help="clip predictions to [-1, 1]")
    args = ap.parse_args()

    LABELS = list(LABEL_NAMES)

    # ------------------------------------------------------------------
    # Load val preds (paths arrays must match — all three trained on split 42)
    # ------------------------------------------------------------------
    z_om = np.load(str(args.val_omnirad), allow_pickle=True)
    z_hp = np.load(str(args.val_hplus), allow_pickle=True)
    z_cn = np.load(str(args.val_cnxl), allow_pickle=True)

    if not (np.array_equal(z_om["paths"], z_hp["paths"]) and np.array_equal(z_om["paths"], z_cn["paths"])):
        raise RuntimeError(
            "val paths differ across models. All three must be trained with split_seed=42 "
            "and run val inference on the same split."
        )

    pred_om = z_om["preds"]
    pred_hp = z_hp["preds"]
    pred_cn = z_cn["preds"]
    labels  = z_om["raw_labels"]  # (N, 9), NaN for blanks

    # ------------------------------------------------------------------
    # Fit ridge per label
    # ------------------------------------------------------------------
    alphas = np.logspace(-3, 3, 13)
    coefs = {}

    print(f"Fitting ridge per label on {pred_om.shape[0]} val samples...\n")
    print(f"{'Label':30s}  {'alpha':>7s}  {'beta_om':>8s}  {'beta_hp':>8s}  {'beta_cn':>8s}  {'intc':>7s}  "
          f"{'ridge_nmse':>11s}  {'equal_nmse':>11s}  {'Δ':>7s}")
    print("-" * 120)

    for j, lab in enumerate(LABELS):
        y = labels[:, j]
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            coefs[lab] = {"intercept": 0.0, "weights": [1.0/3, 1.0/3, 1.0/3], "alpha": None}
            continue
        X = np.stack([pred_om[mask, j], pred_hp[mask, j], pred_cn[mask, j]], axis=1)
        y_clean = y[mask]

        ridge = RidgeCV(alphas=alphas, fit_intercept=True, cv=5)
        ridge.fit(X, y_clean)

        # Diagnostic: compare ridge fit to equal-weight ensemble
        var = float(np.var(y_clean))
        if var <= 0:
            coefs[lab] = {"intercept": float(ridge.intercept_), "weights": list(map(float, ridge.coef_)), "alpha": float(ridge.alpha_)}
            continue
        ridge_pred = ridge.predict(X)
        ridge_nmse = float(np.mean((y_clean - ridge_pred) ** 2) / var)
        eq_pred = X.mean(axis=1)
        eq_nmse = float(np.mean((y_clean - eq_pred) ** 2) / var)

        coefs[lab] = {
            "intercept": float(ridge.intercept_),
            "weights": [float(w) for w in ridge.coef_],
            "alpha": float(ridge.alpha_),
        }
        print(f"{lab:30s}  {ridge.alpha_:7.3f}  {ridge.coef_[0]:+8.3f}  {ridge.coef_[1]:+8.3f}  {ridge.coef_[2]:+8.3f}  "
              f"{ridge.intercept_:+7.3f}  {ridge_nmse:11.4f}  {eq_nmse:11.4f}  {eq_nmse - ridge_nmse:+7.4f}")

    args.out_coef.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_coef, "w") as f:
        json.dump(coefs, f, indent=2)
    print(f"\nwrote {args.out_coef}")

    # ------------------------------------------------------------------
    # Apply to test predictions
    # ------------------------------------------------------------------
    test_om = pd.read_csv(args.test_omnirad).sort_values("Id").reset_index(drop=True)
    test_hp = pd.read_csv(args.test_hplus).sort_values("Id").reset_index(drop=True)
    test_cn = pd.read_csv(args.test_cnxl).sort_values("Id").reset_index(drop=True)
    ids = test_om["Id"].to_numpy()
    if not (np.array_equal(test_hp["Id"].to_numpy(), ids) and np.array_equal(test_cn["Id"].to_numpy(), ids)):
        raise RuntimeError("test CSV Id columns mismatch")

    out = pd.DataFrame({"Id": ids})
    for lab in LABELS:
        c = coefs[lab]
        pred = (
            c["intercept"]
            + c["weights"][0] * test_om[lab].to_numpy()
            + c["weights"][1] * test_hp[lab].to_numpy()
            + c["weights"][2] * test_cn[lab].to_numpy()
        )
        if args.clip:
            pred = np.clip(pred, -1.0, 1.0)
        out[lab] = pred

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, float_format="%.6f")
    print(f"wrote {args.out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
